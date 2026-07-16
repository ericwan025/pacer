// Command controller runs the PID pacing loop as a separate service.
//
// Every -dt it reads each campaign's spend from Redis, computes the budget-
// normalized pacing error against the target curve, updates that campaign's PID
// controller, and writes the new multiplier back to Redis, where the bid server's
// cache picks it up on its next refresh. This mirrors real architecture: pacing
// control is decoupled from bid serving.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/ericwan025/pacer/go/internal/budget"
	"github.com/ericwan025/pacer/go/internal/controller"
)

type profile struct {
	HourlyVolume []float64 `json:"hourly_volume"`
}

func main() {
	redisAddr := flag.String("redis", "localhost:6379", "redis address")
	dt := flag.Duration("dt", 10*time.Second, "control interval")
	kp := flag.Float64("kp", 1.0, "proportional gain")
	ki := flag.Float64("ki", 0.1, "integral gain")
	kd := flag.Float64("kd", 0.0, "derivative gain")
	maxMult := flag.Float64("max-mult", 1.0, "multiplier clamp upper bound")
	profilePath := flag.String("profile", "", "traffic profile JSON (hourly_volume); empty = uniform")
	daySeconds := flag.Float64("day", 86400, "length of the pacing day in seconds")
	flag.Parse()

	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{Addr: *redisAddr})
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Fatalf("redis unreachable: %v", err)
	}
	store := budget.New(rdb)

	var curve controller.TargetCurve = controller.Uniform{Horizon: *daySeconds}
	if *profilePath != "" {
		b, err := os.ReadFile(*profilePath)
		if err != nil {
			log.Fatalf("read profile: %v", err)
		}
		var p profile
		if err := json.Unmarshal(b, &p); err != nil {
			log.Fatalf("parse profile: %v", err)
		}
		curve = controller.NewTrafficAware(p.HourlyVolume)
	}

	cfg := controller.Config{
		Kp: *kp, Ki: *ki, Kd: *kd, Dt: dt.Seconds(),
		MinMult: 0.0, MaxMult: *maxMult, DerivTau: 30.0,
		AntiWindup: true, Mapping: "linear",
	}
	pids := map[int64]*controller.PID{}

	start := time.Now()
	log.Printf("controller started (dt=%s, kp=%g ki=%g kd=%g, max=%g)",
		*dt, *kp, *ki, *kd, *maxMult)
	ticker := time.NewTicker(*dt)
	defer ticker.Stop()
	for range ticker.C {
		elapsed := time.Since(start).Seconds()
		setpoint := curve.Fraction(elapsed)
		ids, err := rdb.SMembers(ctx, "campaigns").Result()
		if err != nil {
			log.Printf("registry read: %v", err)
			continue
		}
		for _, s := range ids {
			id, budgetVal, spend, ok := readCampaign(ctx, rdb, s)
			if !ok || budgetVal <= 0 {
				continue
			}
			pid, exists := pids[id]
			if !exists {
				pid = controller.New(cfg)
				pids[id] = pid
			}
			mult := pid.Update(setpoint, spend/budgetVal)
			if err := store.SetMultiplier(ctx, id, mult); err != nil {
				log.Printf("set multiplier %d: %v", id, err)
			}
		}
		if elapsed >= *daySeconds {
			log.Printf("reached end of pacing day; exiting")
			return
		}
	}
}

func readCampaign(ctx context.Context, rdb *redis.Client, idStr string) (int64, float64, float64, bool) {
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		return 0, 0, 0, false
	}
	vals, err := rdb.HMGet(ctx, "campaign:"+idStr, "budget", "spend").Result()
	if err != nil || len(vals) != 2 {
		return 0, 0, 0, false
	}
	return id, toFloat(vals[0]), toFloat(vals[1]), true
}

func toFloat(v any) float64 {
	s, ok := v.(string)
	if !ok {
		return 0
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	return f
}
