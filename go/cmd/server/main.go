// Command server serves the bid hot path.
//
//	server -addr :8080 -redis localhost:6379 \
//	       -model artifacts/model.onnx -transform artifacts/feature_transform.json \
//	       -mode bidshade -reserve 0.01 -refresh 5s
//
// The pacing-multiplier cache refreshes from Redis in the background; the hot
// path reads it lock-free. Budget charges hit Redis via the atomic Lua CAS.
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"net/http/pprof"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/ericwan025/pacer/go/internal/bid"
	"github.com/ericwan025/pacer/go/internal/budget"
	"github.com/ericwan025/pacer/go/internal/model"
	"github.com/ericwan025/pacer/go/internal/pacing"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	redisAddr := flag.String("redis", "localhost:6379", "redis address")
	modelPath := flag.String("model", "artifacts/model.onnx", "ONNX model path")
	transformPath := flag.String("transform", "artifacts/feature_transform.json", "feature transform JSON")
	reserve := flag.Float64("reserve", 0.01, "auction reserve price")
	modeStr := flag.String("mode", "bidshade", "pacing mode: bidshade|throttle")
	refresh := flag.Duration("refresh", 5*time.Second, "multiplier cache refresh interval")
	enablePprof := flag.Bool("pprof", false, "expose /debug/pprof for CPU profiling")
	flag.Parse()

	ctx := context.Background()

	rdb := redis.NewClient(&redis.Options{Addr: *redisAddr})
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Fatalf("redis unreachable at %s: %v", *redisAddr, err)
	}
	store := budget.New(rdb)

	transform, err := model.LoadTransform(*transformPath)
	if err != nil {
		log.Fatalf("load transform: %v", err)
	}
	nFields := len(transform.FieldOrder())

	scorer, err := model.LoadModel(*modelPath, nFields)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}

	cache := pacing.New(store.AllMultipliers, 1.0)
	errCh := make(chan error, 16)
	go cache.Run(ctx, *refresh, errCh)
	go func() {
		for e := range errCh {
			log.Printf("cache refresh error: %v", e)
		}
	}()

	mode := bid.BidShade
	if *modeStr == "throttle" {
		mode = bid.Throttle
	}
	metrics := bid.NewMetrics()
	metrics.SetStalenessSource(cache.Staleness)
	handler := bid.NewHandler(transform, scorer, cache, store, *reserve, mode, metrics)

	mux := http.NewServeMux()
	mux.Handle("/bid", handler)
	mux.HandleFunc("/healthz", bid.HealthzHandler)
	mux.HandleFunc("/metrics", metrics.MetricsHandler)
	if *enablePprof {
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
		log.Printf("pprof enabled at /debug/pprof/")
	}

	log.Printf("pacer server listening on %s (mode=%s, fields=%d, refresh=%s)",
		*addr, *modeStr, nFields, *refresh)
	srv := &http.Server{Addr: *addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	log.Fatal(srv.ListenAndServe())
}
