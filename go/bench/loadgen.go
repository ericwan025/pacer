// Command loadgen is a closed-loop HTTP load generator for the bid server.
//
// A fixed pool of connections hammers POST /bid for a warm-up period (discarded)
// then a measurement period. It reports sustained QPS and p50/p90/p99/p999
// latency. Run it twice with different -mult to compare the two cost profiles:
//
//	filter-only path : -mult 0  (throttled out: cache + throttle draw, no inference)
//	full path        : -mult 1  (participates: transform + ONNX inference + charge)
//
// The server must run in throttle mode with a campaign whose multiplier the
// -mult flag matches (seed it via bench's setup).
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

type candidate struct {
	ID    int64   `json:"id"`
	Value float64 `json:"value"`
}
type request struct {
	Features   map[string]string `json:"features"`
	Candidates []candidate       `json:"candidates"`
}

func main() {
	url := flag.String("url", "http://localhost:8080/bid", "bid endpoint")
	conns := flag.Int("conns", 64, "concurrent connections")
	warmup := flag.Duration("warmup", 3*time.Second, "warm-up (discarded)")
	dur := flag.Duration("duration", 10*time.Second, "measurement window")
	campaignID := flag.Int64("campaign", 1, "candidate campaign id")
	label := flag.String("label", "full", "path label for the report")
	flag.Parse()

	body, _ := json.Marshal(request{
		Features:   sampleFeatures(),
		Candidates: []candidate{{ID: *campaignID, Value: 10.0}},
	})

	client := &http.Client{
		Timeout: 5 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        *conns * 2,
			MaxIdleConnsPerHost: *conns * 2,
			MaxConnsPerHost:     *conns * 2,
		},
	}

	var measuring atomic.Bool
	var wg sync.WaitGroup
	latencies := make([][]time.Duration, *conns)
	var errs atomic.Int64
	stop := make(chan struct{})

	for w := 0; w < *conns; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			local := make([]time.Duration, 0, 1<<16)
			for {
				select {
				case <-stop:
					latencies[id] = local
					return
				default:
				}
				t0 := time.Now()
				resp, err := client.Post(*url, "application/json", bytes.NewReader(body))
				if err != nil {
					errs.Add(1)
					continue
				}
				// fully drain before close so the keep-alive connection is reused
				_, _ = io.Copy(io.Discard, resp.Body)
				_ = resp.Body.Close()
				d := time.Since(t0)
				if measuring.Load() {
					local = append(local, d)
				}
			}
		}(w)
	}

	time.Sleep(*warmup)
	measuring.Store(true)
	start := time.Now()
	time.Sleep(*dur)
	elapsed := time.Since(start)
	measuring.Store(false)
	close(stop)
	wg.Wait()

	var all []time.Duration
	for _, l := range latencies {
		all = append(all, l...)
	}
	report(*label, all, elapsed, errs.Load(), *conns)
}

func report(label string, lat []time.Duration, elapsed time.Duration, errs int64, conns int) {
	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
	n := len(lat)
	qps := float64(n) / elapsed.Seconds()
	fmt.Printf("=== %s path ===\n", label)
	fmt.Printf("hardware: %s/%s, %d CPUs, GOMAXPROCS=%d\n",
		runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), runtime.GOMAXPROCS(0))
	fmt.Printf("conns=%d requests=%d errors=%d window=%.1fs\n", conns, n, errs, elapsed.Seconds())
	fmt.Printf("QPS: %.0f\n", qps)
	if n > 0 {
		fmt.Printf("p50=%s p90=%s p99=%s p999=%s max=%s\n",
			pct(lat, 0.50), pct(lat, 0.90), pct(lat, 0.99), pct(lat, 0.999), lat[n-1])
	}
	// machine-readable line for the report aggregator
	enc := json.NewEncoder(os.Stdout)
	_ = enc.Encode(map[string]any{
		"label": label, "qps": qps, "requests": n, "errors": errs,
		"p50_us": us(pct(lat, 0.5)), "p90_us": us(pct(lat, 0.9)),
		"p99_us": us(pct(lat, 0.99)), "p999_us": us(pct(lat, 0.999)),
	})
}

func pct(sorted []time.Duration, q float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	i := int(q * float64(len(sorted)-1))
	return sorted[i]
}

func us(d time.Duration) int64 { return d.Microseconds() }

func sampleFeatures() map[string]string {
	return map[string]string{
		"hour": "14102108", "C1": "1005", "banner_pos": "0", "site_category": "a",
		"app_category": "x", "device_type": "1", "device_conn_type": "2",
		"device_id": "dev1", "device_ip": "ip1", "site_id": "s1", "site_domain": "sd1",
		"app_id": "a1", "app_domain": "ad1", "device_model": "m1",
		"C14": "100", "C15": "320", "C16": "50", "C17": "1", "C18": "0",
		"C19": "35", "C20": "-1", "C21": "79",
	}
}
