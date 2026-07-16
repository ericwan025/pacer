package bid

import (
	"fmt"
	"net/http"
	"sync/atomic"
	"time"
)

// Metrics holds hot-path counters. Prometheus-style text exposition on /metrics.
type Metrics struct {
	Requests  atomic.Int64
	Wins      atomic.Int64
	Throttled atomic.Int64
	// latency histogram in fixed microsecond buckets
	buckets []float64
	counts  []atomic.Int64
	// staleness gauge source (set by the server)
	staleness func() time.Duration
}

func NewMetrics() *Metrics {
	b := []float64{50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000}
	return &Metrics{buckets: b, counts: make([]atomic.Int64, len(b)+1)}
}

// SetStalenessSource wires a cache-staleness gauge into /metrics.
func (m *Metrics) SetStalenessSource(f func() time.Duration) { m.staleness = f }

// ObserveLatency records a request latency in microseconds.
func (m *Metrics) ObserveLatency(d time.Duration) {
	us := float64(d.Microseconds())
	for i, edge := range m.buckets {
		if us <= edge {
			m.counts[i].Add(1)
			return
		}
	}
	m.counts[len(m.buckets)].Add(1)
}

// HealthzHandler always reports OK (liveness).
func HealthzHandler(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

// MetricsHandler renders the counters in Prometheus text format.
func (m *Metrics) MetricsHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, "pacer_bid_requests_total %d\n", m.Requests.Load())
	fmt.Fprintf(w, "pacer_bid_wins_total %d\n", m.Wins.Load())
	fmt.Fprintf(w, "pacer_bid_throttled_total %d\n", m.Throttled.Load())

	var cumulative int64
	for i, edge := range m.buckets {
		cumulative += m.counts[i].Load()
		fmt.Fprintf(w, "pacer_bid_latency_us_bucket{le=\"%g\"} %d\n", edge, cumulative)
	}
	cumulative += m.counts[len(m.buckets)].Load()
	fmt.Fprintf(w, "pacer_bid_latency_us_bucket{le=\"+Inf\"} %d\n", cumulative)

	if m.staleness != nil {
		fmt.Fprintf(w, "pacer_cache_staleness_seconds %g\n", m.staleness().Seconds())
	}
}
