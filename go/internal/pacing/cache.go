// Package pacing is the in-process pacing-multiplier cache.
//
// A Redis round trip per bid request would be a latency and throughput ceiling.
// Instead we cache the whole multiplier table in-process and refresh it
// asynchronously from a background goroutine every RefreshInterval. The hot path
// reads a snapshot through an atomic pointer, so it NEVER takes a lock.
//
// The tradeoff: the hot path reads a slightly stale multiplier. That is correct
// and defensible — the PID controller itself runs on a ~10s interval, so a 5s-
// stale multiplier is well inside the control loop's own resolution. A bounded
// staleness beats per-request Redis latency.
package pacing

import (
	"context"
	"sync/atomic"
	"time"
)

// Loader fetches the full multiplier table. Decoupled from Redis for testing.
type Loader func(ctx context.Context) (map[int64]float64, error)

type Cache struct {
	snapshot    atomic.Pointer[map[int64]float64]
	lastRefresh atomic.Int64 // unix nanos of last successful load
	defaultMult float64
	load        Loader
}

func New(load Loader, defaultMult float64) *Cache {
	c := &Cache{defaultMult: defaultMult, load: load}
	empty := map[int64]float64{}
	c.snapshot.Store(&empty)
	return c
}

// Get returns the cached multiplier for a campaign, or the default if absent.
// Lock-free: a single atomic load plus a map read on an immutable snapshot.
func (c *Cache) Get(id int64) float64 {
	m := *c.snapshot.Load()
	if v, ok := m[id]; ok {
		return v
	}
	return c.defaultMult
}

// Refresh loads the table once and atomically swaps it in.
func (c *Cache) Refresh(ctx context.Context) error {
	m, err := c.load(ctx)
	if err != nil {
		return err
	}
	c.snapshot.Store(&m)
	c.lastRefresh.Store(time.Now().UnixNano())
	return nil
}

// Staleness reports how long since the last successful refresh.
func (c *Cache) Staleness() time.Duration {
	last := c.lastRefresh.Load()
	if last == 0 {
		return -1 // never refreshed
	}
	return time.Since(time.Unix(0, last))
}

// Run refreshes immediately, then every interval until ctx is cancelled. Refresh
// errors are reported on errCh (which may be nil) and do not stop the loop — a
// failed refresh just keeps serving the previous snapshot.
func (c *Cache) Run(ctx context.Context, interval time.Duration, errCh chan<- error) {
	if err := c.Refresh(ctx); err != nil && errCh != nil {
		errCh <- err
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if err := c.Refresh(ctx); err != nil && errCh != nil {
				errCh <- err
			}
		}
	}
}
