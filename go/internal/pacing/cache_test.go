package pacing

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

func TestDefaultBeforeRefresh(t *testing.T) {
	c := New(func(context.Context) (map[int64]float64, error) {
		return map[int64]float64{1: 0.5}, nil
	}, 1.0)
	if got := c.Get(1); got != 1.0 {
		t.Errorf("before refresh want default 1.0, got %v", got)
	}
	if c.Staleness() != -1 {
		t.Error("staleness should be -1 before first refresh")
	}
}

func TestRefreshSwapsSnapshot(t *testing.T) {
	c := New(func(context.Context) (map[int64]float64, error) {
		return map[int64]float64{1: 0.5, 2: 0.25}, nil
	}, 1.0)
	if err := c.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if c.Get(1) != 0.5 || c.Get(2) != 0.25 {
		t.Error("expected refreshed values")
	}
	if c.Get(999) != 1.0 {
		t.Error("absent campaign should get default")
	}
	if c.Staleness() < 0 || c.Staleness() > time.Second {
		t.Errorf("unexpected staleness %v", c.Staleness())
	}
}

// TestRunRefreshesAndBoundsStaleness checks the background loop keeps staleness
// under the refresh interval (plus slack) and picks up updated values.
func TestRunRefreshesAndBoundsStaleness(t *testing.T) {
	var val atomic.Int64
	val.Store(100) // multiplier * 1000
	c := New(func(context.Context) (map[int64]float64, error) {
		return map[int64]float64{1: float64(val.Load()) / 1000.0}, nil
	}, 1.0)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	interval := 20 * time.Millisecond
	go c.Run(ctx, interval, nil)

	time.Sleep(30 * time.Millisecond)
	if got := c.Get(1); got != 0.1 {
		t.Errorf("want 0.1, got %v", got)
	}
	val.Store(200) // change the source
	time.Sleep(2 * interval)
	if got := c.Get(1); got != 0.2 {
		t.Errorf("want updated 0.2, got %v", got)
	}
	if s := c.Staleness(); s > 3*interval {
		t.Errorf("staleness %v exceeds bound", s)
	}
}

// TestConcurrentReadsDuringRefresh must be race-clean under -race.
func TestConcurrentReadsDuringRefresh(t *testing.T) {
	flip := atomic.Int64{}
	c := New(func(context.Context) (map[int64]float64, error) {
		return map[int64]float64{1: float64(flip.Add(1)%2) + 0.5}, nil
	}, 1.0)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go c.Run(ctx, time.Millisecond, nil)

	done := time.After(50 * time.Millisecond)
	for {
		select {
		case <-done:
			return
		default:
			_ = c.Get(1) // hammer the hot path while refreshes swap snapshots
		}
	}
}
