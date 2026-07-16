package budget

import (
	"context"
	"os"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/redis/go-redis/v9"
)

// redisAddr returns the test Redis address, skipping the test if unreachable.
func testClient(t *testing.T) *redis.Client {
	t.Helper()
	addr := os.Getenv("REDIS_ADDR")
	if addr == "" {
		addr = "localhost:6379"
	}
	rdb := redis.NewClient(&redis.Options{Addr: addr})
	if err := rdb.Ping(context.Background()).Err(); err != nil {
		t.Skipf("redis not reachable at %s (run `docker compose up -d redis`): %v", addr, err)
	}
	return rdb
}

// TestNoOverspendUnderConcurrency fires 10k concurrent charges of 1.0 at a
// campaign with budget for exactly 100. Exactly 100 must succeed and final spend
// must never exceed budget. Run with -race.
func TestNoOverspendUnderConcurrency(t *testing.T) {
	ctx := context.Background()
	rdb := testClient(t)
	s := New(rdb)

	const budget = 100.0
	const cost = 1.0
	const n = 10_000
	if err := s.SetCampaign(ctx, 1, budget, 1.0); err != nil {
		t.Fatal(err)
	}

	var granted int64
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			ok, err := s.Charge(ctx, 1, cost)
			if err != nil {
				t.Error(err)
				return
			}
			if ok {
				atomic.AddInt64(&granted, 1)
			}
		}()
	}
	wg.Wait()

	if granted != 100 {
		t.Errorf("granted=%d, want exactly 100", granted)
	}
	spend, err := s.Spend(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	if spend > budget {
		t.Errorf("OVERSPEND: spend=%.2f > budget=%.2f", spend, budget)
	}
}
