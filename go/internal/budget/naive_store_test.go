//go:build naive_overspend

package budget

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
)

// TestNaiveOverspends documents the failure mode: the non-atomic charge lets many
// concurrent requests overspend a budget sized for 100. This test is EXPECTED to
// show overspend and is only built under `naive_overspend`.
//
//	go test -tags naive_overspend -run TestNaiveOverspends ./internal/budget/
func TestNaiveOverspends(t *testing.T) {
	ctx := context.Background()
	rdb := testClient(t)
	s := New(rdb)

	const budget = 100.0
	if err := s.SetCampaign(ctx, 2, budget, 1.0); err != nil {
		t.Fatal(err)
	}

	var granted int64
	var wg sync.WaitGroup
	const n = 500
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			if ok, _ := s.NaiveCharge(ctx, 2, 1.0); ok {
				atomic.AddInt64(&granted, 1)
			}
		}()
	}
	wg.Wait()

	spend, _ := s.Spend(ctx, 2)
	if spend <= budget {
		t.Fatalf("expected the naive store to OVERSPEND, but spend=%.1f <= %.1f "+
			"(race window did not open — flaky env?)", spend, budget)
	}
	t.Logf("naive store overspent as expected: granted=%d spend=%.1f budget=%.1f",
		granted, spend, budget)
}
