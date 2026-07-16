//go:build naive_overspend

// This file is DELIBERATELY BROKEN and kept as documentation. It is compiled only
// under the `naive_overspend` build tag, so it never ships in the real binary and
// its failing test never runs in normal CI.
//
// It shows the bug the Lua script in store.go exists to prevent: a
// read-check-decrement done as separate Redis round trips has a window between
// the check and the decrement in which another request can also pass the check.
// Under concurrency, campaigns overspend.
package budget

import (
	"context"
	"time"
)

// NaiveCharge is the WRONG way: HGET spend, check in Go, then HINCRBYFLOAT. The
// check and the decrement are not atomic. The Sleep widens the race window so the
// failure is deterministic for the documentation test; even without it the race
// exists — the sleep just makes it always visible instead of occasionally.
func (s *Store) NaiveCharge(ctx context.Context, id int64, cost float64) (bool, error) {
	k := key(id)
	spend, err := s.rdb.HGet(ctx, k, "spend").Float64()
	if err != nil {
		return false, err
	}
	budget, err := s.rdb.HGet(ctx, k, "budget").Float64()
	if err != nil {
		return false, err
	}
	if spend+cost <= budget {
		time.Sleep(50 * time.Microsecond) // the check-then-act window
		if err := s.rdb.HIncrByFloat(ctx, k, "spend", cost).Err(); err != nil {
			return false, err
		}
		return true, nil
	}
	return false, nil
}
