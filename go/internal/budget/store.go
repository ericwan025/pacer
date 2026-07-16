// Package budget is the Redis-backed budget store.
//
// The critical correctness problem: at high concurrency a naive
// read-check-decrement lets campaigns overspend, because two requests can both
// read spend=99, both see room for 1 more, and both charge — spending past the
// budget. We solve it with a Lua script that does the check AND the decrement
// atomically inside Redis, so only one of the racing requests wins the last
// dollar. See internal/budget/naive for the version that fails this on purpose.
package budget

import (
	"context"
	"fmt"
	"strconv"

	"github.com/redis/go-redis/v9"
)

// chargeScript atomically charges cost to a campaign iff it stays within budget.
// Returns 1 if charged, 0 if rejected. Fields live in a per-campaign hash.
var chargeScript = redis.NewScript(`
local spend  = tonumber(redis.call('HGET', KEYS[1], 'spend')  or '0')
local budget = tonumber(redis.call('HGET', KEYS[1], 'budget') or '0')
local cost   = tonumber(ARGV[1])
if spend + cost <= budget then
  redis.call('HINCRBYFLOAT', KEYS[1], 'spend', cost)
  return 1
end
return 0
`)

type Store struct {
	rdb *redis.Client
}

func New(rdb *redis.Client) *Store { return &Store{rdb: rdb} }

func key(campaignID int64) string { return fmt.Sprintf("campaign:%d", campaignID) }

const registryKey = "campaigns" // set of known campaign ids

// SetCampaign initializes budget + multiplier and resets spend to 0, and records
// the id in the campaign registry so the multiplier table can be bulk-loaded.
func (s *Store) SetCampaign(ctx context.Context, id int64, budget, multiplier float64) error {
	pipe := s.rdb.TxPipeline()
	pipe.HSet(ctx, key(id), map[string]any{
		"budget":     budget,
		"multiplier": multiplier,
		"spend":      0.0,
	})
	pipe.SAdd(ctx, registryKey, id)
	_, err := pipe.Exec(ctx)
	return err
}

// AllMultipliers bulk-loads the multiplier for every registered campaign. Used by
// the in-process pacing cache's background refresh.
func (s *Store) AllMultipliers(ctx context.Context) (map[int64]float64, error) {
	ids, err := s.rdb.SMembers(ctx, registryKey).Result()
	if err != nil {
		return nil, err
	}
	out := make(map[int64]float64, len(ids))
	pipe := s.rdb.Pipeline()
	cmds := make(map[int64]*redis.StringCmd, len(ids))
	parsed := make([]int64, 0, len(ids))
	for _, s := range ids {
		id, perr := strconv.ParseInt(s, 10, 64)
		if perr != nil {
			continue
		}
		parsed = append(parsed, id)
		cmds[id] = pipe.HGet(ctx, key(id), "multiplier")
	}
	if _, err := pipe.Exec(ctx); err != nil && err != redis.Nil {
		return nil, err
	}
	for _, id := range parsed {
		if v, err := cmds[id].Float64(); err == nil {
			out[id] = v
		}
	}
	return out, nil
}

// Charge attempts to spend cost. Returns true iff the charge fit within budget.
// Atomic: no overspend regardless of concurrency.
func (s *Store) Charge(ctx context.Context, id int64, cost float64) (bool, error) {
	res, err := chargeScript.Run(ctx, s.rdb, []string{key(id)}, cost).Int()
	if err != nil {
		return false, err
	}
	return res == 1, nil
}

func (s *Store) Spend(ctx context.Context, id int64) (float64, error) {
	return s.rdb.HGet(ctx, key(id), "spend").Float64()
}

func (s *Store) Multiplier(ctx context.Context, id int64) (float64, error) {
	return s.rdb.HGet(ctx, key(id), "multiplier").Float64()
}

func (s *Store) SetMultiplier(ctx context.Context, id int64, m float64) error {
	return s.rdb.HSet(ctx, key(id), "multiplier", m).Err()
}
