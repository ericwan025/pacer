// Package throttle implements the hot-path participation decision.
//
// Given a participation probability p from the controller, a request participates
// iff a fast PRNG draw is below p. We use math/rand/v2 (PCG), NOT crypto/rand:
// this is on the critical path and cryptographic randomness would be pointless
// overhead. This mirrors the Python UniformThrottle; the statistical properties
// match even though the exact streams differ.
package throttle

import "math/rand/v2"

// Throttle holds a fast PRNG. Not safe for concurrent use by multiple goroutines;
// give each worker its own, or guard it.
type Throttle struct {
	rng *rand.Rand
}

// New seeds a throttle deterministically.
func New(seed uint64) *Throttle {
	return &Throttle{rng: rand.New(rand.NewPCG(seed, seed^0x9E3779B97F4A7C15))}
}

// Participate returns true with probability p. p<=0 never participates, p>=1
// always participates.
func (t *Throttle) Participate(p float64) bool {
	if p <= 0 {
		return false
	}
	if p >= 1 {
		return true
	}
	return t.rng.Float64() < p
}
