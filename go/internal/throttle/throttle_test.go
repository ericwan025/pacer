package throttle

import (
	"math"
	"testing"
)

func TestBoundaries(t *testing.T) {
	th := New(1)
	if th.Participate(0) {
		t.Fatal("p=0 must never participate")
	}
	if !th.Participate(1) {
		t.Fatal("p=1 must always participate")
	}
}

func TestRealizedRateWithin3SE(t *testing.T) {
	for _, p := range []float64{0.05, 0.37, 0.5, 0.83} {
		th := New(42)
		const n = 100_000
		hits := 0
		for i := 0; i < n; i++ {
			if th.Participate(p) {
				hits++
			}
		}
		rate := float64(hits) / n
		se := math.Sqrt(p * (1 - p) / n)
		if math.Abs(rate-p) > 3*se {
			t.Errorf("p=%.2f realized=%.4f outside 3SE (%.4f)", p, rate, 3*se)
		}
	}
}

func TestDeterministicGivenSeed(t *testing.T) {
	a, b := New(7), New(7)
	for i := 0; i < 1000; i++ {
		if a.Participate(0.4) != b.Participate(0.4) {
			t.Fatal("same seed must give same stream")
		}
	}
}
