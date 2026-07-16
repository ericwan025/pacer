package controller

import (
	"math"
	"testing"
)

func TestUniformEndpoints(t *testing.T) {
	u := Uniform{Horizon: 100}
	if u.Fraction(0) != 0 || u.Fraction(100) != 1 || u.Fraction(50) != 0.5 {
		t.Error("uniform endpoints/midpoint wrong")
	}
	if u.Fraction(200) != 1 {
		t.Error("uniform should clamp at 1")
	}
}

func TestTrafficAwareMatchesPythonShape(t *testing.T) {
	// vol [10,45,45]; end of hour 1 -> 10% spent (trough), unlike uniform's 33%.
	ta := NewTrafficAware([]float64{10, 45, 45})
	if got := ta.Fraction(secondsPerHour); math.Abs(got-0.10) > 1e-12 {
		t.Errorf("end of hour 1 = %v, want 0.10", got)
	}
	if got := ta.Fraction(3 * secondsPerHour); math.Abs(got-1.0) > 1e-12 {
		t.Errorf("end = %v, want 1.0", got)
	}
}
