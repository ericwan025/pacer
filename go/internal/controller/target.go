package controller

const secondsPerHour = 3600.0

// TargetCurve returns the fraction of budget that should be spent by elapsed
// seconds into the day. Mirrors pacer.sim.target.
type TargetCurve interface {
	Fraction(elapsed float64) float64
}

// Uniform target: spend proportional to time. Traffic-blind baseline.
type Uniform struct{ Horizon float64 }

func (u Uniform) Fraction(t float64) float64 {
	if u.Horizon <= 0 {
		return 1
	}
	f := t / u.Horizon
	if f < 0 {
		return 0
	}
	if f > 1 {
		return 1
	}
	return f
}

// TrafficAware target: cumulative expected traffic from per-hour volumes.
type TrafficAware struct {
	prefix  []float64 // len n+1 cumulative
	vol     []float64
	total   float64
	horizon float64
}

func NewTrafficAware(hourlyVolume []float64) TrafficAware {
	prefix := make([]float64, len(hourlyVolume)+1)
	var total float64
	for i, v := range hourlyVolume {
		total += v
		prefix[i+1] = total
	}
	return TrafficAware{prefix: prefix, vol: hourlyVolume, total: total,
		horizon: float64(len(hourlyVolume)) * secondsPerHour}
}

func (t TrafficAware) Fraction(elapsed float64) float64 {
	if elapsed <= 0 || t.total <= 0 {
		return 0
	}
	if elapsed >= t.horizon {
		return 1
	}
	h := int(elapsed / secondsPerHour)
	fracIn := (elapsed - float64(h)*secondsPerHour) / secondsPerHour
	cum := t.prefix[h] + fracIn*t.vol[h]
	return cum / t.total
}
