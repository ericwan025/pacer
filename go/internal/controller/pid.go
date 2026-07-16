// Package controller is the Go port of the Python PID pacing controller.
//
// It MUST reproduce pacer.sim.pid.PIDController exactly — same conditional-
// integration anti-windup, same low-pass filtered derivative on the measurement,
// same output map + clamp. The Go/Python parity test guards this: pacing control
// runs as a separate service in Go, but is designed and tuned in Python.
package controller

import "math"

type Config struct {
	Kp, Ki, Kd float64
	Dt         float64
	MinMult    float64
	MaxMult    float64
	DerivTau   float64
	AntiWindup bool
	Mapping    string // "linear" or "sigmoid"
}

type PID struct {
	cfg           Config
	integral      float64
	filteredDeriv float64
	prevMeas      float64
	hasPrev       bool
}

func New(cfg Config) *PID { return &PID{cfg: cfg} }

func (p *PID) Reset() {
	p.integral = 0
	p.filteredDeriv = 0
	p.hasPrev = false
}

func mapOutput(mapping string, raw, lo, hi float64) float64 {
	if mapping == "sigmoid" {
		return lo + (hi-lo)*(1.0/(1.0+math.Exp(-raw)))
	}
	return raw
}

// Update mirrors PIDController.update in Python.
func (p *PID) Update(setpoint, measurement float64) float64 {
	c := p.cfg
	error := setpoint - measurement

	var rawDeriv float64
	if p.hasPrev {
		rawDeriv = -(measurement - p.prevMeas) / c.Dt
	}
	alpha := c.Dt / (c.DerivTau + c.Dt)
	p.filteredDeriv += alpha * (rawDeriv - p.filteredDeriv)
	d := c.Kd * p.filteredDeriv

	pTerm := c.Kp * error

	inc := c.Ki * error * c.Dt
	provisional := pTerm + (p.integral + inc) + d
	mappedProv := mapOutput(c.Mapping, provisional, c.MinMult, c.MaxMult)
	satHigh := mappedProv > c.MaxMult
	satLow := mappedProv < c.MinMult
	pushingOut := (satHigh && error > 0) || (satLow && error < 0)
	if !c.AntiWindup || !pushingOut {
		p.integral += inc
	}

	output := pTerm + p.integral + d
	mult := mapOutput(c.Mapping, output, c.MinMult, c.MaxMult)
	if mult > c.MaxMult {
		mult = c.MaxMult
	}
	if mult < c.MinMult {
		mult = c.MinMult
	}

	p.prevMeas = measurement
	p.hasPrev = true
	return mult
}
