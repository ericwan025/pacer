package controller

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type fixture struct {
	Config struct {
		Kp, Ki, Kd, Dt float64
		MinMult        float64 `json:"min_mult"`
		MaxMult        float64 `json:"max_mult"`
		DerivTau       float64 `json:"deriv_tau"`
		AntiWindup     bool    `json:"anti_windup"`
		Mapping        string
	} `json:"config"`
	Trace   [][]float64 `json:"trace"`
	Outputs []float64   `json:"outputs"`
}

// TestPIDParityWithPython asserts the Go controller reproduces the Python PID
// output sequence to within float tolerance on a shared trace that exercises
// anti-windup saturation and the filtered derivative. Regenerate with:
//
//	cd python && ../.venv/bin/python -m scripts.gen_pid_fixture
func TestPIDParityWithPython(t *testing.T) {
	b, err := os.ReadFile("testdata/pid_parity.json")
	if err != nil {
		t.Fatal(err)
	}
	var fx fixture
	if err := json.Unmarshal(b, &fx); err != nil {
		t.Fatal(err)
	}

	pid := New(Config{
		Kp: fx.Config.Kp, Ki: fx.Config.Ki, Kd: fx.Config.Kd, Dt: fx.Config.Dt,
		MinMult: fx.Config.MinMult, MaxMult: fx.Config.MaxMult,
		DerivTau: fx.Config.DerivTau, AntiWindup: fx.Config.AntiWindup,
		Mapping: fx.Config.Mapping,
	})

	var maxDiff float64
	for i, tr := range fx.Trace {
		got := pid.Update(tr[0], tr[1])
		d := math.Abs(got - fx.Outputs[i])
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 1e-9 {
		t.Errorf("max abs diff %g exceeds 1e-9 — Go and Python PID diverge", maxDiff)
	}
}
