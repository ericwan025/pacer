package model

import (
	"math"
	"testing"
)

// TestONNXParityWithPython runs the same rows through the Go onnxruntime path and
// asserts the pCTR matches Python's onnxruntime within 1e-5. If it fails, the Go
// serving path computes a different pCTR than training and every downstream bid is
// wrong. Regenerate with:
//
//	cd python && ../.venv/bin/python -m scripts.gen_model_fixture
func TestONNXParityWithPython(t *testing.T) {
	var inputs [][]int64
	mustLoad(t, "testdata/model_inputs.json", &inputs)
	var expected []float64
	mustLoad(t, "testdata/model_expected.json", &expected)

	nFields := len(inputs[0])
	m, err := LoadModel("testdata/model.onnx", nFields)
	if err != nil {
		t.Skipf("cannot load ONNX model (is onnxruntime installed? set ONNXRUNTIME_LIB_PATH): %v", err)
	}
	defer func() { _ = m.Close() }()

	got, err := m.Predict(inputs)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(expected) {
		t.Fatalf("got %d preds, want %d", len(got), len(expected))
	}
	var maxDiff float64
	for i := range got {
		d := math.Abs(float64(got[i]) - expected[i])
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff >= 1e-5 {
		t.Errorf("max abs diff %g >= 1e-5", maxDiff)
	}
}
