package model

import (
	"fmt"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var initOnce sync.Once
var initErr error

// sharedLibPath resolves the onnxruntime dynamic library. Override with
// ONNXRUNTIME_LIB_PATH; defaults to the Homebrew location.
func sharedLibPath() string {
	if p := os.Getenv("ONNXRUNTIME_LIB_PATH"); p != "" {
		return p
	}
	return "/opt/homebrew/Cellar/onnxruntime/1.27.1/lib/libonnxruntime.dylib"
}

func initEnv() error {
	initOnce.Do(func() {
		ort.SetSharedLibraryPath(sharedLibPath())
		initErr = ort.InitializeEnvironment()
	})
	return initErr
}

// Model wraps an ONNX session that maps int64 feature vectors to pCTR.
type Model struct {
	session  *ort.DynamicAdvancedSession
	nFields  int
	inName   string
	outName  string
}

// LoadModel opens the exported ONNX model. inputName/outputName default to the
// names used by export.py ("features"/"pctr").
func LoadModel(path string, nFields int) (*Model, error) {
	if err := initEnv(); err != nil {
		return nil, fmt.Errorf("onnxruntime init: %w", err)
	}
	in, out := "features", "pctr"
	s, err := ort.NewDynamicAdvancedSession(path, []string{in}, []string{out}, nil)
	if err != nil {
		return nil, err
	}
	return &Model{session: s, nFields: nFields, inName: in, outName: out}, nil
}

func (m *Model) Close() error { return m.session.Destroy() }

// Predict runs a batch. features is [batch][nFields] int64; returns pCTR[batch].
func (m *Model) Predict(features [][]int64) ([]float32, error) {
	batch := len(features)
	if batch == 0 {
		return nil, nil
	}
	flat := make([]int64, 0, batch*m.nFields)
	for _, row := range features {
		if len(row) != m.nFields {
			return nil, fmt.Errorf("row has %d fields, want %d", len(row), m.nFields)
		}
		flat = append(flat, row...)
	}
	inShape := ort.NewShape(int64(batch), int64(m.nFields))
	inTensor, err := ort.NewTensor(inShape, flat)
	if err != nil {
		return nil, err
	}
	defer inTensor.Destroy()

	outputs := []ort.Value{nil}
	if err := m.session.Run([]ort.Value{inTensor}, outputs); err != nil {
		return nil, err
	}
	defer outputs[0].Destroy()

	out, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected output type %T", outputs[0])
	}
	data := out.GetData()
	res := make([]float32, len(data))
	copy(res, data)
	return res, nil
}
