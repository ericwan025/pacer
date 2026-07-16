package model

import (
	"fmt"
	"os"
	"runtime"
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

// Model wraps a POOL of ONNX sessions mapping int64 feature vectors to pCTR.
//
// onnxruntime_go sessions are not safe for concurrent Run, so the hot path must
// not share one across goroutines. We keep a pool (one session per worker slot)
// handed out through a buffered channel; each Predict borrows a session, runs,
// and returns it. This is both correct under concurrency and lets inference run
// in parallel instead of serializing on a single session.
type Model struct {
	pool    chan *ort.DynamicAdvancedSession
	all     []*ort.DynamicAdvancedSession
	nFields int
}

// LoadModel opens the exported ONNX model with a session pool. poolSize<=0 uses
// GOMAXPROCS. Input/output names are the ones export.py uses ("features"/"pctr").
func LoadModel(path string, nFields int) (*Model, error) {
	return LoadModelPool(path, nFields, 0)
}

func LoadModelPool(path string, nFields, poolSize int) (*Model, error) {
	if err := initEnv(); err != nil {
		return nil, fmt.Errorf("onnxruntime init: %w", err)
	}
	if poolSize <= 0 {
		poolSize = runtime.GOMAXPROCS(0)
	}
	m := &Model{pool: make(chan *ort.DynamicAdvancedSession, poolSize), nFields: nFields}
	for i := 0; i < poolSize; i++ {
		s, err := ort.NewDynamicAdvancedSession(path, []string{"features"}, []string{"pctr"}, nil)
		if err != nil {
			_ = m.Close()
			return nil, err
		}
		m.all = append(m.all, s)
		m.pool <- s
	}
	return m, nil
}

func (m *Model) Close() error {
	for _, s := range m.all {
		_ = s.Destroy()
	}
	return nil
}

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
	defer func() { _ = inTensor.Destroy() }()

	sess := <-m.pool                  // borrow a session
	defer func() { m.pool <- sess }() // return it
	outputs := []ort.Value{nil}
	if err := sess.Run([]ort.Value{inTensor}, outputs); err != nil {
		return nil, err
	}
	defer func() { _ = outputs[0].Destroy() }()

	out, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected output type %T", outputs[0])
	}
	data := out.GetData()
	res := make([]float32, len(data))
	copy(res, data)
	return res, nil
}
