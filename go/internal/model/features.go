// Package model applies the exact Phase 1 feature transform (loaded from JSON)
// and, later, runs ONNX inference. The transform MUST match the Python pipeline
// byte-for-byte, or the served pCTR silently diverges from training. That is
// what the Python/Go parity test guards.
package model

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"time"
)

const (
	fnvOffset32 = 2166136261
	fnvPrime32  = 16777619
	oovIndex    = 0
)

// FNV1a32 must stay identical to pacer.data.features.fnv1a_32 in Python.
func FNV1a32(s string) uint32 {
	h := uint32(fnvOffset32)
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= fnvPrime32
	}
	return h
}

type transformJSON struct {
	Config struct {
		VocabFields []string         `json:"vocab_fields"`
		HashFields  map[string]int64 `json:"hash_fields"`
		MinCount    int              `json:"min_count"`
	} `json:"config"`
	VocabMaps  map[string]map[string]int64 `json:"vocab_maps"`
	FieldOrder []string                    `json:"field_order"`
	OOVIndex   int64                       `json:"oov_index"`
}

type Transform struct {
	fieldOrder []string
	vocabSet   map[string]bool
	hashFields map[string]int64
	vocabMaps  map[string]map[string]int64
}

func LoadTransform(path string) (*Transform, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var tj transformJSON
	if err := json.Unmarshal(b, &tj); err != nil {
		return nil, err
	}
	vs := make(map[string]bool, len(tj.Config.VocabFields))
	for _, f := range tj.Config.VocabFields {
		vs[f] = true
	}
	return &Transform{
		fieldOrder: tj.FieldOrder,
		vocabSet:   vs,
		hashFields: tj.Config.HashFields,
		vocabMaps:  tj.VocabMaps,
	}, nil
}

func (t *Transform) FieldOrder() []string { return t.fieldOrder }

// Apply encodes one raw request row into the integer feature vector, in
// FieldOrder. `raw` holds raw string values keyed by column name; it must include
// "hour" (YYMMDDHH) so time features can be derived exactly as Python does.
func (t *Transform) Apply(raw map[string]string) ([]int64, error) {
	vals := make(map[string]string, len(raw)+2)
	for k, v := range raw {
		vals[k] = v
	}
	dow, hod, err := timeFeatures(raw["hour"])
	if err != nil {
		return nil, err
	}
	vals["day_of_week"] = strconv.Itoa(dow)
	vals["hour_of_day"] = strconv.Itoa(hod)

	out := make([]int64, len(t.fieldOrder))
	for i, f := range t.fieldOrder {
		v := vals[f]
		if t.vocabSet[f] {
			if idx, ok := t.vocabMaps[f][v]; ok {
				out[i] = idx
			} else {
				out[i] = oovIndex
			}
			continue
		}
		if n, ok := t.hashFields[f]; ok {
			if v == "" {
				out[i] = oovIndex
			} else {
				out[i] = 1 + int64(FNV1a32(v))%(n-1)
			}
			continue
		}
		return nil, fmt.Errorf("field %q is neither vocab nor hash", f)
	}
	return out, nil
}

// timeFeatures parses YYMMDDHH into (day_of_week, hour_of_day) matching Python's
// datetime.date.weekday() (Monday=0..Sunday=6).
func timeFeatures(hour string) (int, int, error) {
	if len(hour) < 8 {
		// zero-pad on the left like Python's str.zfill(8)
		hour = fmt.Sprintf("%08s", hour)
	}
	yy, err1 := strconv.Atoi(hour[0:2])
	mm, err2 := strconv.Atoi(hour[2:4])
	dd, err3 := strconv.Atoi(hour[4:6])
	hh, err4 := strconv.Atoi(hour[6:8])
	if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
		return 0, 0, fmt.Errorf("bad hour %q", hour)
	}
	d := time.Date(2000+yy, time.Month(mm), dd, 0, 0, 0, 0, time.UTC)
	// Go: Sunday=0..Saturday=6. Python: Monday=0..Sunday=6.
	dow := (int(d.Weekday()) + 6) % 7
	return dow, hh, nil
}
