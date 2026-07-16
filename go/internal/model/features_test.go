package model

import (
	"encoding/json"
	"os"
	"testing"
)

func TestFNV1a32KnownVectors(t *testing.T) {
	cases := map[string]uint32{"": 0x811C9DC5, "a": 0xE40C292C, "foobar": 0xBF9CF968}
	for s, want := range cases {
		if got := FNV1a32(s); got != want {
			t.Errorf("FNV1a32(%q)=%#x want %#x", s, got, want)
		}
	}
}

// TestFeatureParityWithPython feeds the exact rows Python encoded and asserts the
// Go transform produces identical integer vectors. If this fails, the Go serving
// path computes different features and the whole latency story measures the wrong
// computation. Regenerate the fixture with:
//
//	cd python && ../.venv/bin/python -m scripts.gen_feature_fixture
func TestFeatureParityWithPython(t *testing.T) {
	tr, err := LoadTransform("testdata/transform.json")
	if err != nil {
		t.Fatal(err)
	}

	var rows []map[string]string
	mustLoad(t, "testdata/rows.json", &rows)
	var expected [][]int64
	mustLoad(t, "testdata/expected.json", &expected)
	var fieldOrder []string
	mustLoad(t, "testdata/field_order.json", &fieldOrder)

	if got := tr.FieldOrder(); len(got) != len(fieldOrder) {
		t.Fatalf("field order length %d != %d", len(got), len(fieldOrder))
	}
	for i, f := range fieldOrder {
		if tr.FieldOrder()[i] != f {
			t.Fatalf("field order mismatch at %d: %q vs %q", i, tr.FieldOrder()[i], f)
		}
	}

	if len(rows) != len(expected) {
		t.Fatalf("rows %d != expected %d", len(rows), len(expected))
	}
	for i, raw := range rows {
		got, err := tr.Apply(raw)
		if err != nil {
			t.Fatalf("row %d: %v", i, err)
		}
		for j := range got {
			if got[j] != expected[i][j] {
				t.Fatalf("row %d field %d (%s): go=%d python=%d",
					i, j, fieldOrder[j], got[j], expected[i][j])
			}
		}
	}
}

func mustLoad(t *testing.T, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatal(err)
	}
}
