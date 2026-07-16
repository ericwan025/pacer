package bid

import (
	"context"
	"testing"
)

type fakeTransform struct{}

func (fakeTransform) Apply(map[string]string) ([]int64, error) { return []int64{1, 2, 3}, nil }

type fakeScorer struct {
	p    float32
	runs int
}

func (f *fakeScorer) Predict(x [][]int64) ([]float32, error) {
	f.runs++
	return []float32{f.p}, nil
}

type fakeCache map[int64]float64

func (c fakeCache) Get(id int64) float64 {
	if v, ok := c[id]; ok {
		return v
	}
	return 1.0
}

type fakeCharger struct{ granted map[int64]bool }

func (c *fakeCharger) Charge(_ context.Context, id int64, _ float64) (bool, error) {
	if c.granted == nil {
		return true, nil
	}
	return c.granted[id], nil
}

func TestSecondPriceWinnerAndPrice(t *testing.T) {
	sc := &fakeScorer{p: 0.1}
	h := NewHandler(fakeTransform{}, sc, fakeCache{1: 1, 2: 1}, &fakeCharger{},
		0.0, BidShade, nil)
	// bids: c1 = 0.1*5*1 = 0.5 ; c2 = 0.1*3*1 = 0.3 -> winner c1 pays 0.3
	resp, err := h.Decide(context.Background(), &Request{
		Candidates: []Candidate{{ID: 1, Value: 5}, {ID: 2, Value: 3}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !resp.Won || resp.CampaignID != 1 {
		t.Fatalf("want winner 1, got %+v", resp)
	}
	if diff := resp.Price - 0.3; diff > 1e-6 || diff < -1e-6 {
		t.Errorf("price %v want 0.3", resp.Price)
	}
}

func TestInferenceRunsOncePerRequest(t *testing.T) {
	sc := &fakeScorer{p: 0.2}
	h := NewHandler(fakeTransform{}, sc, fakeCache{}, &fakeCharger{}, 0.0, BidShade, nil)
	_, _ = h.Decide(context.Background(), &Request{
		Candidates: []Candidate{{ID: 1, Value: 1}, {ID: 2, Value: 1}, {ID: 3, Value: 1}},
	})
	if sc.runs != 1 {
		t.Errorf("pCTR should be computed once per impression, ran %d times", sc.runs)
	}
}

func TestThrottleZeroSkipsInference(t *testing.T) {
	sc := &fakeScorer{p: 0.2}
	// multiplier 0 -> participation prob 0 -> nobody participates -> no inference
	h := NewHandler(fakeTransform{}, sc, fakeCache{1: 0.0}, &fakeCharger{}, 0.0, Throttle, nil)
	resp, _ := h.Decide(context.Background(), &Request{
		Candidates: []Candidate{{ID: 1, Value: 5}},
	})
	if resp.Won {
		t.Error("throttled-out candidate should not win")
	}
	if sc.runs != 0 {
		t.Errorf("filter-only path must skip inference, ran %d", sc.runs)
	}
}

func TestRejectedChargeMeansNoBid(t *testing.T) {
	sc := &fakeScorer{p: 0.1}
	h := NewHandler(fakeTransform{}, sc, fakeCache{1: 1}, &fakeCharger{granted: map[int64]bool{1: false}},
		0.0, BidShade, nil)
	resp, _ := h.Decide(context.Background(), &Request{
		Candidates: []Candidate{{ID: 1, Value: 5}},
	})
	if resp.Won {
		t.Error("a rejected (over-budget) charge must result in no-bid")
	}
}
