// Package bid is the hot-path bid handler.
//
// Path for POST /bid:
//  1. parse request, apply the feature transform
//  2. for each candidate campaign, read its multiplier from the local cache
//  3. throttle decision (PRNG draw) — in throttle mode this can skip inference
//  4. if anyone participates: ONNX inference -> pCTR (once per impression)
//  5. compute bids, run the second-price auction
//  6. charge the winner via the atomic budget CAS; a rejected charge = no-bid
//
// Dependencies are interfaces so the handler is testable without Redis or ONNX.
package bid

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/ericwan025/pacer/go/internal/throttle"
)

// Mode selects how the multiplier is applied.
type Mode int

const (
	BidShade Mode = iota // multiplier scales the bid; always participate
	Throttle             // multiplier is participation probability; bid full value
)

type Scorer interface {
	Predict(features [][]int64) ([]float32, error)
}
type MultiplierCache interface {
	Get(id int64) float64
}
type Charger interface {
	Charge(ctx context.Context, id int64, cost float64) (bool, error)
}
type Transformer interface {
	Apply(raw map[string]string) ([]int64, error)
}

type Candidate struct {
	ID    int64   `json:"id"`
	Value float64 `json:"value"` // value_per_click
}

type Request struct {
	Features   map[string]string `json:"features"`
	Candidates []Candidate       `json:"candidates"`
}

type Response struct {
	Won        bool    `json:"won"`
	CampaignID int64   `json:"campaign_id,omitempty"`
	Price      float64 `json:"price,omitempty"`
	PCTR       float64 `json:"pctr,omitempty"`
}

type Handler struct {
	transform Transformer
	scorer    Scorer
	cache     MultiplierCache
	charger   Charger
	reserve   float64
	mode      Mode
	metrics   *Metrics
}

func NewHandler(t Transformer, s Scorer, c MultiplierCache, ch Charger,
	reserve float64, mode Mode, m *Metrics) *Handler {
	if m == nil {
		m = NewMetrics()
	}
	return &Handler{transform: t, scorer: s, cache: c, charger: ch,
		reserve: reserve, mode: mode, metrics: m}
}

// Decide runs the bid logic for one request (no HTTP). Returns the response.
func (h *Handler) Decide(ctx context.Context, req *Request) (*Response, error) {
	feats, err := h.transform.Apply(req.Features)
	if err != nil {
		return nil, err
	}

	// pCTR depends only on the impression, so it is at most ONE inference per
	// request, computed lazily so the throttle-everyone case pays nothing.
	var pctr float64
	haveP := false
	score := func() (float64, error) {
		if !haveP {
			out, err := h.scorer.Predict([][]int64{feats})
			if err != nil {
				return 0, err
			}
			pctr = float64(out[0])
			haveP = true
		}
		return pctr, nil
	}

	var bids []bidEntry
	for _, c := range req.Candidates {
		mult := h.cache.Get(c.ID)
		var bidAmt float64
		if h.mode == Throttle {
			if !throttle.Participate(clamp01(mult)) {
				h.metrics.Throttled.Add(1)
				continue
			}
			p, err := score()
			if err != nil {
				return nil, err
			}
			bidAmt = p * c.Value // full-value bid
		} else {
			p, err := score()
			if err != nil {
				return nil, err
			}
			bidAmt = p * c.Value * mult
		}
		if bidAmt > 0 {
			bids = append(bids, bidEntry{c.ID, bidAmt})
		}
	}

	winner, price, won := secondPrice(bids, h.reserve)
	resp := &Response{Won: won, PCTR: pctr}
	if won {
		ok, err := h.charger.Charge(ctx, winner, price)
		if err != nil {
			return nil, err
		}
		if ok {
			resp.CampaignID = winner
			resp.Price = price
			h.metrics.Wins.Add(1)
		} else {
			resp.Won = false // budget exhausted between cache read and charge
		}
	}
	return resp, nil
}

// ServeHTTP handles POST /bid.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	start := time.Now()
	defer func() { h.metrics.ObserveLatency(time.Since(start)) }()
	h.metrics.Requests.Add(1)
	var req Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	resp, err := h.Decide(r.Context(), &req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

type bidEntry struct {
	id  int64
	bid float64
}

// secondPrice: highest bid wins, pays max(second, reserve); below reserve -> no
// winner. Ties broken by lowest id (deterministic).
func secondPrice(bids []bidEntry, reserve float64) (int64, float64, bool) {
	if len(bids) == 0 {
		return 0, 0, false
	}
	topIdx := 0
	for i := 1; i < len(bids); i++ {
		if bids[i].bid > bids[topIdx].bid ||
			(bids[i].bid == bids[topIdx].bid && bids[i].id < bids[topIdx].id) {
			topIdx = i
		}
	}
	top := bids[topIdx]
	if top.bid < reserve {
		return 0, 0, false
	}
	second := 0.0
	for i, b := range bids {
		if i == topIdx {
			continue
		}
		if b.bid > second {
			second = b.bid
		}
	}
	price := reserve
	if second > price {
		price = second
	}
	return top.id, price, true
}
