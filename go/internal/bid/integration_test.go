package bid_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/ericwan025/pacer/go/internal/bid"
	"github.com/ericwan025/pacer/go/internal/budget"
	"github.com/ericwan025/pacer/go/internal/model"
	"github.com/ericwan025/pacer/go/internal/pacing"
)

const td = "../model/testdata/"

// TestEndToEndBidChargesRedis wires the REAL transform, ONNX model, pacing cache,
// and Redis budget store together and drives one HTTP /bid request, asserting the
// winner's spend actually moved in Redis. Skips if Redis or the model is absent.
func TestEndToEndBidChargesRedis(t *testing.T) {
	addr := os.Getenv("REDIS_ADDR")
	if addr == "" {
		addr = "localhost:6379"
	}
	rdb := redis.NewClient(&redis.Options{Addr: addr})
	ctx := context.Background()
	if err := rdb.Ping(ctx).Err(); err != nil {
		t.Skipf("redis not reachable (run `docker compose up -d redis`): %v", err)
	}
	store := budget.New(rdb)

	transform, err := model.LoadTransform(td + "transform.json")
	if err != nil {
		t.Fatal(err)
	}
	nFields := len(transform.FieldOrder())
	scorer, err := model.LoadModel(td+"integration_model.onnx", nFields)
	if err != nil {
		t.Skipf("cannot load ONNX model: %v", err)
	}
	defer scorer.Close()

	// a fresh campaign with a generous budget and full multiplier
	const cid = 4242
	_ = rdb.Del(ctx, "campaign:4242").Err()
	if err := store.SetCampaign(ctx, cid, 1000.0, 1.0); err != nil {
		t.Fatal(err)
	}

	cache := pacing.New(store.AllMultipliers, 1.0)
	if err := cache.Refresh(ctx); err != nil {
		t.Fatal(err)
	}

	// reserve > 0 so a single uncontested winner actually pays (and spends)
	handler := bid.NewHandler(transform, scorer, cache, store, 0.05, bid.BidShade, nil)
	srv := httptest.NewServer(handler)
	defer srv.Close()

	// load a real raw feature row from the fixture
	var rows []map[string]string
	b, _ := os.ReadFile(td + "rows.json")
	_ = json.Unmarshal(b, &rows)

	reqBody, _ := json.Marshal(bid.Request{
		Features:   rows[0],
		Candidates: []bid.Candidate{{ID: cid, Value: 10.0}},
	})
	resp, err := http.Post(srv.URL, "application/json", strings.NewReader(string(reqBody)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out bid.Response
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}

	if !out.Won || out.CampaignID != cid {
		t.Fatalf("expected campaign %d to win, got %+v", cid, out)
	}
	if out.PCTR <= 0 || out.PCTR >= 1 {
		t.Errorf("pctr %v not a valid probability", out.PCTR)
	}

	// the charge must have hit Redis
	time.Sleep(10 * time.Millisecond)
	spend, err := store.Spend(ctx, cid)
	if err != nil {
		t.Fatal(err)
	}
	if spend <= 0 {
		t.Errorf("expected spend to increase in Redis, got %v", spend)
	}
}
