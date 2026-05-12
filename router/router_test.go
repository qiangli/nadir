package router

import (
	"context"
	"encoding/json"
	"slices"
	"testing"

	"github.com/qiangli/nadir/cache"
	"github.com/qiangli/nadir/classifier"
	"github.com/qiangli/nadir/health"
	"github.com/qiangli/nadir/types"
)

func newTestRouter() *Router {
	cfg := &Config{
		SimpleModel:    "haiku",
		MidModel:       "sonnet",
		ComplexModel:   "opus",
		TierThresholds: [2]float64{0.35, 0.65},
		ProviderForModel: map[string]string{
			"haiku":  "anthropic",
			"sonnet": "anthropic",
			"opus":   "anthropic",
		},
	}
	return New(cfg, classifier.NewHeuristic(classifier.Thresholds{Simple: 0.35, Complex: 0.65, HasMid: true}), cache.NewSession(0), health.New())
}

func userMsg(text string) types.Message {
	b, _ := json.Marshal(text)
	return types.Message{Role: "user", Content: b}
}

func TestRouteAutoSimplePrompt(t *testing.T) {
	r := newTestRouter()
	dec, err := r.Route(context.Background(), &types.ChatRequest{
		Model:    "auto",
		Messages: []types.Message{userMsg("hi")},
	}, &types.User{ID: "u1"})
	if err != nil {
		t.Fatalf("route: %v", err)
	}
	if dec.Tier != types.TierSimple {
		t.Errorf("expected simple tier, got %s (score %v)", dec.Tier, dec.Score)
	}
	if dec.Model != "haiku" {
		t.Errorf("expected haiku, got %s", dec.Model)
	}
}

func TestRouteAutoComplexPrompt(t *testing.T) {
	r := newTestRouter()
	dec, err := r.Route(context.Background(), &types.ChatRequest{
		Model: "auto",
		Messages: []types.Message{userMsg(
			"Refactor this race condition: ```go\nfunc x(){}\n```. " +
				"Walk me through your reasoning step by step and explain why this deadlocks.")},
	}, &types.User{ID: "u1"})
	if err != nil {
		t.Fatalf("route: %v", err)
	}
	if dec.Tier != types.TierComplex {
		t.Errorf("expected complex tier, got %s (score %v)", dec.Tier, dec.Score)
	}
	if dec.Model != "opus" {
		t.Errorf("expected opus, got %s", dec.Model)
	}
}

func TestRouteExplicitAliasResolves(t *testing.T) {
	r := newTestRouter()
	dec, err := r.Route(context.Background(), &types.ChatRequest{
		Model:    "sonnet", // short alias
		Messages: []types.Message{userMsg("hi")},
	}, &types.User{ID: "u1"})
	if err != nil {
		t.Fatalf("route: %v", err)
	}
	if dec.Model != "claude-sonnet-4-5-20250929" {
		t.Errorf("sonnet alias should resolve, got %s", dec.Model)
	}
}

func TestRouteExplicitNonAlias(t *testing.T) {
	r := newTestRouter()
	dec, err := r.Route(context.Background(), &types.ChatRequest{
		Model:    "gpt-4o", // not in alias map
		Messages: []types.Message{userMsg("hi")},
	}, &types.User{ID: "u1"})
	if err != nil {
		t.Fatalf("route: %v", err)
	}
	if dec.Model != "gpt-4o" {
		t.Errorf("non-alias model should pass through, got %s", dec.Model)
	}
}

func TestRouteAgenticUpgrade(t *testing.T) {
	r := newTestRouter()
	// Force tools present so the agentic modifier kicks in.
	dec, err := r.Route(context.Background(), &types.ChatRequest{
		Model:    "auto",
		Messages: []types.Message{userMsg("hi")},
		Tools:    []types.Tool{{Type: "function"}},
	}, &types.User{ID: "u1"})
	if err != nil {
		t.Fatalf("route: %v", err)
	}
	if dec.Tier != types.TierComplex {
		t.Errorf("agentic should bump to complex, got %s", dec.Tier)
	}
	if !slices.Contains(dec.Modifiers, "agentic") {
		t.Errorf("modifiers missing agentic: %v", dec.Modifiers)
	}
}

func TestRouteFallbackChainExcludesPrimary(t *testing.T) {
	r := newTestRouter()
	dec, err := r.Route(context.Background(), &types.ChatRequest{
		Model:    "auto",
		Messages: []types.Message{userMsg("hi")},
	}, &types.User{ID: "u1"})
	if err != nil {
		t.Fatalf("route: %v", err)
	}
	for _, m := range dec.FallbackChain {
		if m == dec.Model {
			t.Errorf("fallback chain should not contain primary: %v", dec.FallbackChain)
		}
	}
}

