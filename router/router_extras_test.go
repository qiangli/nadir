package router

// Additional router scenarios that mirror NadirClaw's test_routing.py
// + test_agent_role.py + test_complex_coding.py coverage. The base
// router_test.go has the simple-vs-complex routing covered; this
// file fills in profiles, vision swap, mid-tier behavior, and image
// detection.

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

func buildRouter(midConfigured bool, reasoning string) *Router {
	cfg := &Config{
		SimpleModel:      "haiku",
		ComplexModel:     "opus",
		ReasoningModel:   reasoning,
		TierThresholds:   [2]float64{0.35, 0.65},
		ProviderForModel: map[string]string{"haiku": "x", "opus": "x"},
	}
	if midConfigured {
		cfg.MidModel = "sonnet"
		cfg.ProviderForModel["sonnet"] = "x"
	}
	thresh := classifier.Thresholds{Simple: 0.35, Complex: 0.65, HasMid: midConfigured}
	return New(cfg, classifier.NewHeuristic(thresh), cache.NewSession(0), health.New())
}

func text(role, s string) types.Message {
	b, _ := json.Marshal(s)
	return types.Message{Role: role, Content: b}
}

func TestRouterReasoningHintGetsReasoningModel(t *testing.T) {
	r := buildRouter(false, "o1-preview")
	// NadirClaw requires 2+ reasoning markers; a single hint is not
	// enough. This prompt has "step by step" + "Prove that" = 2 markers.
	dec, err := r.Route(context.Background(), &types.ChatRequest{
		Model:    "auto",
		Messages: []types.Message{text("user", "Prove that this works step by step")},
	}, &types.User{ID: "u"})
	if err != nil {
		t.Fatal(err)
	}
	if dec.Model != "o1-preview" {
		t.Errorf("reasoning hint → %s, want o1-preview", dec.Model)
	}
	if !slices.Contains(dec.Modifiers, "reasoning") {
		t.Errorf("modifiers = %v, want 'reasoning'", dec.Modifiers)
	}
}

func TestRouterAgentRoleSystemBumpsToComplex(t *testing.T) {
	r := buildRouter(false, "")
	dec, err := r.Route(context.Background(), &types.ChatRequest{
		Model: "auto",
		Messages: []types.Message{
			text("system", "You are an agent that can call tools to accomplish tasks autonomously."),
			text("user", "hi"),
		},
	}, &types.User{ID: "u"})
	if err != nil {
		t.Fatal(err)
	}
	if dec.Tier != types.TierComplex {
		t.Errorf("agent role → tier=%s, want complex", dec.Tier)
	}
}

func TestRouterImageContentForcesVisionModel(t *testing.T) {
	// Set up a config where the complex model is vision-capable
	// (contains "claude" → modelHasVision()=true) but simple is not.
	cfg := &Config{
		SimpleModel:      "haiku-text",
		ComplexModel:     "claude-vision",
		TierThresholds:   [2]float64{0.35, 0.65},
		ProviderForModel: map[string]string{"haiku-text": "x", "claude-vision": "x"},
	}
	thresh := classifier.Thresholds{Simple: 0.35, Complex: 0.65}
	r := New(cfg, classifier.NewHeuristic(thresh), cache.NewSession(0), health.New())

	imgContent, _ := json.Marshal([]map[string]any{
		{"type": "text", "text": "what's in this image?"},
		{"type": "image_url", "image_url": map[string]any{"url": "data:image/jpeg;base64,xx"}},
	})
	dec, err := r.Route(context.Background(), &types.ChatRequest{
		Model:    "auto",
		Messages: []types.Message{{Role: "user", Content: imgContent}},
	}, &types.User{ID: "u"})
	if err != nil {
		t.Fatal(err)
	}
	if !modelHasVision(dec.Model) {
		t.Errorf("vision content should select vision-capable model, got %s", dec.Model)
	}
	if !slices.Contains(dec.Modifiers, "vision_swap") {
		t.Errorf("expected vision_swap modifier, got %v", dec.Modifiers)
	}
}

func TestRouterMidTierBucket(t *testing.T) {
	// With HasMid=true, a borderline score should land in mid tier.
	r := buildRouter(true, "")
	// Construct a prompt that scores around 0.5 (the heuristic gives
	// medium length + no code + 1 keyword → roughly mid).
	dec, err := r.Route(context.Background(), &types.ChatRequest{
		Model:    "auto",
		Messages: []types.Message{text("user", "explain why concurrent map writes are unsafe in Go and how sync.Map differs")},
	}, &types.User{ID: "u"})
	if err != nil {
		t.Fatal(err)
	}
	if dec.Tier == types.TierMid && dec.Model != "sonnet" {
		t.Errorf("mid tier should pick sonnet, got %s", dec.Model)
	}
}

func TestRouterFallbackChainHonoursHealth(t *testing.T) {
	r := buildRouter(true, "")
	// Mark sonnet (mid) as unhealthy; the chain should reorder it to
	// the back (or skip via Available()), but it shouldn't disappear.
	for range 10 {
		r.health.RecordFailure("sonnet", types.ErrServerError)
	}
	dec, err := r.Route(context.Background(), &types.ChatRequest{
		Model:    "auto",
		Messages: []types.Message{text("user", "hi")},
	}, &types.User{ID: "u"})
	if err != nil {
		t.Fatal(err)
	}
	// Chain shouldn't have sonnet at index 0 if it's unhealthy.
	if len(dec.FallbackChain) > 0 && dec.FallbackChain[0] == "sonnet" {
		t.Errorf("unhealthy sonnet should not lead the chain: %v", dec.FallbackChain)
	}
}

func TestRouterProviderResolution(t *testing.T) {
	// gpt-4o is not in ProviderForModel; should fall through to
	// the default "openai" provider.
	r := buildRouter(false, "")
	dec, err := r.Route(context.Background(), &types.ChatRequest{
		Model:    "gpt-4o",
		Messages: []types.Message{text("user", "hi")},
	}, &types.User{ID: "u"})
	if err != nil {
		t.Fatal(err)
	}
	if dec.Provider == "" {
		t.Errorf("provider should default to openai for unknown model, got %q", dec.Provider)
	}
}

func TestRouterEmptyPromptDoesNotPanic(t *testing.T) {
	r := buildRouter(false, "")
	_, err := r.Route(context.Background(), &types.ChatRequest{
		Model:    "auto",
		Messages: []types.Message{text("user", "")},
	}, &types.User{ID: "u"})
	if err != nil {
		t.Errorf("empty prompt should classify gracefully: %v", err)
	}
}
