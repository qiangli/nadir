package cache

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/qiangli/nadir/internal/types"
)

func msgs(role, content string) []types.Message {
	return []types.Message{{Role: role, Content: json.RawMessage(`"` + content + `"`)}}
}

func TestPromptCachePutGet(t *testing.T) {
	c := NewPrompt(10, time.Minute)
	m := msgs("user", "hi")
	resp := &types.ChatResponse{ID: "r1"}
	c.Put("gpt", m, resp)
	got, ok := c.Get("gpt", m)
	if !ok || got.ID != "r1" {
		t.Fatalf("expected hit got %v ok=%v", got, ok)
	}
	if c.Stats().Hits != 1 {
		t.Errorf("hits=%d, want 1", c.Stats().Hits)
	}
}

func TestPromptCacheMiss(t *testing.T) {
	c := NewPrompt(10, time.Minute)
	_, ok := c.Get("gpt", msgs("user", "miss"))
	if ok {
		t.Fatal("expected miss")
	}
}

func TestPromptCacheTTL(t *testing.T) {
	c := NewPrompt(10, 5*time.Millisecond)
	m := msgs("user", "hi")
	c.Put("gpt", m, &types.ChatResponse{ID: "r1"})
	time.Sleep(20 * time.Millisecond)
	if _, ok := c.Get("gpt", m); ok {
		t.Fatal("expected expiry")
	}
}

func TestPromptCacheEviction(t *testing.T) {
	c := NewPrompt(2, time.Minute)
	c.Put("gpt", msgs("user", "a"), &types.ChatResponse{ID: "a"})
	c.Put("gpt", msgs("user", "b"), &types.ChatResponse{ID: "b"})
	c.Put("gpt", msgs("user", "c"), &types.ChatResponse{ID: "c"})
	if _, ok := c.Get("gpt", msgs("user", "a")); ok {
		t.Fatal("a should have been evicted")
	}
	if _, ok := c.Get("gpt", msgs("user", "c")); !ok {
		t.Fatal("c should be present")
	}
}

func TestSessionUpgradeOnly(t *testing.T) {
	s := NewSession(time.Minute)
	conv := msgs("user", "hello")

	got, tier, status := s.UpgradeIfHigher(conv, "haiku", types.TierSimple)
	if status != "new" || got != "haiku" || tier != types.TierSimple {
		t.Errorf("first put: %s %s %s", got, tier, status)
	}

	got, tier, status = s.UpgradeIfHigher(conv, "sonnet", types.TierComplex)
	if status != "upgraded" || tier != types.TierComplex {
		t.Errorf("upgrade: %s %s %s", got, tier, status)
	}

	got, tier, status = s.UpgradeIfHigher(conv, "haiku", types.TierSimple)
	if status != "pinned" || tier != types.TierComplex {
		t.Errorf("downgrade should be rejected: %s %s %s", got, tier, status)
	}
}
