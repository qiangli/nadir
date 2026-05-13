package skillrouter

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/qiangli/nadir/types"
)

// llmStub returns a canned text reply (or error). Records call count
// so tests can assert that the LLM was — or wasn't — consulted.
type llmStub struct {
	reply string
	err   error
	delay time.Duration
	calls atomic.Int64
}

func (s *llmStub) Name() string { return "llm-stub" }

func (s *llmStub) Complete(ctx context.Context, _ *types.ChatRequest) (*types.ChatResponse, error) {
	s.calls.Add(1)
	if s.delay > 0 {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(s.delay):
		}
	}
	if s.err != nil {
		return nil, s.err
	}
	content, _ := json.Marshal(s.reply)
	return &types.ChatResponse{
		Choices: []types.Choice{{Message: types.Message{Role: "assistant", Content: content}}},
	}, nil
}

func (s *llmStub) Stream(_ context.Context, _ *types.ChatRequest) (types.StreamIter, error) {
	return nil, errors.New("not implemented")
}

// helper to build a Cascade against the shared 4D fake catalog.
func newCascade(t *testing.T, llm *llmStub, opts ...CascadeOption) *Cascade {
	t.Helper()
	emb := newFakeEmb()
	primary, err := NewSemantic(context.Background(), emb, newTestCatalog())
	if err != nil {
		t.Fatal(err)
	}
	return NewCascade(primary, llm, "stub-model", opts...)
}

func TestCascade_PrimaryConfident_LLMNotCalled(t *testing.T) {
	// "please review my PR" → top1 ~0.99 on /review, runner-up far below.
	// Margin should comfortably exceed MinMargin → LLM stays cold.
	llm := &llmStub{reply: "/security-review"} // would be wrong if called
	c := newCascade(t, llm)
	d, err := c.Route(context.Background(), "please review my PR")
	if err != nil {
		t.Fatal(err)
	}
	if d.Skill != "/review" {
		t.Errorf("expected primary verdict /review, got %+v", d)
	}
	if llm.calls.Load() != 0 {
		t.Errorf("LLM should not have been called, calls=%d", llm.calls.Load())
	}
}

func TestCascade_NarrowMargin_LLMConsulted(t *testing.T) {
	// "close-call: review or sec" → top1 and top2 nearly tied.
	// Cascade must consult the LLM and trust its verdict.
	llm := &llmStub{reply: "/security-review"}
	c := newCascade(t, llm, WithMinMargin(0.20))
	d, err := c.Route(context.Background(), "close-call: review or sec")
	if err != nil {
		t.Fatal(err)
	}
	if llm.calls.Load() != 1 {
		t.Errorf("LLM should have been consulted, calls=%d", llm.calls.Load())
	}
	if d.Skill != "/security-review" {
		t.Errorf("expected LLM verdict /security-review, got %+v", d)
	}
	// Embedding-side margin must be preserved on the LLM-decided result.
	if d.Margin <= 0 {
		t.Errorf("expected non-zero embedding margin preserved, got %v", d.Margin)
	}
}

func TestCascade_BelowMinScore_NoLLMCall(t *testing.T) {
	// Unrelated prompt: every embedding-side candidate falls below
	// MinScore. The LLM can't help — there's nothing in the catalog
	// to rerank. Cascade must fellthrough WITHOUT spending an LLM call.
	llm := &llmStub{reply: "/init"}
	emb := newFakeEmb()
	primary, _ := NewSemantic(context.Background(), emb, newTestCatalog(), WithMinScore(0.5))
	c := NewCascade(primary, llm, "stub-model")
	d, err := c.Route(context.Background(), "unrelated random text")
	if err != nil {
		t.Fatal(err)
	}
	if !d.FellThrough {
		t.Errorf("expected fellthrough for unrelated prompt, got %+v", d)
	}
	if llm.calls.Load() != 0 {
		t.Errorf("LLM must not be consulted when no candidate is viable, calls=%d", llm.calls.Load())
	}
}

func TestCascade_LLMError_FallsBackToPrimary(t *testing.T) {
	// Margin is narrow → cascade wants to consult LLM. LLM errors.
	// Must silently return the primary's verdict, not propagate the
	// error. Routing always produces an answer.
	llm := &llmStub{err: errors.New("ollama down")}
	c := newCascade(t, llm, WithMinMargin(0.20))
	d, err := c.Route(context.Background(), "close-call: review or sec")
	if err != nil {
		t.Fatalf("expected silent fallback, got error: %v", err)
	}
	if d.FellThrough {
		t.Errorf("expected primary verdict, not fellthrough: %+v", d)
	}
	// "close-call: review or sec" has /review slightly ahead.
	if d.Skill != "/review" {
		t.Errorf("expected fallback to primary's /review, got %+v", d)
	}
}

func TestCascade_LLMTimeout_FallsBackToPrimary(t *testing.T) {
	llm := &llmStub{reply: "/security-review", delay: 50 * time.Millisecond}
	c := newCascade(t, llm,
		WithMinMargin(0.20),
		WithLLMTimeout(5*time.Millisecond),
	)
	d, err := c.Route(context.Background(), "close-call: review or sec")
	if err != nil {
		t.Fatalf("expected silent fallback on timeout, got: %v", err)
	}
	// Primary's verdict for this prompt is /review.
	if d.Skill != "/review" || d.FellThrough {
		t.Errorf("expected primary fallback on timeout, got %+v", d)
	}
}

func TestCascade_LLMFellThrough_Honored(t *testing.T) {
	// If the LLM is consulted and replies "none", trust the LLM:
	// it considered the shortlist and explicitly said nothing fits.
	llm := &llmStub{reply: "none"}
	c := newCascade(t, llm, WithMinMargin(0.20))
	d, err := c.Route(context.Background(), "close-call: review or sec")
	if err != nil {
		t.Fatal(err)
	}
	if !d.FellThrough {
		t.Errorf("expected LLM fellthrough to be honored, got %+v", d)
	}
}

func TestCascade_NoLLMConfigured_PrimaryOnly(t *testing.T) {
	// Edge case: cascade constructed without an LLM client. Should
	// behave like Semantic alone — return primary's verdict even on
	// narrow-margin prompts, never error.
	emb := newFakeEmb()
	primary, _ := NewSemantic(context.Background(), emb, newTestCatalog())
	c := NewCascade(primary, nil, "", WithMinMargin(0.99))
	d, err := c.Route(context.Background(), "close-call: review or sec")
	if err != nil {
		t.Fatal(err)
	}
	if d.Skill != "/review" {
		t.Errorf("expected primary verdict /review when no LLM, got %+v", d)
	}
}

func TestCascade_ShortlistRespected(t *testing.T) {
	// Verify the LLM rerank step only sees the top-K, not the full
	// catalog. We assert this indirectly: a 2-entry shortlist means
	// /init never reaches the LLM, so an LLM reply of "/init" can't
	// match anything and must fellthrough.
	llm := &llmStub{reply: "/init"}
	c := newCascade(t, llm,
		WithMinMargin(0.99), // force LLM consultation
		WithShortlistK(2),
	)
	d, err := c.Route(context.Background(), "close-call: review or sec")
	if err != nil {
		t.Fatal(err)
	}
	if !d.FellThrough {
		t.Errorf("/init was outside the top-2 shortlist; LLM picking it must fellthrough. Got %+v", d)
	}
}

func TestCascade_PreservesPrimaryMarginOnLLMResult(t *testing.T) {
	llm := &llmStub{reply: "/security-review"}
	c := newCascade(t, llm, WithMinMargin(0.20))
	d, _ := c.Route(context.Background(), "close-call: review or sec")
	if d.Margin <= 0 {
		t.Errorf("expected embedding margin preserved on LLM verdict, got %v", d.Margin)
	}
}

func TestCascade_NilPrimaryErrors(t *testing.T) {
	c := NewCascade(nil, &llmStub{}, "x")
	_, err := c.Route(context.Background(), "anything")
	if err == nil {
		t.Fatal("expected error for nil primary")
	}
}
