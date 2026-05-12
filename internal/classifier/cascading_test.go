package classifier

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/qiangli/nadir/internal/types"
)

// stubClassifier returns canned results so we can drive Cascading
// through every branch without booting Ollama or an embedding model.
type stubClassifier struct {
	tier   types.Tier
	score  float64
	conf   float64
	err    error
	calls  int
}

func (s *stubClassifier) Warmup(_ context.Context) error { return nil }
func (s *stubClassifier) Classify(_ context.Context, _ string) (types.Tier, float64, float64, error) {
	s.calls++
	return s.tier, s.score, s.conf, s.err
}

func TestCascadingTrustsPrimaryAboveThreshold(t *testing.T) {
	primary := &stubClassifier{tier: types.TierComplex, score: 0.9, conf: 0.8}
	secondary := &stubClassifier{tier: types.TierSimple, score: 0.1, conf: 0.9}
	c := NewCascading(primary, secondary, 0.5)

	tier, _, _, err := c.Classify(context.Background(), "anything")
	if err != nil {
		t.Fatal(err)
	}
	if tier != types.TierComplex {
		t.Errorf("primary tier should stand, got %s", tier)
	}
	if secondary.calls != 0 {
		t.Errorf("secondary should not have been called, calls=%d", secondary.calls)
	}
}

func TestCascadingConsultsSecondaryBelowThreshold(t *testing.T) {
	primary := &stubClassifier{tier: types.TierSimple, score: 0.4, conf: 0.1}
	secondary := &stubClassifier{tier: types.TierComplex, score: 0.85, conf: 0.7}
	c := NewCascading(primary, secondary, 0.5)

	tier, score, conf, err := c.Classify(context.Background(), "ambiguous")
	if err != nil {
		t.Fatal(err)
	}
	if secondary.calls != 1 {
		t.Errorf("secondary should have been called, calls=%d", secondary.calls)
	}
	if tier != types.TierComplex {
		t.Errorf("secondary verdict should override, got tier=%s score=%v conf=%v", tier, score, conf)
	}
}

func TestCascadingFallsBackOnSecondaryError(t *testing.T) {
	primary := &stubClassifier{tier: types.TierSimple, score: 0.4, conf: 0.1}
	secondary := &stubClassifier{err: errors.New("ollama down")}
	c := NewCascading(primary, secondary, 0.5)

	tier, score, _, err := c.Classify(context.Background(), "ambiguous")
	if err != nil {
		t.Fatalf("expected fallback, not error, got %v", err)
	}
	if tier != types.TierSimple || score != 0.4 {
		t.Errorf("expected primary verdict when secondary fails, got tier=%s score=%v", tier, score)
	}
}

func TestCascadingNilSecondaryActsAsPassthrough(t *testing.T) {
	primary := &stubClassifier{tier: types.TierMid, score: 0.5, conf: 0.05}
	c := NewCascading(primary, nil, 0.5)

	tier, _, _, err := c.Classify(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if tier != types.TierMid {
		t.Errorf("nil secondary should pass through primary verdict, got %s", tier)
	}
}

// ============================================================
// LLM classifier tests — fake LLMClient drives Classify
// ============================================================

type fakeLLM struct {
	reply string
	err   error
	calls int
}

func (f *fakeLLM) Name() string { return "fake-llm" }
func (f *fakeLLM) Complete(_ context.Context, req *types.ChatRequest) (*types.ChatResponse, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	content, _ := json.Marshal(f.reply)
	return &types.ChatResponse{
		Model: req.Model,
		Choices: []types.Choice{
			{Index: 0, Message: types.Message{Role: "assistant", Content: content}, FinishReason: "stop"},
		},
	}, nil
}
func (f *fakeLLM) Stream(_ context.Context, _ *types.ChatRequest) (types.StreamIter, error) {
	return nil, errors.New("not implemented")
}

func TestLLMClassifyParsesScore(t *testing.T) {
	cases := []struct {
		reply string
		want  float64
	}{
		{"0.85", 0.85},
		{"0.85\n", 0.85},
		{"Score: 0.72", 0.72},
		{"`0.5`", 0.5},
		{"  0.3  ", 0.3},
		{"complex (0.9)", 0.9},
		{"0.85 - because it requires multi-step reasoning", 0.85},
	}
	for _, tc := range cases {
		t.Run(tc.reply, func(t *testing.T) {
			fake := &fakeLLM{reply: tc.reply}
			cls := NewLLM(fake, "llama3.2:3b", DefaultThresholds())
			_, score, _, err := cls.Classify(context.Background(), "x")
			if err != nil {
				t.Fatal(err)
			}
			if score < tc.want-1e-9 || score > tc.want+1e-9 {
				t.Errorf("reply=%q score=%v want=%v", tc.reply, score, tc.want)
			}
		})
	}
}

func TestLLMClassifyClampsOutOfRange(t *testing.T) {
	for _, reply := range []string{"-0.5", "2.0", "1.5"} {
		cls := NewLLM(&fakeLLM{reply: reply}, "test", DefaultThresholds())
		_, score, _, _ := cls.Classify(context.Background(), "x")
		if score < 0 || score > 1 {
			t.Errorf("reply=%q score=%v outside [0,1]", reply, score)
		}
	}
}

func TestLLMClassifyEmptyReplyErrors(t *testing.T) {
	cls := NewLLM(&fakeLLM{reply: "I cannot determine"}, "test", DefaultThresholds())
	_, _, _, err := cls.Classify(context.Background(), "x")
	if err == nil {
		t.Error("expected parse error on non-numeric reply")
	}
}
