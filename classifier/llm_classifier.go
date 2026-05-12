package classifier

// LLM-backed classifier: ask an upstream model (typically a small,
// local one like Ollama llama3.2:3b) to rate complexity on a 0–1
// scale. Used as the "deep path" in a Cascading classifier when the
// primary embedding pipeline is uncertain.
//
// Cost: one upstream LLM call (~200–500ms for a 1B–3B local model).
// Worth it when the alternative is misrouting a request to an
// expensive premium model.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/qiangli/nadir/types"
)

const llmClassifyPrompt = `You classify the complexity of a user request for routing to either a cheap or expensive language model.

Reply with EXACTLY one decimal number between 0.0 and 1.0 and nothing else.

Scale:
0.00–0.20 — trivial: greetings, single-word answers, lookups, short summaries
0.20–0.50 — simple: short explanations, one-shot tasks with clear scope
0.50–0.80 — moderate: multi-step but bounded: writing a function, debugging one bug
0.80–1.00 — complex: open-ended design, deep reasoning, multi-file refactors, agentic loops

Request:
%s

Score:`

// LLM is a Classifier that delegates the actual scoring to an
// upstream LLMClient. Use a small/fast model — the only thing we need
// is a number back.
type LLM struct {
	client     types.LLMClient
	model      string
	thresholds Thresholds
	maxTokens  int
}

// NewLLM builds an LLM-backed classifier. The client should be
// configured for a small/local model: NadirClaw recommends a 1B–3B
// parameter Ollama model for ~300–500ms second-opinion latency.
func NewLLM(client types.LLMClient, model string, t Thresholds) *LLM {
	if model == "" {
		model = "llama3.2:3b"
	}
	return &LLM{client: client, model: model, thresholds: t, maxTokens: 8}
}

func (l *LLM) Warmup(ctx context.Context) error {
	_, _, _, err := l.Classify(ctx, "hello")
	return err
}

func (l *LLM) Classify(ctx context.Context, prompt string) (types.Tier, float64, float64, error) {
	if l.client == nil {
		return "", 0, 0, errors.New("llm classifier: nil client")
	}
	req := &types.ChatRequest{
		Model: l.model,
		Messages: []types.Message{
			{Role: "user", Content: jsonString(fmt.Sprintf(llmClassifyPrompt, truncatePrompt(prompt, 4000)))},
		},
		MaxTokens: &l.maxTokens,
	}
	resp, err := l.client.Complete(ctx, req)
	if err != nil {
		return "", 0, 0, fmt.Errorf("llm classifier: %w", err)
	}
	if resp == nil || len(resp.Choices) == 0 {
		return "", 0, 0, errors.New("llm classifier: empty response")
	}
	score, err := parseScore(resp.Choices[0].Message.Content)
	if err != nil {
		return "", 0, 0, fmt.Errorf("llm classifier: %w", err)
	}
	// Confidence is the distance from 0.5 (matches the convention
	// used by the embedding classifier so callers can compare).
	confidence := 2 * abs(score-0.5)
	tier := BucketScore(score, l.thresholds)
	return tier, score, confidence, nil
}

// parseScore is lenient — the model often pads with whitespace, code
// fences, "Score:" prefixes, or trailing commentary. Extract the first
// float-looking token in [0, 1].
func parseScore(raw json.RawMessage) (float64, error) {
	var s string
	if len(raw) > 0 {
		_ = json.Unmarshal(raw, &s)
	}
	if s == "" {
		return 0, errors.New("blank content")
	}
	cleaned := strings.TrimSpace(strings.Trim(s, "`*"))
	cleaned = strings.TrimPrefix(strings.ToLower(cleaned), "score:")
	cleaned = strings.TrimSpace(cleaned)
	// Split on anything that isn't a digit, sign, decimal point, or
	// exponent character — gives us every float-looking token in the
	// reply (handles parens, commas, "complex (0.9)", etc.).
	for _, tok := range strings.FieldsFunc(cleaned, func(r rune) bool {
		switch r {
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', '+', 'e', 'E':
			return false
		}
		return true
	}) {
		v, err := strconv.ParseFloat(strings.TrimRight(tok, "."), 64)
		if err != nil {
			continue
		}
		if v < 0 {
			v = 0
		}
		if v > 1 {
			v = 1
		}
		return v, nil
	}
	return 0, fmt.Errorf("no parseable float in %q", s)
}

func jsonString(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

func truncatePrompt(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

var _ types.Classifier = (*LLM)(nil)
