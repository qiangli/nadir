package skillrouter

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/qiangli/nadir/types"
)

// stubClient returns canned text as the assistant content. It lets us
// drive every parser branch without booting an LLM.
type stubClient struct {
	reply string
	err   error
	delay time.Duration
	calls int
}

func (s *stubClient) Name() string { return "stub" }

func (s *stubClient) Complete(ctx context.Context, _ *types.ChatRequest) (*types.ChatResponse, error) {
	s.calls++
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
		Choices: []types.Choice{
			{Message: types.Message{Role: "assistant", Content: content}},
		},
	}, nil
}

func (s *stubClient) Stream(_ context.Context, _ *types.ChatRequest) (types.StreamIter, error) {
	return nil, errors.New("stub: stream not implemented")
}

var catalog = []Skill{
	{Name: "/review", Description: "Review a pull request"},
	{Name: "/security-review", Description: "Audit pending changes for security issues"},
	{Name: "/init", Description: "Initialize a CLAUDE.md"},
}

func TestRoute_ExactMatch(t *testing.T) {
	client := &stubClient{reply: "/security-review"}
	r := New(client, "stub-model", catalog)
	d, err := r.Route(context.Background(), "audit this branch")
	if err != nil {
		t.Fatal(err)
	}
	if d.Skill != "/security-review" || d.Confidence != 1.0 || d.FellThrough {
		t.Errorf("got %+v", d)
	}
}

func TestRoute_CaseInsensitiveMatch(t *testing.T) {
	client := &stubClient{reply: "/SECURITY-REVIEW"}
	r := New(client, "stub-model", catalog)
	d, err := r.Route(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if d.Skill != "/security-review" || d.Confidence != 0.9 {
		t.Errorf("got %+v", d)
	}
}

func TestRoute_SubstringPicksLongestOnOverlap(t *testing.T) {
	// Reply contains both /review and /security-review; longest must win.
	client := &stubClient{reply: "I think /security-review is best."}
	r := New(client, "stub-model", catalog)
	d, err := r.Route(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if d.Skill != "/security-review" || d.Confidence != 0.7 {
		t.Errorf("longest-overlap rule failed: %+v", d)
	}
}

func TestRoute_NoneFallsThrough(t *testing.T) {
	client := &stubClient{reply: "none"}
	r := New(client, "stub-model", catalog)
	d, err := r.Route(context.Background(), "anything")
	if err != nil {
		t.Fatal(err)
	}
	if !d.FellThrough || d.Skill != "" {
		t.Errorf("expected fellthrough, got %+v", d)
	}
}

func TestRoute_GarbageFallsThrough(t *testing.T) {
	client := &stubClient{reply: "purple monkey dishwasher"}
	r := New(client, "stub-model", catalog)
	d, err := r.Route(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if !d.FellThrough {
		t.Errorf("expected fellthrough on garbage, got %+v", d)
	}
}

func TestRoute_EmptyReplyFallsThrough(t *testing.T) {
	client := &stubClient{reply: ""}
	r := New(client, "stub-model", catalog)
	d, err := r.Route(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if !d.FellThrough {
		t.Errorf("expected fellthrough on empty, got %+v", d)
	}
}

func TestRoute_LabelPrefixStripped(t *testing.T) {
	// Small models often echo the prompt's "Skill:" label.
	client := &stubClient{reply: "Skill: /init"}
	r := New(client, "stub-model", catalog)
	d, err := r.Route(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if d.Skill != "/init" || d.Confidence != 1.0 {
		t.Errorf("expected exact-match after label strip, got %+v", d)
	}
}

func TestRoute_TrailingPunctuationIgnored(t *testing.T) {
	client := &stubClient{reply: "/review."}
	r := New(client, "stub-model", catalog)
	d, err := r.Route(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if d.Skill != "/review" || d.Confidence != 1.0 {
		t.Errorf("expected exact match after trim, got %+v", d)
	}
}

func TestRoute_NilClient(t *testing.T) {
	r := New(nil, "stub-model", catalog)
	_, err := r.Route(context.Background(), "x")
	if err == nil {
		t.Fatal("expected error on nil client")
	}
}

func TestRoute_EmptyCatalog(t *testing.T) {
	r := New(&stubClient{}, "stub-model", nil)
	_, err := r.Route(context.Background(), "x")
	if err == nil {
		t.Fatal("expected error on empty catalog")
	}
}

func TestRoute_UpstreamError(t *testing.T) {
	client := &stubClient{err: errors.New("ollama down")}
	r := New(client, "stub-model", catalog)
	_, err := r.Route(context.Background(), "x")
	if err == nil {
		t.Fatal("expected upstream error to propagate")
	}
}

func TestRoute_Timeout(t *testing.T) {
	client := &stubClient{reply: "/review", delay: 50 * time.Millisecond}
	r := New(client, "stub-model", catalog, WithTimeout(5*time.Millisecond))
	_, err := r.Route(context.Background(), "x")
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

func TestRoute_RawResponsePreserved(t *testing.T) {
	client := &stubClient{reply: "Skill: /review"}
	r := New(client, "stub-model", catalog)
	d, _ := r.Route(context.Background(), "x")
	if d.RawResponse != "Skill: /review" {
		t.Errorf("RawResponse=%q want verbatim reply", d.RawResponse)
	}
}

func TestRenderPrompt_IncludesAllSkills(t *testing.T) {
	got := renderPrompt(catalog, "test")
	for _, s := range catalog {
		if !contains(got, s.Name) {
			t.Errorf("prompt missing skill %q\n%s", s.Name, got)
		}
		if !contains(got, s.Description) {
			t.Errorf("prompt missing description for %q", s.Name)
		}
	}
	if !contains(got, "test") {
		t.Errorf("prompt missing user prompt:\n%s", got)
	}
}

func contains(haystack, needle string) bool {
	return len(needle) > 0 && len(haystack) >= len(needle) &&
		(haystack == needle || indexOf(haystack, needle) >= 0)
}

func indexOf(haystack, needle string) int {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return i
		}
	}
	return -1
}
