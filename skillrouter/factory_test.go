package skillrouter

import (
	"context"
	"testing"
)

func factoryCatalog() []Skill {
	return []Skill{
		{
			Name: "/review",
			Examples: []string{
				"review pull request",
				"look at this diff",
				"check my PR",
				"give feedback on this code change",
			},
		},
		{
			Name: "/security-review",
			Examples: []string{
				"audit for vulnerabilities",
				"check for security issues",
				"find security flaws in the code",
				"security review the auth changes",
			},
		},
		{
			Name: "/init",
			Examples: []string{
				"initialize CLAUDE.md",
				"create the agent guidance file",
				"set up CLAUDE.md from scratch",
			},
		},
	}
}

func TestNewLexical_RoutesEasyCase(t *testing.T) {
	m, err := NewLexical(factoryCatalog())
	if err != nil {
		t.Fatal(err)
	}
	d, err := m.Route(context.Background(), "please review my pull request")
	if err != nil {
		t.Fatal(err)
	}
	if d.Skill != "/review" || d.FellThrough {
		t.Errorf("expected /review, got %+v", d)
	}
}

func TestNewLexical_FellthroughOnOOD(t *testing.T) {
	m, err := NewLexical(factoryCatalog())
	if err != nil {
		t.Fatal(err)
	}
	// "tell me a joke" shares no vocabulary with any skill exemplar →
	// TF-IDF cosine ≈ 0 → below MinScore=0.25 → FellThrough.
	d, err := m.Route(context.Background(), "tell me a joke")
	if err != nil {
		t.Fatal(err)
	}
	if !d.FellThrough {
		t.Errorf("expected fellthrough on OOD prompt, got %+v", d)
	}
}

func TestNewLexical_RejectsEmptyCatalog(t *testing.T) {
	if _, err := NewLexical(nil); err == nil {
		t.Fatal("expected error on empty catalog")
	}
}

func TestNewHybrid_RoutesEasyCaseWithoutLLM(t *testing.T) {
	// On a confident TF-IDF pick (high score, comfortable margin),
	// Cascade must not consult the LLM. We assert by counting calls
	// on a stub client.
	llm := &llmStub{reply: "/init"} // would be wrong if consulted
	m, err := NewHybrid(context.Background(), llm, "stub-model", factoryCatalog())
	if err != nil {
		t.Fatal(err)
	}
	d, err := m.Route(context.Background(), "review pull request")
	if err != nil {
		t.Fatal(err)
	}
	if d.Skill != "/review" {
		t.Errorf("expected /review from primary, got %+v", d)
	}
	if llm.calls.Load() != 0 {
		t.Errorf("LLM must not be consulted on a confident primary pick, calls=%d", llm.calls.Load())
	}
}

func TestNewHybrid_ConsultsLLMOnNarrowMargin(t *testing.T) {
	// "review and audit" hits /review and /security-review almost
	// equally on TF-IDF → narrow margin → Cascade should consult LLM.
	llm := &llmStub{reply: "/security-review"}
	m, err := NewHybrid(context.Background(), llm, "stub-model", factoryCatalog())
	if err != nil {
		t.Fatal(err)
	}
	d, err := m.Route(context.Background(), "review and audit this change")
	if err != nil {
		t.Fatal(err)
	}
	if llm.calls.Load() == 0 {
		t.Errorf("LLM should have been consulted on narrow margin, calls=%d", llm.calls.Load())
	}
	if d.Skill != "/security-review" {
		t.Errorf("expected LLM verdict /security-review, got %+v", d)
	}
}

func TestNewHybrid_RejectsBadArgs(t *testing.T) {
	skills := factoryCatalog()
	ctx := context.Background()
	if _, err := NewHybrid(ctx, &llmStub{}, "m", nil); err == nil {
		t.Error("expected error on empty catalog")
	}
	if _, err := NewHybrid(ctx, nil, "m", skills); err == nil {
		t.Error("expected error on nil client")
	}
	if _, err := NewHybrid(ctx, &llmStub{}, "", skills); err == nil {
		t.Error("expected error on empty model")
	}
}
