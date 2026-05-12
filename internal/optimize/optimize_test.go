package optimize

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/qiangli/nadir/internal/types"
)

func msg(role, content string) types.Message {
	b, _ := json.Marshal(content)
	return types.Message{Role: role, Content: b}
}

func TestOffMode(t *testing.T) {
	in := []types.Message{msg("user", "hello   world  ")}
	r := Apply(in, Off)
	if r.BytesSaved != 0 {
		t.Errorf("Off mode should save nothing, got %d", r.BytesSaved)
	}
}

func TestSafeCollapsesWhitespace(t *testing.T) {
	in := []types.Message{msg("user", "hello     world\n\n\n\n\nlast line")}
	r := Apply(in, Safe)
	if r.BytesSaved <= 0 {
		t.Errorf("safe should save bytes, got %d", r.BytesSaved)
	}
	if !slices.Contains(r.Applied, "whitespace") {
		t.Errorf("applied should include whitespace, got %v", r.Applied)
	}
}

func TestSafeDedupsToolOutputs(t *testing.T) {
	in := []types.Message{
		msg("user", "hi"),
		msg("tool", "same output"),
		msg("tool", "same output"),
	}
	r := Apply(in, Safe)
	if len(r.Messages) != 2 {
		t.Errorf("dedup should drop dup, got %d msgs", len(r.Messages))
	}
}

func TestAggressiveTruncates(t *testing.T) {
	big := strings.Repeat("x", 8000)
	in := []types.Message{msg("tool", big)}
	r := Apply(in, Aggressive)
	if !slices.Contains(r.Applied, "truncate_long_tool") {
		t.Errorf("expected truncate, applied=%v", r.Applied)
	}
}
