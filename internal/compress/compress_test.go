package compress

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/qiangli/nadir/types"
)

func msg(role, content string) types.Message {
	b, _ := json.Marshal(content)
	return types.Message{Role: role, Content: b}
}

func TestNoCompressBelowThreshold(t *testing.T) {
	r := Apply([]types.Message{msg("user", "hi")}, Config{MaxMessages: 50})
	if r.Folded != 0 {
		t.Errorf("should not fold, got %d", r.Folded)
	}
}

func TestFoldsOlderTurns(t *testing.T) {
	msgs := []types.Message{msg("system", "you are helpful")}
	for i := range 20 {
		msgs = append(msgs, msg("user", fmt.Sprintf("turn %d", i)))
		msgs = append(msgs, msg("assistant", fmt.Sprintf("reply %d", i)))
	}
	r := Apply(msgs, Config{MaxMessages: 10, KeepRecent: 4})
	if r.Folded == 0 {
		t.Fatal("should have folded")
	}
	// The result must keep KeepRecent + the system + the synthetic summary.
	if len(r.Messages) < 5 {
		t.Errorf("result too short: %d", len(r.Messages))
	}
	var sumText string
	_ = json.Unmarshal(r.Messages[1].Content, &sumText)
	if !strings.Contains(sumText, "earlier conversation summary") {
		t.Errorf("missing summary marker: %q", sumText)
	}
}
