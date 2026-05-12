package setup

import (
	"bytes"
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/qiangli/nadir/internal/credentials"
	"github.com/qiangli/nadir/internal/ollama"
)

func TestWizardWritesCreds(t *testing.T) {
	dir := t.TempDir()
	in := strings.NewReader("sk-test\n\nAIza-test\n")
	out := &bytes.Buffer{}
	w := &Wizard{
		In: in, Out: out, DataDir: dir,
		Discover: func(_ context.Context) []ollama.Instance { return nil },
	}
	if err := w.Run(context.Background()); err != nil {
		t.Fatalf("run: %v", err)
	}
	s, err := credentials.Open(filepath.Join(dir, "credentials.json"))
	if err != nil {
		t.Fatal(err)
	}
	got, err := s.Token(context.Background(), "openai")
	if err != nil || got != "sk-test" {
		t.Errorf("openai token = %q err=%v", got, err)
	}
	if _, err := s.Token(context.Background(), "anthropic"); err == nil {
		t.Errorf("blank anthropic should be absent")
	}
}
