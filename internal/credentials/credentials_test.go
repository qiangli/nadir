package credentials

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestSetSaveLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "credentials.json")
	s, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	s.Set("openai", "sk-abc")
	s.Set("anthropic", "sk-ant-xyz")
	if err := s.Save(); err != nil {
		t.Fatalf("save: %v", err)
	}

	// Reload and verify.
	s2, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := s2.Token(context.Background(), "openai")
	if err != nil || tok != "sk-abc" {
		t.Errorf("openai = %q err=%v", tok, err)
	}
	tok, err = s2.Token(context.Background(), "anthropic")
	if err != nil || tok != "sk-ant-xyz" {
		t.Errorf("anthropic = %q err=%v", tok, err)
	}
}

func TestUnknownProviderErrors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "credentials.json")
	s, _ := Open(path)
	if _, err := s.Token(context.Background(), "nope"); err == nil {
		t.Error("expected error for unknown provider")
	}
}

func TestMissingFileIsHarmless(t *testing.T) {
	s, err := Open(filepath.Join(t.TempDir(), "doesnotexist.json"))
	if err != nil {
		t.Fatalf("missing file should not error, got %v", err)
	}
	if _, err := s.Token(context.Background(), "openai"); err == nil {
		t.Error("expected error when no tokens are present")
	}
}

func TestSaveCreatesParentDir(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "deep", "credentials.json")
	s, _ := Open(path)
	s.Set("openai", "sk-test")
	if err := s.Save(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Errorf("expected file at %s, got %v", path, err)
	}
}

func TestSaveAtomic(t *testing.T) {
	// Save to a non-existent .tmp shouldn't leak the tmp file.
	path := filepath.Join(t.TempDir(), "credentials.json")
	s, _ := Open(path)
	s.Set("a", "b")
	_ = s.Save()
	if _, err := os.Stat(path + ".tmp"); err == nil {
		t.Errorf(".tmp file should have been renamed away")
	}
}
