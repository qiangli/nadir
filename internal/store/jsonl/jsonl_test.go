package jsonl

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/qiangli/nadir/types"
)

func TestWriterAppendsLines(t *testing.T) {
	path := filepath.Join(t.TempDir(), "log.jsonl")
	w, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	for i := range 3 {
		_ = w.Log(context.Background(), &types.RequestEntry{ID: "r", Model: "m", Status: "ok", PromptTokens: i})
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	if len(lines) != 3 {
		t.Fatalf("lines = %d, want 3", len(lines))
	}
	var first map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
}

func TestRotateReopensFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "log.jsonl")
	w, _ := Open(path)
	defer w.Close()
	_ = w.Log(context.Background(), &types.RequestEntry{ID: "a"})

	// Simulate rotation: move the file aside, then rotate.
	if err := os.Rename(path, path+".1"); err != nil {
		t.Fatal(err)
	}
	if err := w.Rotate(); err != nil {
		t.Fatalf("rotate: %v", err)
	}
	_ = w.Log(context.Background(), &types.RequestEntry{ID: "b"})

	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), `"id":"b"`) {
		t.Fatalf("rotated file missing new row: %q", data)
	}
}
