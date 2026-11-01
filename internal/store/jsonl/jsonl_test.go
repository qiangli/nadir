package jsonl

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
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

	if err := w.Rotate(path + ".1"); err != nil {
		t.Fatalf("rotate: %v", err)
	}
	_ = w.Log(context.Background(), &types.RequestEntry{ID: "b"})

	data, _ := os.ReadFile(path)
	old, err := os.ReadFile(path + ".1")
	if err != nil || !strings.Contains(string(old), `"id":"a"`) || strings.Contains(string(old), `"id":"b"`) {
		t.Fatalf("archive partition: %q, %v", old, err)
	}
	if strings.Contains(string(data), `"id":"a"`) || !strings.Contains(string(data), `"id":"b"`) {
		t.Fatalf("rotated file missing new row: %q", data)
	}
}

func TestRotateFailureKeepsWriterUsable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "log.jsonl")
	w, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if err := w.Log(context.Background(), &types.RequestEntry{ID: "before"}); err != nil {
		t.Fatal(err)
	}
	if err := w.Rotate(filepath.Join(t.TempDir(), "missing", "archive")); err == nil {
		t.Fatal("expected rename failure")
	}
	if err := w.Log(context.Background(), &types.RequestEntry{ID: "after"}); err != nil {
		t.Fatalf("writer did not recover: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil || !strings.Contains(string(data), `"id":"before"`) || !strings.Contains(string(data), `"id":"after"`) {
		t.Fatalf("lost rows: %q, %v", data, err)
	}
}

func TestRotateSerializesConcurrentRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "log.jsonl")
	w, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	done := make(chan error, 1)
	go func() {
		for i := range 100 {
			if err := w.Log(context.Background(), &types.RequestEntry{ID: strconv.Itoa(i)}); err != nil {
				done <- err
				return
			}
		}
		done <- nil
	}()
	if err := w.Rotate(path + ".1"); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	seen := make(map[string]bool)
	for _, file := range []string{path + ".1", path} {
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
			if line == "" {
				continue
			}
			var row types.RequestEntry
			if err := json.Unmarshal([]byte(line), &row); err != nil {
				t.Fatal(err)
			}
			if seen[row.ID] {
				t.Fatalf("duplicate row %q", row.ID)
			}
			seen[row.ID] = true
		}
	}
	if len(seen) != 100 {
		t.Fatalf("retained %d rows, want 100", len(seen))
	}
}
