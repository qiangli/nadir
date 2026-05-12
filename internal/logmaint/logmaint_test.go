package logmaint

import (
	"compress/gzip"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/qiangli/nadir/internal/store/jsonl"
	"github.com/qiangli/nadir/internal/types"
)

func TestRotateGzipsAndPrunes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "log.jsonl")
	w, err := jsonl.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	// Write enough to exceed the 200 byte threshold.
	for range 50 {
		_ = w.Log(context.Background(), &types.RequestEntry{ID: "x", Model: "m", Status: "ok"})
	}
	m := New(Config{Path: path, MaxBytes: 200, Keep: 2}, w)
	if err := m.MaybeRotate(); err != nil {
		t.Fatalf("rotate: %v", err)
	}
	entries, _ := os.ReadDir(dir)
	gz := 0
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".gz") {
			gz++
		}
	}
	if gz != 1 {
		t.Fatalf("want 1 gz archive, got %d", gz)
	}
	// Verify the gz archive decompresses to valid JSONL.
	var archive string
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".gz") {
			archive = filepath.Join(dir, e.Name())
		}
	}
	f, _ := os.Open(archive)
	defer f.Close()
	gr, _ := gzip.NewReader(f)
	defer gr.Close()
	data, _ := io.ReadAll(gr)
	if !strings.Contains(string(data), `"id":"x"`) {
		t.Errorf("archive content unexpected: %q", data[:min(80, len(data))])
	}
}
