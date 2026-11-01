package logmaint

import (
	"compress/gzip"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/qiangli/nadir/internal/store/jsonl"
	"github.com/qiangli/nadir/types"
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
	m := New(Config{Path: path, MaxBytes: 200, Keep: 2, NowFunc: func() time.Time { return time.Unix(1700000000, 0) }}, w)
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

func TestGzipRemovesClosedInput(t *testing.T) {
	path := filepath.Join(t.TempDir(), "archive")
	if err := os.WriteFile(path, []byte("retained rows\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := gzipFile(path); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("input remains: %v", err)
	}
	f, err := os.Open(path + ".gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	data, err := io.ReadAll(gz)
	if err != nil || string(data) != "retained rows\n" {
		t.Fatalf("gzip rows: %q, %v", data, err)
	}
}
