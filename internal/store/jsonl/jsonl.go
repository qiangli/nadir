// Package jsonl is an append-only writer for request log entries.
// It's the human-readable companion to the SQLite store: JSONL is
// easy to grep/jq, easy to ship to log collectors, easy to recover
// from. Pair with internal/logmaint for size-based rotation.
package jsonl

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync"

	"github.com/qiangli/nadir/types"
)

type Writer struct {
	path string

	mu sync.Mutex
	f  *os.File
}

func Open(path string) (*Writer, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	return &Writer{path: path, f: f}, nil
}

func (w *Writer) Log(_ context.Context, e *types.RequestEntry) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.f == nil {
		return errors.New("jsonl writer closed")
	}
	b, err := json.Marshal(e)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	_, err = w.f.Write(b)
	return err
}

// Rotate moves the current log to archive and opens a new active file while
// excluding Log calls. Closing before rename also works on Windows.
func (w *Writer) Rotate(archive string) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.f == nil {
		return errors.New("jsonl writer closed")
	}
	// Never overwrite an existing archive (including a same-second rotation).
	if _, err := os.Lstat(archive); err == nil {
		return os.ErrExist
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	closeErr := w.f.Close()
	w.f = nil
	if closeErr != nil {
		return closeErr
	}
	renameErr := os.Rename(w.path, archive)
	// On rename failure, reopen the original file so subsequent logs remain
	// writable. On success this creates the next active file.
	f, openErr := os.OpenFile(w.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	w.f = f
	return errors.Join(renameErr, openErr)
}

func (w *Writer) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.f == nil {
		return nil
	}
	err := w.f.Close()
	w.f = nil
	return err
}

func (w *Writer) Size() (int64, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.f == nil {
		return 0, io.EOF
	}
	info, err := w.f.Stat()
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}

var _ types.RequestLogger = (*Writer)(nil)
