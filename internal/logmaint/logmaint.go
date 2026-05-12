// Package logmaint handles JSONL log rotation and retention: when the
// active log crosses a size threshold, it's renamed to a timestamped
// `.1` slot, gzipped, and old archives past `keep` are deleted. The
// active file is rotated by the jsonl.Writer.
package logmaint

import (
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type Config struct {
	Path        string        // active log file
	MaxBytes    int64         // rotate when active file >= MaxBytes
	Keep        int           // keep N archives
	CheckEvery  time.Duration // poll interval; 0 disables background loop
	NowFunc     func() time.Time
}

type Maintainer struct {
	cfg     Config
	rotator interface{ Rotate() error }
}

func New(cfg Config, rotator interface{ Rotate() error }) *Maintainer {
	if cfg.MaxBytes <= 0 {
		cfg.MaxBytes = 100 * 1024 * 1024
	}
	if cfg.Keep <= 0 {
		cfg.Keep = 7
	}
	if cfg.NowFunc == nil {
		cfg.NowFunc = time.Now
	}
	return &Maintainer{cfg: cfg, rotator: rotator}
}

// MaybeRotate inspects the active file size and rotates if needed.
// Callers wire this to a ticker; it's safe to call from a goroutine.
func (m *Maintainer) MaybeRotate() error {
	info, err := os.Stat(m.cfg.Path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	if info.Size() < m.cfg.MaxBytes {
		return nil
	}

	stamp := m.cfg.NowFunc().UTC().Format("20060102T150405Z")
	archive := m.cfg.Path + "." + stamp
	if err := os.Rename(m.cfg.Path, archive); err != nil {
		return err
	}
	if m.rotator != nil {
		if err := m.rotator.Rotate(); err != nil {
			return err
		}
	}
	if err := gzipFile(archive); err != nil {
		return err
	}
	return m.pruneOld()
}

// pruneOld deletes archives past the keep count.
func (m *Maintainer) pruneOld() error {
	dir := filepath.Dir(m.cfg.Path)
	base := filepath.Base(m.cfg.Path)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	archives := []string{}
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, base+".") && strings.HasSuffix(name, ".gz") {
			archives = append(archives, filepath.Join(dir, name))
		}
	}
	sort.Sort(sort.Reverse(sort.StringSlice(archives)))
	for i, a := range archives {
		if i >= m.cfg.Keep {
			_ = os.Remove(a)
		}
	}
	return nil
}

func gzipFile(path string) error {
	in, err := os.Open(path)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(path + ".gz")
	if err != nil {
		return err
	}
	defer out.Close()
	gw := gzip.NewWriter(out)
	if _, err := io.Copy(gw, in); err != nil {
		_ = gw.Close()
		return err
	}
	if err := gw.Close(); err != nil {
		return err
	}
	if err := os.Remove(path); err != nil {
		return fmt.Errorf("remove %s: %w", path, err)
	}
	return nil
}
