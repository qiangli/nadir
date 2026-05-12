// Package credentials persists provider API keys to
// ~/.nadir/credentials.json. In v1 this is a static-key store — OAuth
// support (refresh tokens, device codes) is the Phase 5 expansion of
// the TokenProvider interface defined in github.com/qiangli/nadir/types.
package credentials

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"

	"github.com/qiangli/nadir/types"
)

type Store struct {
	path string

	mu   sync.RWMutex
	data map[string]string // provider → token
}

func Open(path string) (*Store, error) {
	s := &Store{path: path, data: map[string]string{}}
	if err := s.load(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Store) load() error {
	b, err := os.ReadFile(s.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return json.Unmarshal(b, &s.data)
}

func (s *Store) Save() error {
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	s.mu.RLock()
	b, err := json.MarshalIndent(s.data, "", "  ")
	s.mu.RUnlock()
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}

func (s *Store) Set(provider, token string) {
	s.mu.Lock()
	s.data[provider] = token
	s.mu.Unlock()
}

func (s *Store) Token(_ context.Context, provider string) (string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.data[provider]
	if !ok || v == "" {
		return "", errors.New("no credentials for provider " + provider)
	}
	return v, nil
}

var _ types.TokenProvider = (*Store)(nil)
