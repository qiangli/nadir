package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"

	"github.com/qiangli/nadir/types"
)

// Session pins (system + first-user-message) → (model, tier). It
// upgrades but never downgrades: a tier=complex pin sticks even if a
// later turn classifies as simple, so multi-turn conversations don't
// bounce between models mid-thread.
type Session struct {
	ttl time.Duration

	mu      sync.RWMutex
	entries map[string]sessionEntry
}

type sessionEntry struct {
	model   string
	tier    types.Tier
	updated time.Time
}

func NewSession(ttl time.Duration) *Session {
	return &Session{ttl: ttl, entries: make(map[string]sessionEntry)}
}

func (s *Session) Get(msgs []types.Message) (string, types.Tier, bool) {
	key := sessionKey(msgs)
	s.mu.RLock()
	e, ok := s.entries[key]
	s.mu.RUnlock()
	if !ok {
		return "", "", false
	}
	if s.ttl > 0 && time.Since(e.updated) > s.ttl {
		s.mu.Lock()
		delete(s.entries, key)
		s.mu.Unlock()
		return "", "", false
	}
	return e.model, e.tier, true
}

// UpgradeIfHigher records (model, tier) unless an existing pin is at a
// higher or equal tier. status is "new" when first stored, "upgraded"
// when we replace a lower tier, "pinned" when caller keeps the existing.
func (s *Session) UpgradeIfHigher(msgs []types.Message, model string, tier types.Tier) (string, types.Tier, string) {
	key := sessionKey(msgs)
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.entries[key]
	if !ok {
		s.entries[key] = sessionEntry{model: model, tier: tier, updated: time.Now()}
		return model, tier, "new"
	}
	if tierRank(tier) <= tierRank(e.tier) {
		return e.model, e.tier, "pinned"
	}
	s.entries[key] = sessionEntry{model: model, tier: tier, updated: time.Now()}
	return model, tier, "upgraded"
}

func tierRank(t types.Tier) int {
	switch t {
	case types.TierFree:
		return 0
	case types.TierSimple:
		return 1
	case types.TierMid:
		return 2
	case types.TierComplex:
		return 3
	case types.TierReasoning:
		return 4
	default:
		return 0
	}
}

func sessionKey(msgs []types.Message) string {
	h := sha256.New()
	for _, m := range msgs {
		if m.Role == "system" || m.Role == "user" {
			h.Write([]byte(m.Role))
			h.Write([]byte{0})
			h.Write([]byte(m.Content))
			h.Write([]byte{0})
			if m.Role == "user" {
				break // pin on (all system + first user)
			}
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

var _ types.SessionCache = (*Session)(nil)
