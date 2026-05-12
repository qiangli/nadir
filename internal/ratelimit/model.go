package ratelimit

import (
	"sync"
	"time"

	"github.com/qiangli/nadir/internal/types"
)

type Model struct {
	mu        sync.Mutex
	cooldowns map[string]time.Time
	now       func() time.Time
}

func NewModel() *Model {
	return &Model{cooldowns: make(map[string]time.Time), now: time.Now}
}

// Check returns retryAfter > 0 (and ok=false) while a cooldown is
// active for the model. Otherwise ok=true.
func (m *Model) Check(model string) (time.Duration, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	until, ok := m.cooldowns[model]
	if !ok {
		return 0, true
	}
	now := m.now()
	if !now.Before(until) {
		delete(m.cooldowns, model)
		return 0, true
	}
	return until.Sub(now), false
}

// Record marks the model in cooldown until now+retryAfter. Callers
// invoke this after a 429 with a Retry-After hint.
func (m *Model) Record(model string, retryAfter time.Duration) {
	if retryAfter <= 0 {
		retryAfter = 30 * time.Second
	}
	m.mu.Lock()
	m.cooldowns[model] = m.now().Add(retryAfter)
	m.mu.Unlock()
}

var _ types.ModelRateLimiter = (*Model)(nil)
