// Package health tracks per-model success/failure history. Order
// reorders a candidate chain so unhealthy models drift to the back —
// the fallback loop still tries them, just last, so a single transient
// 5xx doesn't quarantine a model.
package health

import (
	"sort"
	"sync"
	"time"

	"github.com/qiangli/nadir/types"
)

const (
	defaultCooldown    = 30 * time.Second
	defaultMaxFailures = 5
)

type Tracker struct {
	cooldown    time.Duration
	maxFailures int

	mu      sync.RWMutex
	records map[string]*record
	now     func() time.Time
}

type record struct {
	failures        int
	lastFailureKind types.ErrorKind
	lastFailureAt   time.Time
	lastSuccessAt   time.Time
}

func New() *Tracker {
	return &Tracker{
		cooldown:    defaultCooldown,
		maxFailures: defaultMaxFailures,
		records:     make(map[string]*record),
		now:         time.Now,
	}
}

func (t *Tracker) RecordSuccess(model string) {
	t.mu.Lock()
	r := t.get(model)
	r.failures = 0
	r.lastSuccessAt = t.now()
	t.mu.Unlock()
}

func (t *Tracker) RecordFailure(model string, kind types.ErrorKind) {
	t.mu.Lock()
	r := t.get(model)
	r.failures++
	r.lastFailureKind = kind
	r.lastFailureAt = t.now()
	t.mu.Unlock()
}

// Order returns candidates sorted by health. Healthy models keep their
// relative ordering; unhealthy ones move to the end but stay in the
// list — Available() is what the fallback loop uses to actually skip.
func (t *Tracker) Order(candidates []string) []string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make([]string, len(candidates))
	copy(out, candidates)
	sort.SliceStable(out, func(i, j int) bool {
		return t.score(out[i]) < t.score(out[j])
	})
	return out
}

func (t *Tracker) Available(model string) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	r, ok := t.records[model]
	if !ok {
		return true
	}
	if r.failures < t.maxFailures {
		return true
	}
	return t.now().Sub(r.lastFailureAt) > t.cooldown
}

func (t *Tracker) get(model string) *record {
	r, ok := t.records[model]
	if !ok {
		r = &record{}
		t.records[model] = r
	}
	return r
}

func (t *Tracker) score(model string) int {
	r, ok := t.records[model]
	if !ok {
		return 0
	}
	// Failures within the cooldown window penalise the model; old
	// failures fade so the model gets re-tried.
	if t.now().Sub(r.lastFailureAt) > t.cooldown {
		return 0
	}
	return r.failures
}

var _ types.ProviderHealth = (*Tracker)(nil)
