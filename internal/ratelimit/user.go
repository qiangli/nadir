// Package ratelimit implements per-user and per-model sliding-window
// rate limiters. The user limiter is the front-door check; the model
// limiter throttles upstream model selection so a 429 storm against
// one provider doesn't starve the rest of the chain.
package ratelimit

import (
	"sync"
	"time"

	"github.com/qiangli/nadir/internal/types"
)

type User struct {
	window time.Duration
	limit  int

	mu      sync.Mutex
	buckets map[string][]time.Time
	now     func() time.Time
}

func NewUser(window time.Duration, limit int) *User {
	if window <= 0 {
		window = time.Minute
	}
	if limit <= 0 {
		limit = 120
	}
	return &User{
		window:  window,
		limit:   limit,
		buckets: make(map[string][]time.Time),
		now:     time.Now,
	}
}

func (u *User) Check(userID string) (time.Duration, bool) {
	u.mu.Lock()
	defer u.mu.Unlock()
	now := u.now()
	cutoff := now.Add(-u.window)
	hits := u.buckets[userID]

	// Drop entries older than the window.
	i := 0
	for ; i < len(hits) && hits[i].Before(cutoff); i++ {
	}
	hits = hits[i:]

	if len(hits) >= u.limit {
		return hits[0].Add(u.window).Sub(now), false
	}
	hits = append(hits, now)
	u.buckets[userID] = hits
	return 0, true
}

var _ types.UserRateLimiter = (*User)(nil)
