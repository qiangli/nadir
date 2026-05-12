// Package cache holds two in-memory caches used on the hot path:
// PromptCache (model + messages → cached response, TTL + LRU bounded)
// and SessionCache (system + first-user-message → pinned model, used
// to keep multi-turn conversations on the same tier).
package cache

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sync"
	"sync/atomic"
	"time"

	"github.com/qiangli/nadir/types"
)

type Prompt struct {
	maxSize int
	ttl     time.Duration

	mu      sync.Mutex
	entries map[string]*list.Element
	order   *list.List

	hits      atomic.Uint64
	misses    atomic.Uint64
	evictions atomic.Uint64
}

type promptEntry struct {
	key     string
	resp    *types.ChatResponse
	expires time.Time
}

func NewPrompt(maxSize int, ttl time.Duration) *Prompt {
	if maxSize <= 0 {
		maxSize = 256
	}
	return &Prompt{
		maxSize: maxSize,
		ttl:     ttl,
		entries: make(map[string]*list.Element, maxSize),
		order:   list.New(),
	}
}

func (c *Prompt) Get(model string, msgs []types.Message) (*types.ChatResponse, bool) {
	key := hashKey(model, msgs)
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.entries[key]
	if !ok {
		c.misses.Add(1)
		return nil, false
	}
	e := el.Value.(*promptEntry)
	if c.ttl > 0 && time.Now().After(e.expires) {
		c.removeLocked(el)
		c.misses.Add(1)
		return nil, false
	}
	c.order.MoveToFront(el)
	c.hits.Add(1)
	return e.resp, true
}

func (c *Prompt) Put(model string, msgs []types.Message, resp *types.ChatResponse) {
	if resp == nil {
		return
	}
	key := hashKey(model, msgs)
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.entries[key]; ok {
		el.Value.(*promptEntry).resp = resp
		el.Value.(*promptEntry).expires = c.expiry()
		c.order.MoveToFront(el)
		return
	}
	el := c.order.PushFront(&promptEntry{key: key, resp: resp, expires: c.expiry()})
	c.entries[key] = el
	for c.order.Len() > c.maxSize {
		back := c.order.Back()
		if back == nil {
			break
		}
		c.removeLocked(back)
		c.evictions.Add(1)
	}
}

func (c *Prompt) Stats() types.CacheStats {
	c.mu.Lock()
	size := c.order.Len()
	c.mu.Unlock()
	return types.CacheStats{
		Size:      size,
		Hits:      c.hits.Load(),
		Misses:    c.misses.Load(),
		Evictions: c.evictions.Load(),
	}
}

func (c *Prompt) expiry() time.Time {
	if c.ttl <= 0 {
		return time.Time{}
	}
	return time.Now().Add(c.ttl)
}

func (c *Prompt) removeLocked(el *list.Element) {
	e := el.Value.(*promptEntry)
	delete(c.entries, e.key)
	c.order.Remove(el)
}

func hashKey(model string, msgs []types.Message) string {
	h := sha256.New()
	h.Write([]byte(model))
	h.Write([]byte{0})
	b, _ := json.Marshal(msgs)
	h.Write(b)
	return hex.EncodeToString(h.Sum(nil))
}

var _ types.PromptCache = (*Prompt)(nil)
