package ratelimit

import (
	"testing"
	"time"
)

func TestUserAllowsBelowLimit(t *testing.T) {
	u := NewUser(time.Minute, 3)
	for i := range 3 {
		if _, ok := u.Check("alice"); !ok {
			t.Fatalf("call %d should be allowed", i)
		}
	}
}

func TestUserBlocksAtLimit(t *testing.T) {
	u := NewUser(time.Minute, 2)
	u.Check("bob")
	u.Check("bob")
	retry, ok := u.Check("bob")
	if ok {
		t.Fatal("third call should be blocked")
	}
	if retry <= 0 {
		t.Fatalf("retry %v should be positive", retry)
	}
}

func TestUserIsolatesUsers(t *testing.T) {
	u := NewUser(time.Minute, 1)
	u.Check("alice")
	if _, ok := u.Check("bob"); !ok {
		t.Fatal("bob is a distinct user; should be allowed")
	}
}

func TestModelCooldown(t *testing.T) {
	m := NewModel()
	if _, ok := m.Check("gpt"); !ok {
		t.Fatal("fresh model should pass")
	}
	m.Record("gpt", time.Second)
	if _, ok := m.Check("gpt"); ok {
		t.Fatal("model in cooldown should be blocked")
	}

	// Force time forward.
	m.now = func() time.Time { return time.Now().Add(2 * time.Second) }
	if _, ok := m.Check("gpt"); !ok {
		t.Fatal("cooldown should have expired")
	}
}
