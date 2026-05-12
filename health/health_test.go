package health

import (
	"testing"
	"time"

	"github.com/qiangli/nadir/types"
)

func TestTrackerOrderHealthyFirst(t *testing.T) {
	tr := New()
	for range 3 {
		tr.RecordFailure("bad", types.ErrServerError)
	}
	tr.RecordSuccess("good")
	got := tr.Order([]string{"bad", "good"})
	if got[0] != "good" {
		t.Errorf("healthy should be first, got %v", got)
	}
}

func TestTrackerAvailableAfterCooldown(t *testing.T) {
	tr := New()
	for range 10 {
		tr.RecordFailure("m", types.ErrServerError)
	}
	if tr.Available("m") {
		t.Fatal("model with many failures should be unavailable")
	}
	tr.now = func() time.Time { return time.Now().Add(time.Minute) }
	if !tr.Available("m") {
		t.Fatal("after cooldown should be available")
	}
}

func TestTrackerSuccessResets(t *testing.T) {
	tr := New()
	for range 10 {
		tr.RecordFailure("m", types.ErrServerError)
	}
	tr.RecordSuccess("m")
	if !tr.Available("m") {
		t.Fatal("success should reset failure count")
	}
}
