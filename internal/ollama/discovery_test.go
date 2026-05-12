package ollama

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestDiscoverFindsHost(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"models":[{"name":"llama3.1:8b"},{"name":"qwen2.5:7b"}]}`))
	}))
	defer srv.Close()

	d := &Discoverer{HTTP: srv.Client(), Hosts: []string{srv.URL}}
	got := d.Discover(context.Background())
	if len(got) != 1 {
		t.Fatalf("want 1 instance, got %d", len(got))
	}
	if len(got[0].Models) != 2 {
		t.Errorf("want 2 models, got %v", got[0].Models)
	}
	if got[0].BaseURL != srv.URL+"/v1" {
		t.Errorf("base url = %q", got[0].BaseURL)
	}
}

func TestDiscoverSkipsUnreachable(t *testing.T) {
	d := &Discoverer{HTTP: http.DefaultClient, Hosts: []string{"http://127.0.0.1:1"}} // closed port
	got := d.Discover(context.Background())
	if len(got) != 0 {
		t.Errorf("expected 0 reachable, got %v", got)
	}
}
