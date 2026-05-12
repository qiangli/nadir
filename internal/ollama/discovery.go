// Package ollama probes for local Ollama instances. Default probe
// hits the standard port 11434 on localhost; callers may extend the
// candidate list to scan additional hosts (Docker, LAN peers).
package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"time"
)

type Instance struct {
	BaseURL string   `json:"base_url"`
	Models  []string `json:"models"`
}

type Discoverer struct {
	HTTP  *http.Client
	Hosts []string
}

func NewDefault() *Discoverer {
	return &Discoverer{
		HTTP:  &http.Client{Timeout: 1500 * time.Millisecond},
		Hosts: []string{"http://localhost:11434", "http://127.0.0.1:11434", "http://host.docker.internal:11434"},
	}
}

// Discover probes the candidate hosts and returns those that respond
// to /api/tags with a model list. Each probe runs sequentially —
// fan-out isn't worth it for a 3-host default.
func (d *Discoverer) Discover(ctx context.Context) []Instance {
	out := []Instance{}
	seen := make(map[string]bool)
	for _, host := range d.Hosts {
		if seen[host] {
			continue
		}
		seen[host] = true
		inst, ok := d.probe(ctx, host)
		if !ok {
			continue
		}
		out = append(out, inst)
	}
	return out
}

func (d *Discoverer) probe(ctx context.Context, host string) (Instance, bool) {
	req, err := http.NewRequestWithContext(ctx, "GET", host+"/api/tags", nil)
	if err != nil {
		return Instance{}, false
	}
	resp, err := d.HTTP.Do(req)
	if err != nil {
		return Instance{}, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return Instance{}, false
	}
	var body struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return Instance{}, false
	}
	names := make([]string, 0, len(body.Models))
	for _, m := range body.Models {
		names = append(names, m.Name)
	}
	return Instance{BaseURL: host + "/v1", Models: names}, true
}
