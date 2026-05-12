// Package modelmeta holds the static cost / context-window / vision
// capability table used by the router (vision swaps), the budget
// tracker (cost estimation), and the report renderer. The table is
// embedded so the binary is fully self-contained; updates require
// editing data.json and rebuilding.
package modelmeta

import (
	_ "embed"
	"encoding/json"
	"strings"
	"sync"
)

//go:embed data.json
var dataJSON []byte

type Model struct {
	Name        string  `json:"name"`
	InputPer1K  float64 `json:"input_per_1k"`
	OutputPer1K float64 `json:"output_per_1k"`
	Context     int     `json:"context"`
	Vision      bool    `json:"vision"`
}

type Table struct {
	models map[string]Model
}

var (
	defaultOnce sync.Once
	defaultTbl  *Table
)

// Default returns the singleton table built from embedded data.json.
func Default() *Table {
	defaultOnce.Do(func() {
		t, err := Load(dataJSON)
		if err != nil {
			defaultTbl = &Table{models: map[string]Model{}}
			return
		}
		defaultTbl = t
	})
	return defaultTbl
}

func Load(raw []byte) (*Table, error) {
	var doc struct {
		Models []Model `json:"models"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, err
	}
	m := make(map[string]Model, len(doc.Models))
	for _, model := range doc.Models {
		m[model.Name] = model
	}
	return &Table{models: m}, nil
}

// Lookup returns the model entry, falling back to a fuzzy substring
// match so caller-provided aliases ("haiku" → "claude-haiku-4-5") and
// vendor prefixes still hit the table.
func (t *Table) Lookup(name string) (Model, bool) {
	if m, ok := t.models[name]; ok {
		return m, true
	}
	n := strings.ToLower(name)
	for k, v := range t.models {
		if strings.Contains(strings.ToLower(k), n) || strings.Contains(n, strings.ToLower(k)) {
			return v, true
		}
	}
	return Model{}, false
}

// Cost returns the dollar cost for a prompt/completion token split.
// Unknown models return 0 so callers can detect "no data" via a sum check.
func (t *Table) Cost(name string, promptTokens, completionTokens int) float64 {
	m, ok := t.Lookup(name)
	if !ok {
		return 0
	}
	return (float64(promptTokens)/1000.0)*m.InputPer1K + (float64(completionTokens)/1000.0)*m.OutputPer1K
}

func (t *Table) HasVision(name string) bool {
	m, ok := t.Lookup(name)
	return ok && m.Vision
}

func (t *Table) Context(name string) int {
	m, _ := t.Lookup(name)
	return m.Context
}

// All returns the full list, sorted by input price ascending. Used by
// the report renderer.
func (t *Table) All() []Model {
	out := make([]Model, 0, len(t.models))
	for _, m := range t.models {
		out = append(out, m)
	}
	// Sort cheapest first by input_per_1k; ties broken by name.
	for i := 1; i < len(out); i++ {
		for j := i; j > 0; j-- {
			if out[j].InputPer1K < out[j-1].InputPer1K ||
				(out[j].InputPer1K == out[j-1].InputPer1K && out[j].Name < out[j-1].Name) {
				out[j], out[j-1] = out[j-1], out[j]
				continue
			}
			break
		}
	}
	return out
}
