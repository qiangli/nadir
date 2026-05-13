// Command skillrouter-eval runs each available Embedder against a
// labeled skill-routing corpus and reports top-1 accuracy, breakdown
// by case hardness, fallthrough behavior on out-of-distribution
// prompts, and embedding-call latency.
//
// Default build runs the two pure-Go embedders (TF-IDF, Hashing). The
// `-tags onnx` build additionally evaluates MiniLM-L6-v2 if assets/
// is present.
//
// Usage:
//
//	go run ./tools/skillrouter-eval
//	go run -tags onnx ./tools/skillrouter-eval --assets ./assets
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"time"

	"github.com/qiangli/nadir/skillrouter"
)

type evalCorpus struct {
	Skills []skillCorpusEntry `json:"skills"`
	Cases  []evalCase         `json:"cases"`
}

type skillCorpusEntry struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Examples    []string `json:"examples"`
}

type evalCase struct {
	Prompt   string `json:"prompt"`
	Expected string `json:"expected"`
	Hardness string `json:"hardness"`
}

// UnmarshalJSON is custom-handled to accept null for Expected.
func (c *evalCase) UnmarshalJSON(data []byte) error {
	var raw struct {
		Prompt   string  `json:"prompt"`
		Expected *string `json:"expected"`
		Hardness string  `json:"hardness"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	c.Prompt = raw.Prompt
	c.Hardness = raw.Hardness
	if raw.Expected != nil {
		c.Expected = *raw.Expected
	}
	return nil
}

func (c evalCase) isOOD() bool { return c.Expected == "" }

type namedEmbedder struct {
	name string
	emb  skillrouter.Embedder
}

// namedMatcher wraps a fully-constructed Matcher for eval rows where
// the matcher isn't a plain Semantic (e.g., RRF fusion across two
// Semantic matchers). Confidence semantics may differ per matcher
// kind — RRF scores are rank-derived, not cosines — so the OOD
// "mean top-1" column for these rows lives on a different scale.
type namedMatcher struct {
	name    string
	matcher skillrouter.Matcher
}

type evalResult struct {
	Name       string
	N          int
	TopOneAcc  float64
	ByHardness map[string]hardnessStats
	OODStats   oodStats
	LatencyP50 time.Duration
	LatencyP95 time.Duration
}

type hardnessStats struct {
	Total   int
	Correct int
}

func (h hardnessStats) accuracy() float64 {
	if h.Total == 0 {
		return 0
	}
	return float64(h.Correct) / float64(h.Total)
}

type oodStats struct {
	N            int
	MaxScoreMean float64
	BelowThreshN int // count of OOD prompts whose top-1 score < threshold
}

func main() {
	corpusPath := flag.String("corpus", "testdata/skillrouter_eval_corpus.json", "labeled corpus path")
	assetsPath := flag.String("assets", "./assets", "ONNX assets dir (only used in onnx build)")
	threshold := flag.Float64("ood-threshold", 0.30, "top-1 cosine threshold below which OOD prompts are considered 'correctly rejected'")
	flag.Parse()

	corpus, err := loadCorpus(*corpusPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load corpus:", err)
		os.Exit(1)
	}

	skills := toSkills(corpus.Skills)
	ctx := context.Background()

	embedders := buildEmbedders(*assetsPath, skills)
	if len(embedders) == 0 {
		fmt.Fprintln(os.Stderr, "no embedders available")
		os.Exit(1)
	}

	var results []evalResult
	for _, ne := range embedders {
		r, err := runEval(ctx, ne, skills, corpus.Cases, *threshold)
		if err != nil {
			fmt.Fprintf(os.Stderr, "eval %s: %v\n", ne.name, err)
			continue
		}
		results = append(results, r)
	}

	// Composite matcher row: RRF fusion across the two pure-Go
	// embedders. Operates on rankings, not cosines, so its OOD column
	// is on a different scale (rank-fused score, theoretical max
	// 2/k=0.033 with k=60). Reported alongside for accuracy comparison.
	if rrf, err := buildRRFMatcher(ctx, skills); err == nil {
		r, err := runEvalMatcher(ctx, rrf, corpus.Cases, *threshold)
		if err == nil {
			results = append(results, r)
		}
	}

	// Cascade matcher: pure-Go TF-IDF primary + Ollama LLM rerank on
	// the uncertain shortlist. This is the "compensate with Ollama"
	// path — fast pure-Go for the easy cases, LLM semantic for the
	// hard ones.
	if cas, ok := tryOllamaCascadeMatcher(ctx, ollamaBaseURL(), ollamaCascadeLLMModel(), skills); ok {
		r, err := runEvalMatcher(ctx, cas, corpus.Cases, *threshold)
		if err == nil {
			results = append(results, r)
		}
	}

	printResults(results, *threshold)
}

func loadCorpus(path string) (*evalCorpus, error) {
	abs, _ := filepath.Abs(path)
	b, err := os.ReadFile(abs)
	if err != nil {
		return nil, err
	}
	var c evalCorpus
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, err
	}
	return &c, nil
}

func toSkills(entries []skillCorpusEntry) []skillrouter.Skill {
	out := make([]skillrouter.Skill, 0, len(entries))
	for _, e := range entries {
		out = append(out, skillrouter.Skill{
			Name:        e.Name,
			Description: e.Description,
			Examples:    e.Examples,
		})
	}
	return out
}

// buildRRFMatcher constructs the RRF-fused matcher used in the
// composite row: two Semantic matchers (TF-IDF and Hashing) combined
// via Reciprocal Rank Fusion. Returns the namedMatcher ready to run.
func buildRRFMatcher(ctx context.Context, skills []skillrouter.Skill) (namedMatcher, error) {
	tf, err := skillrouter.NewSemantic(ctx, skillrouter.NewTFIDFFromSkills(skills), skills, skillrouter.WithMinScore(0))
	if err != nil {
		return namedMatcher{}, err
	}
	hs, err := skillrouter.NewSemantic(ctx, skillrouter.NewHashEmbedder(), skills, skillrouter.WithMinScore(0))
	if err != nil {
		return namedMatcher{}, err
	}
	fused, err := skillrouter.NewFusedSemantic([]*skillrouter.Semantic{tf, hs})
	if err != nil {
		return namedMatcher{}, err
	}
	return namedMatcher{name: "RRF: TF-IDF + Hashing (pure Go)", matcher: fused}, nil
}

// runEvalMatcher is the Matcher-shaped version of runEval. It reads
// Decision.Skill and Decision.Confidence (which is the matcher-
// specific top-1 score: cosine for Semantic, RRF score for FusedSemantic).
func runEvalMatcher(ctx context.Context, nm namedMatcher, cases []evalCase, threshold float64) (evalResult, error) {
	r := evalResult{
		Name:       nm.name,
		N:          len(cases),
		ByHardness: map[string]hardnessStats{},
	}
	latencies := make([]time.Duration, 0, len(cases))
	correct := 0
	labeled := 0
	oodScores := []float64{}

	for _, tc := range cases {
		start := time.Now()
		d, err := nm.matcher.Route(ctx, tc.Prompt)
		latencies = append(latencies, time.Since(start))
		if err != nil {
			continue
		}
		var topSkill string
		var topScore float64
		if d != nil {
			topSkill = d.Skill
			topScore = d.Confidence
		}

		if tc.isOOD() {
			oodScores = append(oodScores, topScore)
			if topScore < threshold {
				r.OODStats.BelowThreshN++
			}
			r.OODStats.N++
			continue
		}

		labeled++
		stats := r.ByHardness[tc.Hardness]
		stats.Total++
		if topSkill == tc.Expected {
			stats.Correct++
			correct++
		}
		r.ByHardness[tc.Hardness] = stats
	}

	if labeled > 0 {
		r.TopOneAcc = float64(correct) / float64(labeled)
	}
	if len(oodScores) > 0 {
		var sum float64
		for _, s := range oodScores {
			sum += s
		}
		r.OODStats.MaxScoreMean = sum / float64(len(oodScores))
	}
	slices.Sort(latencies)
	if len(latencies) > 0 {
		r.LatencyP50 = latencies[len(latencies)/2]
		r.LatencyP95 = latencies[(len(latencies)*95)/100]
	}
	return r, nil
}

// runEval builds a Semantic matcher from the embedder, embeds every
// case prompt once for latency measurement, and tallies top-1 accuracy
// + by-hardness + OOD scoring.
func runEval(ctx context.Context, ne namedEmbedder, skills []skillrouter.Skill, cases []evalCase, threshold float64) (evalResult, error) {
	// MinScore=0 so Route never fellsthrough on labeled cases; we
	// measure OOD behavior separately by reading raw scores.
	primary, err := skillrouter.NewSemantic(ctx, ne.emb, skills, skillrouter.WithMinScore(0))
	if err != nil {
		return evalResult{}, err
	}

	r := evalResult{
		Name:       ne.name,
		N:          len(cases),
		ByHardness: map[string]hardnessStats{},
	}
	latencies := make([]time.Duration, 0, len(cases))
	correct := 0
	labeled := 0
	oodScores := []float64{}

	for _, tc := range cases {
		start := time.Now()
		ranked, err := primary.Rank(ctx, tc.Prompt, 1)
		latencies = append(latencies, time.Since(start))
		if err != nil {
			continue
		}
		var topSkill string
		var topScore float64
		if len(ranked) > 0 {
			topSkill = ranked[0].Skill
			topScore = ranked[0].Score
		}

		if tc.isOOD() {
			oodScores = append(oodScores, topScore)
			if topScore < threshold {
				r.OODStats.BelowThreshN++
			}
			r.OODStats.N++
			continue
		}

		labeled++
		stats := r.ByHardness[tc.Hardness]
		stats.Total++
		if topSkill == tc.Expected {
			stats.Correct++
			correct++
		}
		r.ByHardness[tc.Hardness] = stats
	}

	if labeled > 0 {
		r.TopOneAcc = float64(correct) / float64(labeled)
	}
	if len(oodScores) > 0 {
		var sum float64
		for _, s := range oodScores {
			sum += s
		}
		r.OODStats.MaxScoreMean = sum / float64(len(oodScores))
	}

	slices.Sort(latencies)
	if len(latencies) > 0 {
		r.LatencyP50 = latencies[len(latencies)/2]
		r.LatencyP95 = latencies[(len(latencies)*95)/100]
	}
	return r, nil
}

func printResults(results []evalResult, threshold float64) {
	hardness := []string{"easy", "paraphrase", "overlap"}

	fmt.Printf("# Skill-router embedder comparison\n\n")
	fmt.Printf("- runtime: %s/%s, %d CPUs\n", runtime.GOOS, runtime.GOARCH, runtime.NumCPU())
	fmt.Printf("- OOD threshold: top-1 cosine < %.2f counted as correctly rejected\n\n", threshold)

	fmt.Println("## Top-1 accuracy (in-catalog only)")
	fmt.Println()
	fmt.Println("| Embedder | Overall | Easy | Paraphrase | Overlap |")
	fmt.Println("|---|---|---|---|---|")
	for _, r := range results {
		fmt.Printf("| %s | **%.1f%%** ", r.Name, 100*r.TopOneAcc)
		for _, h := range hardness {
			s := r.ByHardness[h]
			if s.Total == 0 {
				fmt.Print("| n/a ")
			} else {
				fmt.Printf("| %.1f%% (%d/%d) ", 100*s.accuracy(), s.Correct, s.Total)
			}
		}
		fmt.Println("|")
	}

	fmt.Println("\n## Out-of-distribution behavior")
	fmt.Println()
	fmt.Println("Mean top-1 cosine on OOD prompts — *lower is better* (easier to reject). Fraction below threshold = how often Semantic would correctly Fellthrough at that threshold.")
	fmt.Println()
	fmt.Println("| Embedder | Mean top-1 (OOD) | Rejected at threshold |")
	fmt.Println("|---|---|---|")
	for _, r := range results {
		var rejected string
		if r.OODStats.N > 0 {
			rejected = fmt.Sprintf("%d/%d (%.0f%%)", r.OODStats.BelowThreshN, r.OODStats.N, 100*float64(r.OODStats.BelowThreshN)/float64(r.OODStats.N))
		} else {
			rejected = "n/a"
		}
		fmt.Printf("| %s | %.3f | %s |\n", r.Name, r.OODStats.MaxScoreMean, rejected)
	}

	fmt.Println("\n## Latency per Embed call")
	fmt.Println()
	fmt.Println("| Embedder | p50 | p95 |")
	fmt.Println("|---|---|---|")
	for _, r := range results {
		fmt.Printf("| %s | %s | %s |\n", r.Name, fmtDur(r.LatencyP50), fmtDur(r.LatencyP95))
	}
	fmt.Println()
}

func fmtDur(d time.Duration) string {
	switch {
	case d < time.Microsecond:
		return fmt.Sprintf("%dns", d.Nanoseconds())
	case d < time.Millisecond:
		return fmt.Sprintf("%.1fµs", float64(d.Nanoseconds())/1000)
	default:
		return fmt.Sprintf("%.2fms", float64(d.Nanoseconds())/1e6)
	}
}
