package skillrouter

import (
	"context"
	"math"
	"sort"
	"strings"
	"unicode"
)

// TFIDF is a pure-Go Embedder that fits a vocabulary + IDF table from a
// caller-supplied corpus (typically the union of all skill exemplars)
// and embeds text as an L2-normalized TF-IDF vector over that vocabulary.
//
// Quality is lexical only — it has no notion of synonyms, so prompts
// that use different vocabulary than the exemplars will score low. The
// upside: zero dependencies, zero model weights, sub-millisecond
// inference, and surprisingly competitive when skills ship 3–10 rich
// exemplars covering the expected paraphrases.
//
// Numerics match the standard sklearn TF-IDF formulation:
//
//   - TF        = count(term, doc) / sum(counts in doc)
//   - IDF       = ln((N+1) / (df+1)) + 1   (smoothed)
//   - Vector[i] = TF[i] * IDF[i], then L2-normalized
//
// Use this as a fallback Embedder for the Semantic matcher when
// libonnxruntime isn't available (e.g., a static binary deploy).
type TFIDF struct {
	vocab map[string]int
	idf   []float32
	dim   int
}

// NewTFIDF fits a TF-IDF embedder over the supplied corpus. Returns an
// embedder with Dim() == |vocabulary| ; an empty or all-stopword corpus
// produces a zero-dim embedder that returns empty vectors (Semantic
// will then score every skill at 0 and Fellthrough cleanly).
func NewTFIDF(corpus []string) *TFIDF {
	df := map[string]int{}
	for _, doc := range corpus {
		seen := map[string]struct{}{}
		for _, tok := range tokenizeForTFIDF(doc) {
			if _, ok := seen[tok]; ok {
				continue
			}
			seen[tok] = struct{}{}
			df[tok]++
		}
	}
	tokens := make([]string, 0, len(df))
	for t := range df {
		tokens = append(tokens, t)
	}
	sort.Strings(tokens) // deterministic vocabulary indexing

	vocab := make(map[string]int, len(tokens))
	idf := make([]float32, len(tokens))
	N := float64(len(corpus))
	for i, t := range tokens {
		vocab[t] = i
		idf[i] = float32(math.Log((N+1)/(float64(df[t])+1)) + 1)
	}
	return &TFIDF{vocab: vocab, idf: idf, dim: len(tokens)}
}

// NewTFIDFFromSkills is a convenience: pull exemplars + descriptions
// out of a skill catalog and fit. Use this in the same call site that
// constructs Semantic so the vocabulary is implicitly aligned with the
// catalog.
func NewTFIDFFromSkills(skills []Skill) *TFIDF {
	var corpus []string
	for _, s := range skills {
		corpus = append(corpus, s.Examples...)
		if s.Description != "" {
			corpus = append(corpus, s.Description)
		}
		if s.Name != "" {
			corpus = append(corpus, s.Name)
		}
	}
	return NewTFIDF(corpus)
}

// Embed tokenizes the text, computes its TF-IDF vector over the fitted
// vocabulary, and L2-normalizes. Out-of-vocabulary tokens are dropped
// silently — TF-IDF has no notion of subword similarity, so an unknown
// word genuinely contributes no signal.
func (t *TFIDF) Embed(_ context.Context, text string) ([]float32, error) {
	vec := make([]float32, t.dim)
	if t.dim == 0 {
		return vec, nil
	}
	counts := map[int]int{}
	inVocab := 0
	for _, tok := range tokenizeForTFIDF(text) {
		if idx, ok := t.vocab[tok]; ok {
			counts[idx]++
			inVocab++
		}
	}
	if inVocab == 0 {
		return vec, nil
	}
	invTotal := float32(1) / float32(inVocab)
	for idx, c := range counts {
		tf := float32(c) * invTotal
		vec[idx] = tf * t.idf[idx]
	}
	l2Normalize(vec)
	return vec, nil
}

// Dim reports the vocabulary size (== embedding dimension).
func (t *TFIDF) Dim() int { return t.dim }

// Close is a no-op — TFIDF holds no external resources.
func (t *TFIDF) Close() error { return nil }

// tokenizeForTFIDF lowercases and splits on non-alphanumeric. Slash
// prefixes from skill names ("/review") are stripped naturally because
// '/' is non-alphanumeric. Stopwords are NOT removed — for short skill
// exemplars IDF already downweights them, and a fixed stopword list
// would hurt non-English use cases.
func tokenizeForTFIDF(s string) []string {
	s = strings.ToLower(s)
	var out []string
	var sb strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			sb.WriteRune(r)
		} else if sb.Len() > 0 {
			out = append(out, sb.String())
			sb.Reset()
		}
	}
	if sb.Len() > 0 {
		out = append(out, sb.String())
	}
	return out
}

// l2Normalize divides vec by its L2 norm in-place. No-op on zero vectors.
func l2Normalize(vec []float32) {
	var sumSq float64
	for _, v := range vec {
		sumSq += float64(v) * float64(v)
	}
	if sumSq == 0 {
		return
	}
	norm := float32(math.Sqrt(sumSq))
	for i := range vec {
		vec[i] /= norm
	}
}

var _ Embedder = (*TFIDF)(nil)
