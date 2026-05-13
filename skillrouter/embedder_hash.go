package skillrouter

import (
	"context"
	"hash/fnv"
)

// HashEmbedder is a pure-Go Embedder that maps text to a fixed-dim
// vector via the "hashing trick" applied to character n-grams — the
// fastText subword approach reduced to its essence. No vocabulary
// fitting, no IDF table, no model file: every text gets the same
// dimensionality regardless of vocabulary.
//
// Mechanic: for each token, prepend '<' and append '>' (word-boundary
// markers so prefixes/suffixes count distinctly), extract all rune
// n-grams in [MinNgram, MaxNgram], hash each n-gram with FNV-1a and
// mod into the output dim, accumulate the count, then L2-normalize.
//
// Compared to TFIDF: HashEmbedder captures sub-lexical similarity —
// "review" and "reviewer" share trigrams, so cosine stays high under
// morphological variation and small typos. It cannot capture
// synonyms (no semantic knowledge) but does better than TFIDF on
// catalogs where exemplars and queries use different inflections.
type HashEmbedder struct {
	dim      int
	minNgram int
	maxNgram int
}

// HashOption configures a HashEmbedder.
type HashOption func(*HashEmbedder)

// WithHashDim sets the output dimension. Default 256 — enough headroom
// for ~200 distinct n-grams (typical for a 30-skill catalog) to land
// in mostly-distinct buckets. Larger dims reduce collision but cost
// more memory in Semantic's prototype cache. Must be > 0.
func WithHashDim(d int) HashOption {
	return func(h *HashEmbedder) {
		if d > 0 {
			h.dim = d
		}
	}
}

// WithNgramRange sets the inclusive range of character n-gram sizes
// extracted. Default [3,5]. Smaller (e.g., 2) increases recall but
// produces noisier vectors; larger (e.g., 6+) needs longer tokens
// before any n-gram fits.
func WithNgramRange(minN, maxN int) HashOption {
	return func(h *HashEmbedder) {
		if minN >= 1 && maxN >= minN {
			h.minNgram = minN
			h.maxNgram = maxN
		}
	}
}

// NewHashEmbedder builds a HashEmbedder with the given options. Zero-
// arg form returns dim=256, ngrams=[3,5].
func NewHashEmbedder(opts ...HashOption) *HashEmbedder {
	e := &HashEmbedder{dim: 256, minNgram: 3, maxNgram: 5}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

// Embed tokenizes the text the same way TFIDF does (lowercase split
// on non-alphanumeric), then hashes each rune-level n-gram into the
// output vector. L2-normalized so dot product is cosine.
func (e *HashEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	vec := make([]float32, e.dim)
	tokens := tokenizeForTFIDF(text)
	dim32 := uint32(e.dim)
	for _, tok := range tokens {
		runes := []rune("<" + tok + ">")
		for n := e.minNgram; n <= e.maxNgram; n++ {
			if len(runes) < n {
				continue
			}
			for i := 0; i+n <= len(runes); i++ {
				gram := string(runes[i : i+n])
				h := fnv.New32a()
				_, _ = h.Write([]byte(gram))
				idx := h.Sum32() % dim32
				vec[idx]++
			}
		}
	}
	l2Normalize(vec)
	return vec, nil
}

// Dim returns the configured output dimension.
func (e *HashEmbedder) Dim() int { return e.dim }

// Close is a no-op — HashEmbedder holds no external resources.
func (e *HashEmbedder) Close() error { return nil }

var _ Embedder = (*HashEmbedder)(nil)
