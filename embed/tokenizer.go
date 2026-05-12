// Package embed holds the tokenizer + ONNX session that turn a prompt
// into a 384-dim L2-normalized embedding. The tokenizer is pure Go and
// compiles into every build; the ONNX session lives behind the `onnx`
// build tag so default builds stay CGO-free and stand-alone.
package embed

import (
	"bufio"
	"encoding/json"
	"errors"
	"io"
	"os"
	"strings"
	"unicode"
)

// TokenizerConfig mirrors what tools/export-onnx/export.py writes to
// assets/tokenizer_config.json.
type TokenizerConfig struct {
	MaxLen      int    `json:"max_len"`
	DoLowerCase bool   `json:"do_lower_case"`
	CLSToken    string `json:"cls_token"`
	SEPToken    string `json:"sep_token"`
	PADToken    string `json:"pad_token"`
	UNKToken    string `json:"unk_token"`
}

// Tokenizer is a BERT WordPiece tokenizer that matches the
// preprocessing done by sentence-transformers/all-MiniLM-L6-v2 for
// short single-segment inputs. Long-form fidelity (multi-segment,
// special token handling beyond CLS/SEP) is not implemented because
// the classifier only sees a single user prompt at a time.
type Tokenizer struct {
	cfg   TokenizerConfig
	vocab map[string]int32
	cls   int32
	sep   int32
	pad   int32
	unk   int32
}

// LoadTokenizer reads vocab.txt + tokenizer_config.json from a
// directory. Both files come from tools/export-onnx/export.py.
func LoadTokenizer(assetsDir string) (*Tokenizer, error) {
	cfgRaw, err := os.ReadFile(assetsDir + "/tokenizer_config.json")
	if err != nil {
		return nil, err
	}
	var cfg TokenizerConfig
	if err := json.Unmarshal(cfgRaw, &cfg); err != nil {
		return nil, err
	}
	if cfg.MaxLen <= 0 {
		cfg.MaxLen = 128
	}

	f, err := os.Open(assetsDir + "/vocab.txt")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return loadTokenizerFromReader(cfg, f)
}

// LoadTokenizerFromBytes builds a tokenizer from in-memory blobs.
// Used in tests and when the tokenizer assets are //go:embed-ed.
func LoadTokenizerFromBytes(cfgBytes, vocabBytes []byte) (*Tokenizer, error) {
	var cfg TokenizerConfig
	if err := json.Unmarshal(cfgBytes, &cfg); err != nil {
		return nil, err
	}
	if cfg.MaxLen <= 0 {
		cfg.MaxLen = 128
	}
	return loadTokenizerFromReader(cfg, strings.NewReader(string(vocabBytes)))
}

func loadTokenizerFromReader(cfg TokenizerConfig, r io.Reader) (*Tokenizer, error) {
	t := &Tokenizer{cfg: cfg, vocab: make(map[string]int32, 30522)}
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)
	idx := int32(0)
	for scanner.Scan() {
		token := strings.TrimRight(scanner.Text(), "\r\n")
		if _, exists := t.vocab[token]; !exists {
			t.vocab[token] = idx
		}
		idx++
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	get := func(s string) (int32, error) {
		id, ok := t.vocab[s]
		if !ok {
			return 0, errors.New("vocab missing required token: " + s)
		}
		return id, nil
	}
	var err error
	if t.cls, err = get(cfg.CLSToken); err != nil {
		return nil, err
	}
	if t.sep, err = get(cfg.SEPToken); err != nil {
		return nil, err
	}
	if t.pad, err = get(cfg.PADToken); err != nil {
		return nil, err
	}
	if t.unk, err = get(cfg.UNKToken); err != nil {
		return nil, err
	}
	return t, nil
}

// Encode produces (input_ids, attention_mask, token_type_ids) for a
// single sequence, padded/truncated to MaxLen. token_type_ids are
// always 0 (single-segment input).
func (t *Tokenizer) Encode(text string) (ids, mask, types []int64) {
	maxLen := t.cfg.MaxLen
	ids = make([]int64, maxLen)
	mask = make([]int64, maxLen)
	types = make([]int64, maxLen)

	// Body tokens; reserve 2 slots for [CLS] and [SEP].
	body := t.tokenize(text)
	if len(body) > maxLen-2 {
		body = body[:maxLen-2]
	}

	ids[0] = int64(t.cls)
	mask[0] = 1
	for i, id := range body {
		ids[1+i] = int64(id)
		mask[1+i] = 1
	}
	ids[1+len(body)] = int64(t.sep)
	mask[1+len(body)] = 1
	for i := 1 + len(body) + 1; i < maxLen; i++ {
		ids[i] = int64(t.pad)
	}
	return ids, mask, types
}

// MaxLen exposes the configured sequence length.
func (t *Tokenizer) MaxLen() int { return t.cfg.MaxLen }

// tokenize is the basic-tokenizer + WordPiece pass: split on
// whitespace + punctuation, lowercase + strip-accents if configured,
// then greedy longest-match prefix per word with `##` continuation.
func (t *Tokenizer) tokenize(text string) []int32 {
	if t.cfg.DoLowerCase {
		text = strings.ToLower(text)
		text = stripAccents(text)
	}
	out := []int32{}
	for _, word := range splitBasic(text) {
		out = append(out, t.wordpiece(word)...)
	}
	return out
}

// wordpiece runs the greedy-longest-match algorithm over a single
// pre-tokenized word, returning the [UNK] id if no decomposition exists.
func (t *Tokenizer) wordpiece(word string) []int32 {
	if word == "" {
		return nil
	}
	runes := []rune(word)
	if len(runes) > 100 {
		return []int32{t.unk}
	}
	out := []int32{}
	start := 0
	for start < len(runes) {
		end := len(runes)
		var matchedID int32 = -1
		var matchedEnd int
		for end > start {
			sub := string(runes[start:end])
			if start > 0 {
				sub = "##" + sub
			}
			if id, ok := t.vocab[sub]; ok {
				matchedID = id
				matchedEnd = end
				break
			}
			end--
		}
		if matchedID < 0 {
			return []int32{t.unk}
		}
		out = append(out, matchedID)
		start = matchedEnd
	}
	return out
}

// splitBasic splits on whitespace and punctuation, mirroring BERT's
// BasicTokenizer. Each punctuation rune becomes its own token.
func splitBasic(text string) []string {
	out := []string{}
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	for _, r := range text {
		switch {
		case unicode.IsSpace(r):
			flush()
		case isPunct(r):
			flush()
			out = append(out, string(r))
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return out
}

// isPunct matches BERT's punctuation classifier: standard Unicode
// punctuation plus ASCII non-alphanumeric symbols.
func isPunct(r rune) bool {
	if unicode.IsPunct(r) {
		return true
	}
	if r >= 33 && r <= 47 {
		return true
	}
	if r >= 58 && r <= 64 {
		return true
	}
	if r >= 91 && r <= 96 {
		return true
	}
	if r >= 123 && r <= 126 {
		return true
	}
	return false
}

// stripAccents decomposes Unicode and drops combining marks, matching
// BERT's `do_lower_case=True` preprocessing.
func stripAccents(s string) string {
	var b strings.Builder
	for _, r := range s {
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}
