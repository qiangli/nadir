package embed

import (
	"strings"
	"testing"
)

// miniVocab is just enough to tokenize the test prompts. Real builds
// use the full ~30k-entry BERT vocab from tools/export-onnx.
const miniVocab = `[PAD]
[UNK]
[CLS]
[SEP]
hello
world
,
.
he
##llo
test
##ing
how
are
you
?
`

const miniCfg = `{"max_len":16,"do_lower_case":true,"cls_token":"[CLS]","sep_token":"[SEP]","pad_token":"[PAD]","unk_token":"[UNK]"}`

func newTestTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	tok, err := LoadTokenizerFromBytes([]byte(miniCfg), []byte(miniVocab))
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestEncodeShapesAndSpecials(t *testing.T) {
	tok := newTestTokenizer(t)
	ids, mask, types := tok.Encode("hello world")
	if len(ids) != 16 || len(mask) != 16 || len(types) != 16 {
		t.Fatalf("shapes wrong: %d %d %d", len(ids), len(mask), len(types))
	}
	if ids[0] != int64(tok.cls) {
		t.Errorf("ids[0] = %d, want CLS=%d", ids[0], tok.cls)
	}
	// Find SEP — it must follow the body tokens.
	sepIdx := -1
	for i, v := range ids {
		if int32(v) == tok.sep {
			sepIdx = i
			break
		}
	}
	if sepIdx <= 1 {
		t.Errorf("SEP at idx %d, want > 1", sepIdx)
	}
	// Padding past SEP must be PAD with mask=0.
	for i := sepIdx + 1; i < 16; i++ {
		if int32(ids[i]) != tok.pad || mask[i] != 0 {
			t.Errorf("padding [%d] id=%d mask=%d", i, ids[i], mask[i])
		}
	}
}

func TestEncodeLowerCases(t *testing.T) {
	tok := newTestTokenizer(t)
	idsA, _, _ := tok.Encode("HELLO world")
	idsB, _, _ := tok.Encode("hello world")
	for i := range idsA {
		if idsA[i] != idsB[i] {
			t.Errorf("lowercase divergence at %d: %d vs %d", i, idsA[i], idsB[i])
		}
	}
}

func TestEncodePunctuationSplits(t *testing.T) {
	tok := newTestTokenizer(t)
	ids, mask, _ := tok.Encode("hello, world.")
	body := []int32{}
	for i, m := range mask {
		if m == 1 && int32(ids[i]) != tok.cls && int32(ids[i]) != tok.sep {
			body = append(body, int32(ids[i]))
		}
	}
	if len(body) < 4 {
		t.Errorf("expected hello , world . → 4 tokens, got %d (%v)", len(body), body)
	}
}

func TestEncodeUnknownGoesToUNK(t *testing.T) {
	tok := newTestTokenizer(t)
	ids, mask, _ := tok.Encode("xyzzy")
	foundUNK := false
	for i, m := range mask {
		if m == 1 && int32(ids[i]) == tok.unk {
			foundUNK = true
			break
		}
	}
	if !foundUNK {
		t.Errorf("unknown word should yield UNK, got ids=%v", ids[:6])
	}
}

func TestWordpieceContinuation(t *testing.T) {
	tok := newTestTokenizer(t)
	// "testing" is "test" + "##ing" in our mini-vocab.
	pieces := tok.wordpiece("testing")
	if len(pieces) != 2 {
		t.Fatalf("expected 2 pieces, got %d (%v)", len(pieces), pieces)
	}
	if pieces[0] != tok.vocab["test"] || pieces[1] != tok.vocab["##ing"] {
		t.Errorf("pieces = %v", pieces)
	}
}

func TestEncodeTruncation(t *testing.T) {
	tok := newTestTokenizer(t)
	long := strings.Repeat("hello world ", 50)
	ids, _, _ := tok.Encode(long)
	if len(ids) != 16 {
		t.Errorf("truncation broken: len=%d", len(ids))
	}
	if int32(ids[15]) != tok.sep && int32(ids[15]) != tok.pad {
		// last slot must be SEP (if exact fit) or PAD — never a body
		// token, which would mean SEP was omitted.
		t.Errorf("last slot=%d, want SEP or PAD", ids[15])
	}
}
