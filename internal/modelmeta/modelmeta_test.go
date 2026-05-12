package modelmeta

import "testing"

func TestDefaultLookup(t *testing.T) {
	tbl := Default()
	m, ok := tbl.Lookup("gpt-4o-mini")
	if !ok {
		t.Fatal("gpt-4o-mini missing")
	}
	if m.Context != 128000 {
		t.Errorf("context = %d", m.Context)
	}
}

func TestFuzzyLookup(t *testing.T) {
	tbl := Default()
	// short alias should still hit via substring
	m, ok := tbl.Lookup("haiku")
	if !ok || m.Name == "" {
		t.Fatalf("haiku fuzzy = %v ok=%v", m, ok)
	}
}

func TestCost(t *testing.T) {
	tbl := Default()
	cost := tbl.Cost("gpt-4o-mini", 1000, 1000)
	want := 0.00015 + 0.0006
	if abs(cost-want) > 1e-9 {
		t.Errorf("cost = %v, want %v", cost, want)
	}
}

func TestUnknownModelZeroCost(t *testing.T) {
	tbl := Default()
	if c := tbl.Cost("nonexistent-9999", 1000, 1000); c != 0 {
		t.Errorf("unknown cost = %v, want 0", c)
	}
}

func TestAllSortedByPrice(t *testing.T) {
	tbl := Default()
	all := tbl.All()
	if len(all) < 2 {
		t.Skip("need at least 2 models")
	}
	for i := 1; i < len(all); i++ {
		if all[i].InputPer1K < all[i-1].InputPer1K {
			t.Errorf("not sorted at %d: %v < %v", i, all[i].InputPer1K, all[i-1].InputPer1K)
		}
	}
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}
