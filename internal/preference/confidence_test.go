package preference

import (
	"math"
	"testing"
)

func TestConfidenceContract(t *testing.T) {
	for _, tc := range []struct {
		count int64
		p     int
		c     float64
	}{
		{0, 0, 1}, {1, 1, 28.725887222397812}, {9, 1, 93.10340371976183},
	} {
		p, c, err := FromCount(tc.count, 40)
		if err != nil || p != tc.p || math.Abs(c-tc.c) > 1e-10 {
			t.Fatalf("count %d: %d %g %v", tc.count, p, c, err)
		}
	}
	if _, _, err := FromCount(-1, 40); err == nil {
		t.Fatal("negative count accepted")
	}
	if _, _, err := FromCount(1, math.NaN()); err == nil {
		t.Fatal("NaN accepted")
	}
}
