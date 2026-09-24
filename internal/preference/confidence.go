// Package preference defines the same confidence contract as the Python pipeline.
package preference

import (
	"fmt"
	"math"
)

func FromCount(count int64, kappa float64) (int, float64, error) {
	if count < 0 || kappa < 0 || math.IsNaN(kappa) || math.IsInf(kappa, 0) {
		return 0, 0, fmt.Errorf("count and kappa must be nonnegative and kappa finite")
	}
	p := 0
	if count > 0 {
		p = 1
	}
	return p, 1 + kappa*math.Log1p(float64(count)), nil
}
