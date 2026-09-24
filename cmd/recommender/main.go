// Initial CLI scaffold for the future online recommendation engine.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"lastfm-msvd/internal/preference"
	"os"
)

func main() {
	count := flag.Int64("count", 0, "observed plays")
	kappa := flag.Float64("kappa", 40, "confidence scale")
	flag.Parse()
	p, c, err := preference.FromCount(*count, *kappa)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := json.NewEncoder(os.Stdout).Encode(map[string]any{
		"count": *count, "preference": p, "confidence": c,
	}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
