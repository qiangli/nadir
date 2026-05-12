package main

import (
	"os"
	"path/filepath"
)

// nadirDataDir resolves the on-disk root for nadir state. Honors
// NADIR_HOME when set; otherwise defaults to ~/.nadir.
func nadirDataDir() (string, error) {
	if v := os.Getenv("NADIR_HOME"); v != "" {
		return v, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".nadir"), nil
}
