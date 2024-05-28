package config

import (
	"flag"
	"fmt"
	"os"
	"path"
	"streambox/pkg/env"

	"go.uber.org/config"
)

var cfgPath string

func NewProvider() (config.Provider, error) {
	flag.StringVar(&cfgPath, "config", "", "path to the config file")
	flag.Parse()

	if cfgPath == "" {
		cfgPath = os.Getenv("CONFIG_FILE")
	}

	if len(cfgPath) == 0 {
		cfgPath = path.Join(fmt.Sprintf("%s.yaml", env.Get()))
	}

	fmt.Fprintf(os.Stderr, "loading config from: %s\n", cfgPath)

	return config.NewYAML(config.File(cfgPath), config.Expand(os.LookupEnv))
}
