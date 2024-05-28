package batcher

import (
	"github.com/go-playground/validator/v10"
)

const (
	ConfigurationKey = "batcher"
)

type Config struct {
	Port int `yaml:"port" validate:"required"`
}

var _validator = validator.New()

func (c Config) Validate() error {
	return _validator.Struct(c)
}

func NewDefaultConfig() Config {
	return Config{
		Port: 8080,
	}
}
