package env

import "os"

const (
	TEST    = "test"
	PROD    = "prod"
	UNKNOWN = "unknown"

	envKey = "SERVERLESS_ENV"
)

func Get() string {
	switch os.Getenv(envKey) {
	case TEST:
		return TEST
	case PROD:
		return PROD
	default:
		return UNKNOWN
	}
}
