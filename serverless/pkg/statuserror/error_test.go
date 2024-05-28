package statuserror_test

import (
	"errors"
	"streambox/pkg/statuserror"
	"testing"
)

func TestError(t *testing.T) {
	err := statuserror.WithHttpStatusCode(
		400,
		errors.New("oops"),
	)
	var se *statuserror.HttpError
	switch {
	case errors.As(err, &se):
		t.Log(se)
	default:
		t.Fatal("should get a StatusError")
	}
}
