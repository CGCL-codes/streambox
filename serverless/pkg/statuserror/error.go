package statuserror

import "fmt"

type Error struct {
	err  error
	code int
}

type HttpError struct {
	err  error
	code int
}

func WithStatusCode(code int, err error) *Error {
	return &Error{
		err:  err,
		code: code,
	}
}

func WithHttpStatusCode(code int, err error) *HttpError {
	return &HttpError{
		err:  err,
		code: code,
	}
}

func (e *Error) Code() int {
	return e.code
}

func (e *Error) Error() string {
	return fmt.Sprintf("code: %d, err: %v", e.code, e.err)
}

func (e *HttpError) Code() int {
	return e.code
}

func (e *HttpError) Error() string {
	return fmt.Sprintf("http code: %d, err: %v", e.code, e.err)
}
