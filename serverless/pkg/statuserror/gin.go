package statuserror

import (
	"errors"
	"net/http"

	"github.com/gin-gonic/gin"
)

type CommonVO struct {
	Msg  string `json:"msg"`
	Code int    `json:"code"`
}

func HandleError(ctx *gin.Context, fn func(*gin.Context) error) {
	err := fn(ctx)
	if err == nil {
		return
	}

	var se *Error
	var hse *HttpError
	switch {
	case errors.As(err, &se):
		ctx.JSON(
			http.StatusOK,
			CommonVO{
				Msg:  err.Error(),
				Code: se.code,
			},
		)
	case errors.As(err, &hse):
		ctx.JSON(
			hse.code,
			CommonVO{
				Msg:  err.Error(),
				Code: hse.code,
			},
		)
	default:
		ctx.JSON(
			http.StatusInternalServerError,
			CommonVO{
				Msg: err.Error(),
			},
		)
	}
}
