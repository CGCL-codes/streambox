package x

import (
	"streambox/pkg/log"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

func RequestDurationMiddleware() gin.HandlerFunc {
	return func(ctx *gin.Context) {
		logger := log.WithContext(ctx)
		startTime := time.Now()

		ctx.Next()

		endTime := time.Now()
		duration := endTime.Sub(startTime)

		logger.Info("ginx time observe", zap.String("method", ctx.Request.Method), zap.String("path", ctx.Request.URL.Path), zap.Int64("duration(us)", duration.Microseconds()))
	}
}
