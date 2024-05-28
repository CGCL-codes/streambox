package log

import (
	"context"
	"fmt"
	"streambox/pkg/env"
	"streambox/pkg/util"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

var (
	_logger = zap.NewNop()
)

func init() {
	encoderConfig := zapcore.EncoderConfig{
		TimeKey:        "time",
		LevelKey:       "level",
		NameKey:        "logger",
		CallerKey:      "caller",
		MessageKey:     "msg",
		StacktraceKey:  "stacktrace",
		LineEnding:     zapcore.DefaultLineEnding,
		EncodeLevel:    zapcore.LowercaseLevelEncoder, // 小写编码器
		EncodeTime:     zapcore.ISO8601TimeEncoder,    // ISO8601 UTC 时间格式
		EncodeDuration: zapcore.SecondsDurationEncoder,
		EncodeCaller:   zapcore.FullCallerEncoder, // 全路径编码器
	}

	var logLevel zapcore.Level
	switch env.Get() {
	case env.PROD:
		logLevel = zap.InfoLevel
	default:
		logLevel = zap.DebugLevel
	}

	config := zap.Config{
		Level:            zap.NewAtomicLevelAt(logLevel), // 日志级别
		Development:      true,                           // 开发模式，堆栈跟踪
		Encoding:         "json",                         // 输出格式 console 或 json
		EncoderConfig:    encoderConfig,                  // 编码器配置
		OutputPaths:      []string{"stdout"},             // 输出到指定文件 stdout（标准输出，正常颜色） stderr（错误输出，红色）
		ErrorOutputPaths: []string{"stderr"},
	}

	// 构建日志
	logger, err := config.Build()
	if err != nil {
		panic(fmt.Sprintf("log init failed: %v", err))
	}
	logger.Info("log init success")

	_logger = logger.WithOptions(zap.WrapCore(func(c zapcore.Core) zapcore.Core {
		return zapcore.NewSamplerWithOptions(c, time.Second, 10, 1e4)
	})).With(zap.String("env", env.Get()))
}

func WithContext(ctx context.Context) *zap.Logger {
	if id := util.RequestIDFromContext(ctx); id != "" {
		return _logger.With(zap.String(util.RequestIDKey, id))
	}
	return _logger
}

func Default() *zap.Logger {
	return _logger
}
