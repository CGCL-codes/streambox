package main

import (
	"fmt"
	"net"
	"os"
	"os/signal"
	"streambox/pkg/config"
	"streambox/pkg/log"
	"streambox/pkg/x"
	"streambox/service/batcher"
	"streambox/service/batcher/impl"
	"syscall"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

func initConfig() (*batcher.Config, error) {
	configProvider, err := config.NewProvider()
	if err != nil {
		return nil, fmt.Errorf("config provider err: %v", err)
	}
	config := batcher.NewDefaultConfig()

	if err := configProvider.Get(batcher.ConfigurationKey).Populate(&config); err != nil {
		return nil, fmt.Errorf("populate config err: %v", err)
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("validate ginfx config: %v", err)
	}
	return &config, nil
}

func main() {
	config, err := initConfig()
	if err != nil {
		panic(err)
	}

	gin.SetMode(gin.ReleaseMode)
	var (
		engine = gin.New()
		addr   = fmt.Sprintf("0.0.0.0:%d", config.Port)
	)
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		panic(err)
	}
	config.Port = lis.Addr().(*net.TCPAddr).Port

	log.Default().Info(
		"starting gin server",
		zap.Int("port", config.Port),
	)

	g := engine.Group("api")
	g.Use(
		x.RequestDurationMiddleware(),
	)
	streamBoxBatcher := impl.NewDefaultStreamBoxBatcher()
	g.POST("/batcher", batcher.NewBatcherHandler(streamBoxBatcher).AddReq)

	go func() {
		if err := engine.RunListener(lis); err != nil {
			log.Default().Error("failed to start gin", zap.Error(err))
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Default().Info("Shutting down server...")

	if err := lis.Close(); err != nil {
		log.Default().Error("failed to close listener", zap.Error(err))
	}
}
