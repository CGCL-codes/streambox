package batcher

import (
	"fmt"
	"streambox/pkg/statuserror"
	"streambox/service/batcher/impl"

	"github.com/gin-gonic/gin"
)

type BatcherHandler struct {
	Batcher *impl.StreamBoxBatcher
}

func NewBatcherHandler(batcher *impl.StreamBoxBatcher) *BatcherHandler {
	return &BatcherHandler{
		Batcher: batcher,
	}
}

func (h *BatcherHandler) AddReq(ctx *gin.Context) {
	statuserror.HandleError(ctx, h.addReq)
}

func (h *BatcherHandler) addReq(ctx *gin.Context) error {
	var req impl.StreamBoxRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		return fmt.Errorf("invalid request: %v", err)
	}

	req.Input.ID = req.RequestID

	if err := h.Batcher.AddOne(ctx, &req); err != nil {
		return fmt.Errorf("add one err: %v", err)
	}
	h.Batcher.Show(ctx)
	return nil
}
