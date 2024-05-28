package impl

import (
	"context"
	"fmt"
	"streambox/pkg/domain"
	"streambox/pkg/log"
	"sync"
	"time"

	"go.uber.org/zap"
)

type StreamBoxBatcher struct {
	sync.Mutex   // 加入锁
	TimeWindow   int
	MaxBatchSize int
	Queues       map[string][]Data
}

func NewDefaultStreamBoxBatcher() *StreamBoxBatcher {
	return &StreamBoxBatcher{
		TimeWindow:   100,
		MaxBatchSize: 32,
		Queues:       map[string][]Data{},
	}
}

func (b *StreamBoxBatcher) SetTimeWindow(timeWindow int) {
	b.TimeWindow = timeWindow
}

func (b *StreamBoxBatcher) SetBatchSize(batchSize int) {
	b.MaxBatchSize = batchSize
}

func (b *StreamBoxBatcher) AddOne(ctx context.Context, req domain.Request) error {
	logger := log.WithContext(ctx)
	logger.Info("add one", zap.String("request_id", req.GetRequestID()), zap.String("model", req.GetModel()))

	b.Lock()
	defer b.Unlock()

	if _, exists := b.Queues[req.GetModel()]; !exists {
		b.Queues[req.GetModel()] = []Data{}
		go func(req domain.Request) {
			for {
				time.Sleep(time.Duration(b.TimeWindow) * time.Millisecond)
				b.Batch(ctx, req.GetModel(), "timewindow")
			}
		}(req)
	}

	b.Queues[req.GetModel()] = append(b.Queues[req.GetModel()], req.GetData().(Data))
	if len(b.Queues[req.GetModel()]) >= b.MaxBatchSize {
		b.Batch(ctx, req.GetModel(), "batchsize")
	}
	return nil
}

func (b *StreamBoxBatcher) Batch(ctx context.Context, model string, reason string) (interface{}, error) {
	logger := log.WithContext(ctx)

	var batchedReqs []Data
	b.Lock()
	if queue, exists := b.Queues[model]; exists {
		if len(queue) == 0 {
			b.Unlock()
			return nil, nil
		}
		batchedReqs = make([]Data, len(queue))
		copy(batchedReqs, queue)
		b.Queues[model] = []Data{} // 用空切片替换旧切片，以清空它
	}
	b.Unlock()

	logger.Info("batch", zap.String("model", model), zap.String("reason", reason), zap.Int("batch_size", len(batchedReqs)))
	return batchedReqs, nil
}

func (b *StreamBoxBatcher) Show(ctx context.Context) error {

	b.Lock()
	defer b.Unlock()
	for model, queue := range b.Queues {
		fmt.Println("model:", model, "len:", len(queue))
		for _, req := range queue {
			fmt.Println("id:", req.ID, "vector:", req.Vector)
		}
	}

	return nil
}
