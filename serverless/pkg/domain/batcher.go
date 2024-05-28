package domain

import (
	"context"
)

type Batcher interface {
	SetTimeWindow(timeWindow int)
	SetBatchSize(batchSize int)
	AddOne(ctx context.Context, req Request) error
	Batch(ctx context.Context, model string, reason string) (interface{}, error)
}
