package impl

type StreamBoxRequest struct {
	RequestID string `json:"request_id" binding:"required"`
	Model     string `json:"model" binding:"required"`
	Input     Data   `json:"data" binding:"required"`
}

func (r *StreamBoxRequest) GetRequestID() string {
	return r.RequestID
}

func (r *StreamBoxRequest) GetModel() string {
	return r.Model
}

func (r *StreamBoxRequest) GetData() interface{} {
	return r.Input
}
