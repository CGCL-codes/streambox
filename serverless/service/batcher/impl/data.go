package impl

type Data struct {
	ID     string
	Vector []float64 `json:"vector" binding:"required"`
}
