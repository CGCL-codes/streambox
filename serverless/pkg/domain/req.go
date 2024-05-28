package domain

type Request interface {
	GetRequestID() string
	GetModel() string
	GetData() interface{}
}
