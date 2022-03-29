import numpy as np
from latte.tensor import Tensor, Function


def relu(input: 'Tensor') -> 'Tensor':
    relu_fn = ReLU()
    return relu_fn(input)


def sigmoid(input: 'Tensor') -> 'Tensor':
    sigmoid_fn = Sigmoid()
    return sigmoid_fn(input)


class ReLU(Function):
    def __repr__(self) -> str:
        return 'Function(ReLU)'

    def forward(self, a: 'Tensor') -> 'Tensor':
        self.save_backward_node([a])
        out = a.data
        out[out < 0] = 0
        return Tensor(out, grad_fn=self, requires_grad=a.requires_grad)

    def backward(self, out: np.ndarray):
        a = self.prev[0]
        a.data[a.data <= 0] = 0
        a.data[a.data > 0] = 1
        a.grad = a.data * out


class Sigmoid(Function):
    def __repr__(self) -> str:
        return 'Function(Sigmoid)'

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def forward(self, a: 'Tensor') -> 'Tensor':
        self.save_backward_node([a])
        out = self.sigmoid(a.data)
        return Tensor(out, grad_fn=self, requires_grad=a.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        a = self.prev[0]
        d_sigmoid = self.sigmoid(a.data) * (1 - self.sigmoid(a.data))
        a.grad = d_sigmoid * out
