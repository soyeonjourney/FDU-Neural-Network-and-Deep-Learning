from turtle import forward
import numpy as np
from typing import List


class Tensor:
    def __init__(
        self, data: np.ndarray, grad_fn: 'Function' = None, requires_grad: bool = False
    ) -> None:
        self.data = data
        self.grad = None
        self.grad_fn = grad_fn
        self.requires_grad = requires_grad

    def __repr__(self) -> str:
        if self.data is None:
            return 'Tensor()'
        else:
            return f'Tensor(data={self.data}, grad={self.grad}, \
                grad_fn={self.grad_fn}, requires_grad={self.requires_grad})'

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def T(self) -> 'Tensor':
        def _backward():
            self.grad = out.grad.T

        out = Tensor(self.data.T, grad_fn=_backward, requires_grad=self.requires_grad)
        return out

    def __len__(self):
        return len(self.data)

    def __add__(self, tensor: 'Tensor') -> 'Tensor':
        add_fn = Add()
        return add_fn(self, tensor)

    def __neg__(self) -> 'Tensor':
        neg_fn = Neg()
        return neg_fn(self)

    def __sub__(self, tensor: 'Tensor') -> 'Tensor':
        sub_fn = Sub()
        return sub_fn(self, tensor)

    def dot(self, tensor: 'Tensor') -> 'Tensor':
        dot_fn = Dot()
        return dot_fn(self, tensor)

    def reshape(self, *shape):
        reshape_fn = Reshape()
        return reshape_fn(self, *shape)

    def backward(self) -> None:
        # Build computational graph
        graph = []
        visited = set()

        def build_graph(node: 'Tensor'):
            if node.requires_grad is True and node not in visited:
                visited.add(node)

                # Post-order traversal
                if node.grad_fn is not None:
                    for prev_node in node.grad_fn.prev:
                        build_graph(prev_node)

                graph.append(node)

        build_graph(self)

        # Backpropagate gradients
        self.grad = np.array([1.0]).reshape(1, 1)  # Create implicit gradient
        for node in reversed(graph):
            if node.grad_fn is not None:
                node.grad_fn.backward()


class Function:
    def __init__(self) -> None:
        self.prev = []

    def __call__(self, *inputs: 'Tensor') -> None:
        return self.forward(*inputs)

    def forward(self, *inputs: 'Tensor') -> None:
        raise NotImplementedError

    def backward(self, *inputs: np.ndarray) -> None:
        raise NotImplementedError

    def save_backward_node(self, tensors: List['Tensor']) -> None:
        self.prev = tensors


class Add(Function):
    def __repr__(self) -> str:
        return 'Function(Add)'

    def forward(self, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        self.save_backward_node([a, b])
        return Tensor(
            a.data + b.data,
            grad_fn=self,
            requires_grad=(a.requires_grad or b.requires_grad),
        )

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        a.grad = out
        b.grad = out


class Neg(Function):
    def __repr__(self) -> str:
        return 'Function(Neg)'

    def forward(self, a: 'Tensor') -> 'Tensor':
        self.save_backward_node([a])
        return Tensor(-a.data, grad_fn=self, requires_grad=a.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        a = self.prev[0]
        a.grad = -out


class Sub(Function):
    def __repr__(self) -> str:
        return 'Function(Sub)'

    def forward(self, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        self.save_backward_node([a, b])
        return Tensor(
            a.data - b.data,
            grad_fn=self,
            requires_grad=(a.requires_grad or b.requires_grad),
        )

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        a.grad = out
        b.grad = -out


class Dot(Function):
    def __repr__(self) -> str:
        return 'Function(Dot)'

    def forward(self, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        self.save_backward_node([a, b])
        return Tensor(
            a.data.dot(b.data),
            grad_fn=self,
            requires_grad=(a.requires_grad or b.requires_grad),
        )

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        a.grad = out.dot(b.T.data)
        b.grad = a.T.data.dot(out)


class Reshape(Function):
    def __repr__(self) -> str:
        return 'Function(Reshape)'

    def forward(self, a: 'Tensor', *shape: int or tuple) -> 'Tensor':
        self.save_backward_node([a])
        if isinstance(shape[0], int):
            return Tensor(
                a.data.reshape(*shape), grad_fn=self, requires_grad=a.requires_grad
            )
        else:
            return Tensor(
                a.data.reshape(*shape), grad_fn=self, requires_grad=a.requires_grad
            )

    def backward(self, out: np.ndarray) -> None:
        a = self.prev[0]
        a.grad = out.reshape(a.shape)
