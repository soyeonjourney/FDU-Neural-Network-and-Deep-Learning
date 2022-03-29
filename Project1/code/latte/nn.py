import numpy as np
from typing import List, Tuple
from latte.tensor import Tensor, Function


#########################################################################################
#                                         Layer                                         #
#########################################################################################


class Module:
    def __init__(self) -> None:
        self._modules = []
        self._params = []

    def __repr__(self) -> str:
        return 'Module()'

    def __call__(self, input: 'Tensor') -> None:
        return self.forward(input)

    def forward(self, input: 'Tensor') -> None:
        # Overwritten by subclass
        raise NotImplementedError

    def parameters(self) -> List['Tensor']:
        return self._params

    def __setattr__(self, __name: str, __value) -> None:
        if isinstance(__value, Module):
            self._modules.append(__value)
            self._params.extend(__value.parameters())
        # object.__setattr__(self, __name, __value)
        super(Module, self).__setattr__(__name, __value)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__()
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * 0.01, requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        self._params.extend([self.weight, self.bias])

    def __repr__(self) -> str:
        return f'Linear({self.weight.shape[0]}, {self.weight.shape[1]})'

    def forward(self, input: 'Tensor') -> 'Tensor':
        return input.dot(self.weight) + self.bias  # x @ W + b


# TODO: complete conv2d
class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...] = (1, 1),
        padding: Tuple[int, ...] = (0, 0),
    ) -> None:
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weight and bias
        self.weight = Tensor()


#########################################################################################
#                                     Loss Function                                     #
#########################################################################################


class BCELoss(Function):
    def __repr__(self) -> str:
        return 'Function(BCELoss)'

    def forward(self, input: 'Tensor', target: 'Tensor') -> 'Tensor':
        self.save_backward_node([input, target])
        input_data = input.data
        target_data = target.data
        loss = -(
            target_data * np.log(input_data)
            + (1 - target_data) * np.log(1 - input_data)
        )

        return Tensor(np.mean(loss), grad_fn=self, requires_grad=input.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        m = a.size
        input_data = a.data
        target_data = b.data
        a.grad = (
            (-target_data / input_data + (1 - target_data) / (1 - input_data)) / m
        )


class CrossEntropyLoss(Function):
    """Combines log_softmax and nll_loss in a single function."""

    def __repr__(self) -> str:
        return 'Function(CrossEntropyLoss)'

    def forward(self, input: 'Tensor', target: 'Tensor') -> 'Tensor':
        self.save_backward_node([input, target])
        m = input.shape[0]
        input_data = input.data
        target_data = target.data

        # softmax = exp(x[target]) / sum(exp(x[i]), axis=1)
        neg_log_softmax = -input_data[range(m), target_data] + np.log(
            np.sum(np.exp(input_data), axis=1)
        )

        return Tensor(
            np.mean(neg_log_softmax), grad_fn=self, requires_grad=input.requires_grad
        )

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        m = a.shape[0]
        input_data = a.data
        target_data = b.data

        # neg_log_softmax' = softmax - 1 * (i == target)
        softmax = np.exp(input_data) / np.sum(np.exp(input_data), axis=1).reshape(-1, 1)
        softmax[range(m), target_data] -= 1
        a.grad = softmax / m
