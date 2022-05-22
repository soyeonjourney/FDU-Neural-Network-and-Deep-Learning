import numpy as np
from typing import Tuple
from latte.tensor import Tensor, Function


#########################################################################################
#                                        Function                                       #
#########################################################################################


def relu(input: 'Tensor') -> 'Tensor':
    relu_fn = ReLU()
    return relu_fn(input)


def sigmoid(input: 'Tensor') -> 'Tensor':
    sigmoid_fn = Sigmoid()
    return sigmoid_fn(input)


def dropout(input: 'Tensor', p: float = 0.5, training: bool = True) -> 'Tensor':
    dropout_fn = Dropout(p, training)
    return dropout_fn(input)


def conv2d(
    input: 'Tensor',
    weight: 'Tensor',
    bias: 'Tensor',
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
) -> 'Tensor':
    conv2d_fn = Conv2d(stride, padding)
    return conv2d_fn(input, weight, bias)


#########################################################################################
#                                         Class                                         #
#########################################################################################


class ReLU(Function):
    def __repr__(self) -> str:
        return 'Function(ReLU)'

    def forward(self, x: 'Tensor') -> 'Tensor':
        self.save_backward_node([x])
        out = x.data
        out[out < 0] = 0
        return Tensor(out, grad_fn=self, requires_grad=x.requires_grad)

    def backward(self, out: np.ndarray):
        x = self.prev[0]
        x.data[x.data <= 0] = 0
        x.data[x.data > 0] = 1
        x.grad = x.data * out


class Sigmoid(Function):
    def __repr__(self) -> str:
        return 'Function(Sigmoid)'

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def forward(self, x: 'Tensor') -> 'Tensor':
        self.save_backward_node([x])
        out = self.sigmoid(x.data)
        return Tensor(out, grad_fn=self, requires_grad=x.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        x = self.prev[0]
        d_sigmoid = self.sigmoid(x.data) * (1 - self.sigmoid(x.data))
        x.grad = d_sigmoid * out


class Dropout(Function):
    def __init__(self, p: float = 0.5, training: bool = True) -> None:
        super().__init__()
        self.p = p
        self.training = training
        self.mask = None

    def __repr__(self) -> str:
        return 'Function(Dropout)'

    def forward(self, x: 'Tensor') -> 'Tensor':
        if self.training:
            self.save_backward_node([x])
            out = x.data
            self.mask = np.random.binomial(1, self.p, size=x.shape) / (1 - self.p)
            out = out * self.mask
            return Tensor(out, grad_fn=self, requires_grad=x.requires_grad)

        else:
            return Tensor(x.data, grad_fn=self, requires_grad=x.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        x = self.prev[0]
        x.grad = self.mask * out


class Conv2d(Function):
    def __init__(
        self, stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0)
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding

    def __repr__(self) -> str:
        return 'Function(Conv2d)'

    def forward(self, x: 'Tensor', weight: 'Tensor', bias: 'Tensor') -> 'Tensor':
        self.save_backward_node([x, weight, bias])

        batch_size, num_channels, image_h, image_w = x.shape
        _, _, kernel_h, kernel_w = weight.shape
        padding_h, padding_w = self.padding
        stride_h, stride_w = self.stride
        output_h = int((image_h + 2 * padding_h - kernel_h) / stride_h + 1)
        output_w = int((image_w + 2 * padding_w - kernel_w) / stride_w + 1)

        x = np.pad(
            x,
            (
                (0, 0),
                (0, 0),
                (padding_h, padding_h + stride_h - 1),
                (padding_w, padding_w + stride_w - 1),
            ),
            mode='constant',
            constant_values=(0,),
        )
        x_tmp = np.zeros((batch_size, num_channels, kernel_h, kernel_w, output_h, output_w))
        for i in range(kernel_h):
            i_max = i + stride_h * output_h
            for j in range(kernel_w):
                j_max = j + stride_w * output_w
                x_tmp[:, :, i, j, :, :] = x[:, :, i:i_max:stride_h, j:j_max:stride_w]

        output = np.tensordot(x_tmp, weight.data, axes=([1, 2, 3], [1, 2, 3]))
        if bias is not None:
            output += bias.data

        output = np.transpose(output, (0, 3, 1, 2))
        return Tensor(output, grad_fn=self, requires_grad=x.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        x, weight, bias = self.prev

        batch_size, num_channels, image_h, image_w = x.shape
        _, _, kernel_h, kernel_w = weight.shape

        # Gradient of x
        

#########################################################################################
#                                        Utility                                        #
#########################################################################################

