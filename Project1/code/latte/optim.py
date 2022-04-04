import numpy as np
from typing import List
from latte.tensor import Tensor


class Optimizer:
    def __init__(self, params: List['Tensor'], lr: float = 1e-3) -> None:
        self.params = params
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = 0

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """SGD with momentum."""

    def __init__(
        self, params: List['Tensor'], lr: float = 1e-3, momentum: float = 0.9
    ) -> None:
        super(SGD, self).__init__(params, lr)
        self.momentum = momentum
        self.v = [np.zeros(param.shape) for param in self.params]

    def step(self) -> None:
        for param, v in zip(self.params, self.v):
            v = self.momentum * v + self.lr * param.grad
            # `param.data -= v` is not broadcastable
            param.data = param.data - v


class Adam(Optimizer):
    def __init__(
        self,
        params: List['Tensor'],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super(Adam, self).__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = [np.zeros(param.shape) for param in self.params]
        self.v = [np.zeros(param.shape) for param in self.params]
        self.t = 0  # number of steps
        self.eps = eps

    def step(self) -> None:
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        eps = self.eps * self.t ** 0.5
        for param, m, v in zip(self.params, self.m, self.v):
            m = self.beta1 * m + (1 - self.beta1) * param.grad
            v = self.beta2 * v + (1 - self.beta2) * param.grad ** 2
            # `param.data -= lr_t * m / (np.sqrt(v) + eps)` is not broadcastable
            param.data = param.data - lr_t * m / (np.sqrt(v) + eps)
