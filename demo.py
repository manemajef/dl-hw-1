from typing import Protocol
import torch
from dataclasses import dataclass
from IPython.display import Math, display

Tensor = torch.Tensor


class Activation(Protocol):
    def forward(self, x: Tensor) -> Tensor: ...
    def derivative(self, x: Tensor) -> Tensor: ...


class ReLU:
    def forward(self, x: Tensor) -> Tensor:
        return x.clamp(min=0)

    def derivative(self, x: Tensor) -> Tensor:
        return (x > 0).to(x.dtype)


class Identity:
    def forward(self, x: Tensor) -> Tensor:
        return x

    def derivative(self, x: Tensor) -> Tensor:
        return torch.ones_like(x)


class Sigmoid:
    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + torch.exp(-x))

    def derivative(self, x: Tensor) -> Tensor:
        s = self.forward(x)
        return s * (1 - s)


class Loss(Protocol):
    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor: ...
    def derivative(self, y_hat: Tensor, y: Tensor) -> Tensor: ...


class RSS:
    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if y.shape != y_hat.shape:
            raise ValueError(
                f"""
                You messed up dimentions along the way...
                expected y_hat to be of shape {y.shape} but got {y_hat.shape}.
                Terminated
                """
            )
        return ((y_hat - y) ** 2).sum()


class LogLoss:
    """
    assumes y in Set(0,1) & y_hat in range(0,1)
    """


@dataclass
class Layer:
    size: int
    activation: Activation


# class MLP:
#     def __init__(self, layers)


from pydantic import InstanceOf


class Network:
    def __init__(self, in_size, loss: Loss, layers: list[Layer]):
        self.in_size = in_size
        self.layers = layers
        self.L = loss
        self.layers_num = len(layers)
        self.init_weights()

    def init_weights(self):
        g = torch.Generator().manual_seed(42)
        in_size = self.in_size
        weights: list[Tensor] = []
        biases: list[Tensor] = []
        for i in range(self.layers_num):
            out_size = self.layers[i].size
            W = torch.randn(out_size, in_size, generator=g)
            b = torch.zeros(out_size)
            weights.append(W)
            biases.append(b)
            in_size = out_size

        self.y_size = out_size
        self.weights = weights
        self.biases = biases

    def display(self):
        l = self.layers_num - 1

        def tex(t, p=2):
            t = t.reshape(-1, 1) if t.dim() == 1 else t
            body = r" \\ ".join(
                " & ".join(f"{round(v, p):g}" for v in row) for row in t.tolist()
            )
            return rf"\begin{{bmatrix}} {body} \end{{bmatrix}}"

        def build_tex(l):
            if l < 0:
                return r"\mathbf{x}"
            act = repr(self.layers[l].activation)
            W, b = self.weights[l], self.biases[l]

            inner_tex = rf"{tex(W)} {build_tex(l - 1)} + {tex(b)}"

            if l == self.layers_num - 1 and isinstance(
                self.layers[l].activation, Identity
            ):
                return inner_tex

            return rf"\operatorname{{{act}}}\left( {tex(W)} {build_tex(l - 1)} + {tex(b)} \right)"

        display(Math(r"\mathbf{y} = " + build_tex(self.layers_num - 1)))
