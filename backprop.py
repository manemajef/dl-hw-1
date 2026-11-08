#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Homework n. 1
# 

# ## Imports
# 

# In[1]:


from typing import Protocol
import torch
from dataclasses import dataclass

Tensor = torch.Tensor  # avoid typing `torch.Tensor` at every function decleration


# ## Designing The network's inputs
# 

# ### The Activation function
# 
# All activation function follow the same `Activation` protocl - allowing the Network to not care about which activation it uses per layer.
# 

# In[2]:


class Activation(Protocol):  # Later the `MLP` class could use a `list(Activation)` type
    def f(self, z: Tensor) -> Tensor: ...
    def df(self, z: Tensor) -> Tensor: ...
    def __repr__(self) -> str:
        return type(self).__name__


class ReLU(Activation):
    def f(self, z: Tensor) -> Tensor:
        return z.clamp(min=0)

    def df(self, z: Tensor) -> Tensor:
        return (z > 0).to(z.dtype)


class Identity(Activation):
    def f(self, z: Tensor) -> Tensor:
        return z

    def df(self, z: Tensor) -> Tensor:
        return torch.ones_like(z)


class Sigmoid(Activation):
    def f(self, z: Tensor) -> Tensor:
        return 1 / (1 + torch.exp(-z))

    def df(self, z: Tensor) -> Tensor:
        s = self.f(z)
        return s * (1 - s)


# ### The Loss Function
# 
# Similar to `Activation`, we define a `Loss` protocol.
# 

# In[3]:


class Loss(Protocol):
    def verify_dim_match(self, y_hat: Tensor, y: Tensor) -> None:
        if y_hat.shape != y.shape:
            raise ValueError(
                f"Expected matching shapes, got {y_hat.shape} and {y.shape}. Go check your dimensions ;)"
            )

    def f(self, y_hat: Tensor, y: Tensor) -> Tensor: ...
    def df(self, y_hat: Tensor, y: Tensor) -> Tensor: ...


class RSS(Loss):
    def f(self, y_hat: Tensor, y: Tensor) -> Tensor:
        self.verify_dim_match(y_hat, y)
        return ((y_hat - y) ** 2).sum()

    def df(self, y_hat: Tensor, y: Tensor) -> Tensor:
        self.verify_dim_match(y_hat, y)
        return 2 * (y_hat - y)


class LogLoss(Loss):  # assumes y is a binary and y_hat in range(0,1)
    def _clamp(self, y_hat: Tensor) -> Tensor:
        return y_hat.clamp(1e-7, 1 - 1e-7)

    def f(self, y_hat: Tensor, y: Tensor) -> Tensor:
        self.verify_dim_match(y_hat, y)
        return -1 * (y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)).sum()

    def df(self, y_hat: Tensor, y: Tensor) -> Tensor:
        self.verify_dim_match(y_hat, y)
        p = self._clamp(y_hat)  # avoid ∞
        return -1 * (y / p) + (1 - y) / (1 - p)


class AbsoluteError(Loss):
    def f(self, y_hat: Tensor, y: Tensor) -> Tensor:
        self.verify_dim_match(y_hat, y)
        return torch.abs(y_hat - y).sum()

    def df(self, y_hat: Tensor, y: Tensor) -> Tensor:
        self.verify_dim_match(y_hat, y)
        return (y_hat - y).sign()


# ### Helper objects for Network cache and information
# 
# The network will use few objects to help it recive, store, and return information. We choose to store this information in `dataclass` objects, but a simple `dict` could also work. The `dataclass` makes it easier to debug, and stay type-safe.
# 
# <!-- It needs to know how many hidden layers it gets, and which actiation each has. It needs to store the $z = W^{(l)} h^{(l-1)} + b^{l}$ and $h^({l}) = \sigma(z^{l})$ from the forward cache, and  -->
# 

# In[4]:


@dataclass
class LayerSpec:
    size: int
    activation: Activation


@dataclass
class ForwardCache:
    h_prev: Tensor
    z: Tensor


@dataclass
class GradCache:
    db: Tensor
    dW: Tensor


@dataclass
class RunResult:
    x: Tensor
    y: Tensor
    y_hat: Tensor
    err: Tensor
    loss: Tensor
    grads: list[GradCache]


# ## Designing The Algorithm
# 
# We will design a `Network` object, which store ordered `Layer` objects in its `self.layers` field.
# 

# ### Layer
# 

# The `Layer` object will recive an input size, information about its size and activation function, and a seeded number generator. Uppon initilization, it will generate a $W \in \mathbb{R}^{out \times in}$ weight matrix using the provided generaotr, and create $b \in \mathbb{R}^{out} = \vec{0}$ bias. It will declare a `cache` and `grad` fields, but keep them as `None` untill forward pass.
# 
# #### Forward pass
# 
# The forward method will require the output from the previous layer, denoted as $h\_prev$. If $h\_prev$ is a vector, it is treated as a column and set $z^{(l)} = W^{(l)} \cdot h\_prev$. If $H\_prev$ is a matrix, we assume a Batch where the inputs are rows, and thus use $Z=H\_prev \cdot W^\top$. The method will store $z$ and $h\_prev$ in a `ForwardCache` object, and return $\sigma(z)$ for the next layer to use.
# 

# In[5]:


STEP_SIZE = 0.03


class Layer:
    def __init__(self, specs: LayerSpec, in_size: int, g: torch.Generator):
        self.activation = specs.activation
        self.W = torch.randn(specs.size, in_size, generator=g)
        self.b = torch.zeros(specs.size)

        self.cache: ForwardCache | None = None
        self.grad: GradCache | None = None

    def forward(self, h_prev) -> Tensor:
        z = (
            self.W @ h_prev + self.b
            if h_prev.dim() == 1  # col vec
            else h_prev @ self.W.T + self.b  # rows are the vecs
        )
        self.cache = ForwardCache(h_prev, z)
        return self.activation.f(z)

    def backward(self, dh, step=False) -> Tensor:
        if not self.cache:
            raise RuntimeError("Layer has no forward cahce")
        c = self.cache

        z, h_prev = c.z, c.h_prev
        delta = dh * self.activation.df(z)

        if h_prev.dim() == 1:  # vector (col vec)
            dW = torch.outer(delta, h_prev)
            db = delta
            prev_dh = self.W.T @ delta
        else:
            dW = delta.T @ h_prev  # batch (the rows are the vecs)
            db = delta.sum(0)
            prev_dh = delta @ self.W

        self.grad = GradCache(db, dW)

        if step:
            self.W = self.W - STEP_SIZE * dW
            self.b = self.b - STEP_SIZE * db
        return prev_dh

    def get_grad(self) -> GradCache:
        if self.grad is None:
            raise RuntimeError("Layer has no gradients; run backward first")
        return self.grad

    def clear(self):
        self.cache = None
        self.grad = None


# ### The Network
# 

# In[6]:


class Network:
    def __init__(self, in_size: int, loss: Loss, layers_specs: list[LayerSpec]):
        self.in_size = in_size
        self.depth = len(layers_specs)
        self.loss = loss
        self.layers = self.make_layers(layers_specs)

    def make_layers(self, layers_specs: list[LayerSpec]) -> list[Layer]:
        g = torch.Generator().manual_seed(42)
        layers = []
        in_size = self.in_size
        for spec in layers_specs:
            layers.append(Layer(spec, in_size, g))
            in_size = spec.size
        return layers

    def run(self, x: Tensor, y: Tensor, epocs: int = 0) -> RunResult:
        if epocs < 0:
            raise ValueError("epocs must be at least 0")
        dtype = torch.get_default_dtype()
        x = torch.as_tensor(x, dtype=dtype)
        y = torch.as_tensor(y, dtype=dtype)

        y_hat = x
        for i in range(epocs + 1):  # not required to learn; default is 1
            y_hat = x
            for l in self.layers:
                y_hat = l.forward(y_hat)
            dh = self.loss.df(y_hat, y)
            for l in reversed(self.layers):
                dh = l.backward(dh, step=i < epocs - 1)

        return RunResult(
            x,
            y,
            y_hat,
            y_hat - y,
            self.loss.f(y_hat, y),
            [l.get_grad() for l in self.layers],
        )


# ## Displaying The Network
# 

# ### Helper functions for displaying using LaTex if available
# 

# In[7]:


CAN_TEXIFY = False
try:
    from IPython import get_ipython
    from IPython.display import display, Latex

    CAN_TEXIFY = get_ipython() is not None
except ImportError:
    pass


def fmt(x) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")


def texify(t: Tensor) -> str:
    if not CAN_TEXIFY:
        return t.__repr__()
    if t.dim() == 0:
        return fmt(t.item())
    if t.dim() == 1:
        rows = [fmt(x) for x in t]  # column vector
    else:
        rows = [" & ".join(fmt(x) for x in row) for row in t]

    body = (r" \\" + "\n").join(rows)
    return r"\begin{bmatrix}" + "\n" + body + "\n" + r"\end{bmatrix}"


def partial(symb: str, l: int) -> str:
    return rf"\frac{{\partial \mathcal{{L}}}}{{\partial {symb}^{{({l})}}}}"


def show(tex: str, plain: str) -> None:
    if CAN_TEXIFY:
        display(Latex(tex))
    else:
        print(plain)


# ### Display Network's weights
# 

# In[8]:


def show_network(net: Network) -> None:
    sizes = [l.W.shape[0] for l in net.layers]

    def net_layer(i: int) -> str:
        if i < 0:
            return "x"
        l = net.layers[i]
        body = rf"{texify(l.W)}{net_layer(i - 1)} + {texify(l.b)}"
        if isinstance(l.activation, Identity):
            return body
        return rf"{l.activation!r}\left({body}\right)"

    show(
        rf"$$ \hat{{y}} = {net_layer(net.depth - 1)} $$",
        f"Network: input {net.in_size}, hidden {sizes[:-1]}, output {sizes[-1]}",
    )


# ### Display a run report
# 

# In[9]:


def report(r: RunResult) -> None:
    show(rf"$x = {texify(r.x)},\; y = {texify(r.y)}$", f"x = {r.x}\ny = {r.y}")
    show(
        rf"$\hat{{y}} = {texify(r.y_hat)},\; \text{{Error}} = {texify(r.err)},\; "
        rf"\mathcal{{L}} = {fmt(r.loss.item())}$",
        f"y_hat = {r.y_hat}\nerror = {r.err}\nloss = {r.loss.item():.4f}",
    )
    for num, g in enumerate(r.grads, start=1):
        show(
            rf"$\displaystyle {partial('W', num)} = {texify(g.dW)},\; "
            rf"{partial('b', num)} = {texify(g.db)}$",
            f"dW^({num}) = {g.dW}\ndb^({num}) = {g.db}",
        )


# ## Running
# 

# In[10]:


relu = ReLU()
loss = RSS()
layers_spec = [LayerSpec(size, relu) for size in [3, 3]] + [LayerSpec(2, Identity())]
net = Network(2, loss, layers_spec)
show_network(net)


x = torch.tensor([1.0, 1.0])
y = torch.tensor([-1.0, 0.5])
report(net.run(x, y))

