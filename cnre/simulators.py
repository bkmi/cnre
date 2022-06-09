import torch


class Parabola(object):
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def g(self, theta: torch.Tensor) -> torch.Tensor:
        return theta**2

    def log_likelihood(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Normal(x, self.scale).log_prob(theta)

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Normal(self.g(theta), self.scale).sample()


class Gaussian(object):
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def g(self, theta: torch.Tensor) -> torch.Tensor:
        return theta

    def log_likelihood(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Normal(x, self.scale).log_prob(theta)

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Normal(self.g(theta), self.scale).sample()
