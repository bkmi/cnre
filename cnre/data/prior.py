from typing import Any, Iterator

from torch.utils.data import IterableDataset


class PriorSampler(IterableDataset):
    def __init__(self, proposal: Any, batch_size: int) -> None:
        super().__init__()
        self.proposal = proposal
        self.batch_size = batch_size

    def sample(self):
        while True:
            yield self.proposal.sample((self.batch_size,))

    def __iter__(self) -> Iterator:
        return self.sample()


if __name__ == "__main__":
    pass
