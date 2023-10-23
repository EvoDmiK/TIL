from torch.utils.data import sampler
import numpy as np

def InfiniteSampler(n):

    idx   = n - 1
    order = np.random.permutation(n)

    while True:

        yield order[idx]

        idx += 1
        if idx >= n:
            np.random.seed()
            order = np.random.permutation(n)
            idx   = 0



class InfiniteSamplerWrapper(sampler.Sampler):

    def __init__(self, source): self.num_samples = len(source)

    def __iter__(self): return iter(InfiniteSampler(self.num_samples))

    def __len__(self): return np.power(2, 31)