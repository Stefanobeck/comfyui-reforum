"""Seed behaviour between frames. Port of seed.py of the Deforum extension for AUTOMATIC1111 (MIT, commit a7b829e)."""
import random


class SeedState:
    def __init__(self, seed, behavior="iter", iter_n=1):
        self.seed = int(seed)
        self.behavior = behavior
        self.iter_n = max(1, int(iter_n))
        self.internal = 0

    def next(self):
        if self.behavior == "iter":
            self.seed += 1 if self.internal % self.iter_n == 0 else 0
            self.internal += 1
        elif self.behavior == "ladder":
            self.seed += 2 if self.internal == 0 else -1
            self.internal = 1 if self.internal == 0 else 0
        elif self.behavior == "alternate":
            self.seed += 1 if self.internal == 0 else -1
            self.internal = 1 if self.internal == 0 else 0
        elif self.behavior in ("fixed", "schedule"):
            pass                              # 'schedule' is set from seed_schedule every frame
        else:                                 # random
            self.seed = random.randint(0, 2 ** 32 - 1)
        return self.seed


def fixed_seed(seed):
    """-1 -> a random seed, as A1111's get_fixed_seed."""
    seed = int(seed)
    if seed == -1:
        return int(random.randrange(4294967294))
    return seed
