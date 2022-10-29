import random
import time
from collections import namedtuple, deque

import numpy as np


## Application logic

def burn_cpu(secs):
    t0 = time.process_time()
    elapsed = 0
    while (elapsed <= secs):
        for _ in range(1000):
            pass
        elapsed = time.process_time() - t0


class GreenletA():
    def run(self):
        burn_cpu(0.1)


class GreenletB():
    def run(self):
        burn_cpu(0.2)


MyTuple = namedtuple("MyTuple", field_names=["field_a", "field_b", "field_c"])
buffer_size = 1000000
k_samples = 10000


class SampleA():
    def run(self):
        a = deque(maxlen=buffer_size)
        for x in range(buffer_size):
            a.append(MyTuple(x, x + 1, x + 2))
        dat = random.sample(a, k=k_samples)
        print(len(dat))


class SampleB():
    def run(self):
        a = deque(maxlen=buffer_size)
        for x in range(buffer_size):
            a.append((x, x + 1, x + 2))
        selected = np.random.choice(len(a), k_samples, False)
        dat = [a[i] for i in selected]
        print(len(dat))


a = GreenletA().run()
b = GreenletB().run()

SampleA().run()
SampleB().run()
