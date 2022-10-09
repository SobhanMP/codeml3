import itertools as it
from random import shuffle


args = {
    'lr': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001],
    'epochs': [100, 200, 300, 500, 700, 1000],
    'negative_sample_rate': [1, 2, 3, 5, 10],
    'batch_size': [32, 64, 128, 256, 512],
    'step_size': [50, 100, 200],
    'step_gamma': [0.1, 0.5, 1],
    'noise_df': [True, False],
    'sigma': [0.0, 0.0, 0.1, 0.5, 1.0, 2.0],
    'coinflip': [0.0, 0.1, 0.3, 0.5, 1.0]
}
largs = list(args.items())
keys = [k for (k, v) in largs]
values = [v for (k, v) in largs]
values = list(it.product(*values))
shuffle(values)
for vs in values:
    print(" ".join(f"--{k}={v}"for (k, v) in zip(keys, vs)))

