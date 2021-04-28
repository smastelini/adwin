import random
from river import stats

from uadwin import UADWIN

rng = random.Random(7)

x = [rng.gauss(0, 1) for _ in range(1000)]
y = []


def pred1(x):
    return 2 * x + 1


def pred2(x):
    return 2 * x + 5


for i, v in enumerate(x):
    if i < 50:
        y.append(pred1(v) + rng.gauss(0, 0.1))
    else:
        y.append(pred2(v) + rng.gauss(0, 0.1))


mean = stats.Mean()
adwin = UADWIN(0.001)

i = 1
for x_v, y_v in zip(x, y):
    in_drift, _ = adwin.update(y_v - pred1(x_v))

    if in_drift:
        print(f'Drift detected at {i}')

    i += 1
