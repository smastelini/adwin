import random
from river import stats

from adwin import ADWIN

rng = random.Random(7)

x = [rng.gauss(0, 1) for _ in range(1000)]
y = []


def pred1(x):
    return 2 * x + 1


def pred2(x):
    return 2 * x + 5


for i, v in enumerate(x):
    if i < 500:
        y.append(pred1(v) + rng.gauss(0, 0.1))
    else:
        y.append(pred2(v) + rng.gauss(0, 0.1))


mean = stats.Mean()
adwin = ADWIN(0.0001)

i = 1
for x_v, y_v in zip(x, y):
    adwin.update(y_v - pred1(x_v))

    if adwin.drift_detected:
        print(f'Drift detected at {i}')
        adwin.reset()

    i += 1
