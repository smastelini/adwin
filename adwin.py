import collections
import math

from river import base, stats


class ADWIN(base.DriftDetector):
    def __init__(self, delta=0.05, max_buckets=5, min_samples_test=10):
        super().__init__()
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_samples_test = min_samples_test
        # Grace period: the minimum total number of samples to perform the tests
        self._gp = 2 * self.min_samples_test
        self.reset()

    def reset(self):
        super().reset()
        self._levels = collections.deque()
        self._total_var = stats.Var()

        self._tick = 0
        self._clock = 2**self.max_buckets

        self._n_detections = 0

    @property
    def size(self):
        return self._total_var.mean.n

    @property
    def mean(self):
        return self._total_var.mean.get()

    @property
    def variance(self):
        return self._total_var.get()

    @property
    def n_detections(self):
        return self._n_detections

    def update(self, x):
        # The level of capacity 2 ** 0 needs to be created
        if self.size == 0:
            self._levels.append(collections.deque([x]))
        else:
            self._levels[-1].append(x)

        self._total_var.update(x)
        self._compress()
        self._drift_detected = self._detect_change()

        return self

    def _compress(self):
        if len(self._levels[-1]) <= self.max_buckets:
            return

        new_bucket = stats.Var()
        # Create var object using two observations
        new_bucket.update(self._levels[-1].popleft())
        new_bucket.update(self._levels[-1].popleft())

        # Only the level whose elements have capacity equals to 1 is present until now
        if len(self._levels) == 1:
            # Create a new level with double the capacity of the current one
            self._levels.appendleft(collections.deque([new_bucket]))
            return

        self._levels[-2].append(new_bucket)
        # Try to compress the remaining buckets
        for i in range(len(self._levels) - 2, -1, -1):
            # All the needed buckets are merged
            if len(self._levels[i]) <= self.max_buckets:
                break

            new_bucket = self._levels[i].popleft() + self._levels[i].popleft()

            # Add the new bucket to the previous level
            if i > 0:
                self._levels[i - 1].append(new_bucket)
            else:  # A new level needs to be created
                self._levels.appendleft(collections.deque([new_bucket]))

    def _detect_change(self):
        self._tick += 1
        if (
            not self._tick % self._clock == 0
            or self.size < self._gp
        ):
            return False

        change_detected = False
        stop_flag = False

        # Multiple testing correction
        delta = self.delta / math.log(self.size)

        w0 = stats.Var()
        i = 0
        while not stop_flag and i < len(self._levels) - 1:
            # Remove empty levels
            if len(self._levels) > 0 and len(self._levels[0]) == 0:
                self._levels.popleft()
                continue

            if self.size < self._gp:
                break

            j = 0
            while self.size > self._gp and j < len(self._levels[i]):
                w0 += self._levels[i][j]
                w1 = self._total_var - w0

                if w0.mean.n < self.min_samples_test:
                    j += 1
                    continue

                # From here onward there are not enough samples to test
                if w1.mean.n < self.min_samples_test:
                    stop_flag = True
                    break

                n0 = w0.mean.n
                n1 = w1.mean.n

                u0 = w0.mean.get()
                u1 = w1.mean.get()
                m = 1 / (1 / n0 + 1 / n1)
                e_cut = math.sqrt(
                    (2 / m) * self._total_var.get() * math.log(2 / delta)
                ) + (2 / (3 * m)) * math.log(2 / delta)

                if abs(u0 - u1) > e_cut:
                    change_detected = True
                    w0 = stats.Var()
                    self._total_var = w1

                    # Remove previous levels if needed
                    for _ in range(i):
                        self._levels.popleft()
                    i = 0

                    # Remove buckets from this level
                    for _ in range(j + 1):
                        self._levels[0].popleft()
                    j = 0
                else:
                    j += 1
            i += 1

        # Remove empty levels
        if len(self._levels) > 0 and len(self._levels[0]) == 0:
            self._levels.popleft()

        if change_detected:
            self._n_detections += 1

        return change_detected
