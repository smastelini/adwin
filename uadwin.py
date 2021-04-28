import collections
import math
from scipy.stats import ttest_ind_from_stats as t_test

from river import stats


class UADWIN:
    def __init__(self, delta=0.05, max_buckets=5, min_samples_test=10):
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_samples_test = min_samples_test

        self._levels = collections.deque()
        self._capacities = collections.deque()
        self._total_var = stats.Var()

    @property
    def size(self):
        return self._total_var.mean.n

    @property
    def mean(self):
        return self._total_var.mean.get()

    @property
    def variance(self):
        return self._total_var.get()

    def update(self, x):
        # The level of capacity 2 ** 0 needs to be created
        if self.size == 0:
            self._levels.append(collections.deque([x]))
            self._capacities.append(1)
        else:
            self._levels[-1].append(x)

        self._total_var.update(x)
        self._compress()
        return self._detect_change(), False

    def _compress(self):
        if len(self._levels[-1]) <= self.max_buckets:
            return

        new_bucket = stats.Var()
        # Create var object using two observations
        new_bucket.update(self._levels[-1].popleft())
        new_bucket.update(self._levels[-1].popleft())

        # Only the level whose elements have capacity equals to 1 is present until now
        if len(self._levels) == 1:
            self._levels.appendleft(collections.deque([new_bucket]))
            self._capacities.appendleft(2)
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
                # Set the capacity to a value two times higher than the current maximum capacity
                self._capacities.appendleft(2 ** len(self._capacities))

    def _detect_change(self):
        if self.size < 2 * self.min_samples_test:
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
                self._capacities.popleft()

                if self.size == 0:
                    break
                continue

            if self.size < 2 * self.min_samples_test:
                break

            j = 0
            while self.size > 0 and j < len(self._levels[i]):
                w0 += self._levels[i][j]
                w1 = self._total_var - w0

                if w0.mean.n < self.min_samples_test:
                    j += 1
                    continue

                # From here onward there are not enough samples to test
                if w1.mean.n < self.min_samples_test:
                    stop_flag = True
                    break

                p_value = t_test(
                    w0.mean.get(), math.sqrt(w0.get()), w0.mean.n,
                    w1.mean.get(), math.sqrt(w1.get()), w1.mean.n,
                    equal_var=False
                ).pvalue

                if p_value <= delta:
                    change_detected = True
                    w0 = stats.Var()
                    self._total_var = w1

                    # Remove previous levels if needed
                    for _ in range(i):
                        self._levels.popleft()
                        self._capacities.popleft()
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
            self._capacities.popleft()

        return change_detected
