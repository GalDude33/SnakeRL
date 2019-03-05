from collections import deque
import numpy as np


class ReplayBuffer:
    """
    Self-maintaining limited length replay-buffer, with option to append records and sample batches
    """

    def __init__(self, maxlen, prefer_new=True, new_range=300, new_rate=0.40):
        self.buffer = deque(maxlen=maxlen)

        # "prefer new" parameters
        self.prefer_new = prefer_new
        self.new_range = new_range
        self.new_rate = new_rate

    def __len__(self):
        return self.buffer.__len__()

    def record(self, prev, action, reward, new):
        self.buffer.append((prev, action, reward, new))

    def sample(self, batch_size):
        if self.buffer.__len__() > batch_size:
            if self.prefer_new:
                probs = np.ones(self.buffer.__len__())
                probs[-self.new_range:] *= (self.new_rate * self.buffer.__len__())/self.new_range
                probs /= probs.sum()
            else:
                probs = None
            batch = np.array(self.buffer)[np.random.choice(self.buffer.__len__(), batch_size, p=probs)]
        else:
            batch = np.array(list(self.buffer))

        prev = np.stack(batch[:, 0])
        actions = np.vstack(batch[:, 1]).squeeze()
        rewards = np.vstack(batch[:, 2]).squeeze()
        new = np.stack(batch[:, 3])
        return prev, actions, rewards, new
