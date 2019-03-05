from keras import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout, LeakyReLU

from policies.policy_BaseDQN import BaseDQN, SQUARE, SPATIAL


class DQN(BaseDQN):
    """
    A Deep Q-Learning implementation with a Multi-Layered Dense network as the Q estimator
    """

    def _additional_args(self, policy_args):
        self.flatten = SPATIAL
        self.state_radius = 6
        self.step_forward = True
        self.state_rep = SQUARE

        self.doubleDQN = True
        self.batch_size = 96

        self.epsilon_decay = 0.98
        self.min_epsilon = 0.1
        self.gamma = 0.75

        # for self
        self.dropout_rate = 0.0
        self.activation = LeakyReLU
        return policy_args

    def _build_model(self):
        model = Sequential()
        model.add(Conv1D(input_shape=self.input_shape, filters=4, kernel_size=1))
        model.add(self.activation())
        model.add(Flatten())

        model.add(Dense(units=128))
        model.add(self.activation())

        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate))

        model.add(Dense(units=128))
        model.add(self.activation())

        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate))

        model.add(Dense(units=128))
        model.add(self.activation())

        model.add(Dense(units=3))
        return model
