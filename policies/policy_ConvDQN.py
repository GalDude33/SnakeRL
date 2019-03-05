from keras import Sequential
from keras.layers import Flatten, Dense, Conv2D, BatchNormalization

from policies.policy_BaseDQN import BaseDQN, SQUARE, NO, FLIP, SPATIAL, FULL


class ConvDQN(BaseDQN):
    """
    A Deep Q-Learning implementation with a Convolutional Network as the Q estimator
    """

    def _additional_args(self, policy_args):
        self.flatten = NO

        self.state_radius = 6
        self.step_forward = True
        self.state_rep = SQUARE
        self.doubleDQN = True
        self.gamma = 0.75

        # for self
        self.batch_norm = False
        return policy_args

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(4, 1, strides=1,
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(16, 4, strides=2,
                         activation='relu', padding='same'))
        if self.batch_norm:
            model.add(BatchNormalization())
        model.add(Conv2D(32, 4, strides=2,
                         activation='relu', padding='same'))
        if self.batch_norm:
           model.add(BatchNormalization())
        model.add(Conv2D(64, 3, strides=1,
                         activation='relu'))
        if self.batch_norm:
            model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        if self.batch_norm:
            model.add(BatchNormalization())
        model.add(Dense(3, activation=None))
        return model