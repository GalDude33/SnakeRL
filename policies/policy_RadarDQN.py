from keras import Sequential
from keras.engine import InputLayer
from keras.layers import Flatten, Dense, Conv2D, Dropout, LeakyReLU

from policies.policy_BaseDQN import BaseDQN, DIAMOND, SQUARE, RADAR, NO, FLIP, SPATIAL, FULL, NUM_PER_TYPE


class RadarDQN(BaseDQN):
    """
    Our attempt at implementing a network that uses only Radar State representation
    """

    def _additional_args(self, policy_args):
        #self.state_radius = 4
        #self.step_forward = True
        self.state_rep = RADAR

        self.doubleDQN = True

        # for self
        self.dropout_rate = 0.0
        self.activation = LeakyReLU
        return policy_args

    def _build_model(self):
        model = Sequential()

        model.add(InputLayer(self.input_shape))
        model.add(Conv2D(filters=64, kernel_size=1, data_format='channels_first'))
        model.add(self.activation())
        model.add(Conv2D(filters=64, kernel_size=1, data_format='channels_first'))
        model.add(self.activation())
        model.add(Conv2D(filters=16, kernel_size=1, data_format='channels_first'))
        model.add(self.activation())

        model.add(Flatten())

        model.add(Dense(units=256))
        model.add(self.activation())

        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate))

        model.add(Dense(units=256))
        model.add(self.activation())

        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate))

        model.add(Dense(units=128))
        model.add(self.activation())

        model.add(Dense(units=3))
        return model
