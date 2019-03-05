from keras import Input, Model
from keras.layers import Conv1D, Flatten, Dense, Dropout, LeakyReLU, Concatenate

from policies.policy_RadarBaseDQN import RadarBaseDQN, DIAMOND, SQUARE, RADAR, NO, FLIP, SPATIAL, FULL,\
    NUM_PER_TYPE, RADAR_PLUS


class RadarPlusDQN(RadarBaseDQN):
    """
    Our attempt at implementing a more advanced network that uses Radar and Square states together
    """

    def _additional_args(self, policy_args):
        self.flatten = SPATIAL
        self.state_radius = 6
        self.step_forward = True
        self.state_rep = RADAR_PLUS

        self.doubleDQN = True
        self.batch_size = 96

        self.epsilon_decay = 0.98
        self.min_epsilon = 0.2

        # for self
        self.dropout_rate = 0.0
        self.activation = LeakyReLU
        return policy_args

    def _build_model(self):
        input_layer = Input(self.input_shape[0])
        net = self.activation()(Conv1D(filters=4, kernel_size=1)(input_layer))
        net = Flatten()(net)
        net = self.dense_layer(net, units=128)

        input_layer_radar = Input(self.input_shape[1])
        net_radar = Flatten(input_shape=self.input_shape[1])(input_layer_radar)
        net_radar = self.dense_layer(net_radar, units=64)

        net = Concatenate()([net, net_radar])
        net = self.dense_layer(net, units=128)

        Q_out = self.dense_layer(net, units=3)

        return Model(inputs=[input_layer, input_layer_radar], outputs=[Q_out])

    def dense_layer(self, input_layer, units):
        net = Dense(units=units)(input_layer)
        net = self.activation()(net)  # apply activation
        if self.dropout_rate > 0:
            net = Dropout(self.dropout_rate)(net)
        return net
