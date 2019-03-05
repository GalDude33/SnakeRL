from keras import Input, Model
from keras.layers import Conv1D, Flatten, Dense, Dropout, Lambda, Add, LeakyReLU
import keras.backend as K

from policies.policy_BaseDQN import BaseDQN, SQUARE, SPATIAL


class DuelDQN(BaseDQN):
    """
    A Dueling Deep Q-Learning implementation with a Multi-Layered Dense network as the Q estimator
    """

    def _additional_args(self, policy_args):
        self.flatten = SPATIAL
        self.state_radius = 5
        self.step_forward = True
        self.state_rep = SQUARE

        self.doubleDQN = True

        # for self
        self.dropout_rate = 0.0
        self.activation = LeakyReLU
        return policy_args

    def _build_model(self):
        input_layer = Input(self.input_shape)
        net = self.activation()(Conv1D(input_shape=self.input_shape, filters=4, kernel_size=1)(input_layer))
        net = Flatten()(net)

        net = self.dense_layer(net, units=128)

        streamA = self.dense_layer(net, units=128)
        streamV = self.dense_layer(net, units=128)

        Advantage = Dense(units=3)(streamA)
        Value = Dense(units=1)(streamV)

        advt = Lambda(lambda advt: advt - K.mean(advt, axis=-1, keepdims=True))(Advantage)
        value = Lambda(lambda value: K.tile(value, [1, 3]))(Value)
        Q_out = Add()([value, advt])

        return Model(inputs=[input_layer], outputs=[Q_out])

    def dense_layer(self, input_layer, units):
        net = Dense(units=units)(input_layer)
        net = self.activation()(net)  # apply activation
        if self.dropout_rate > 0:
            net = Dropout(self.dropout_rate)(net)
        return net