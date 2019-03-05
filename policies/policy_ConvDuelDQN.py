from keras import Input, Model
from keras.layers import Flatten, Dense, Conv2D, Lambda, Add, BatchNormalization
import keras.backend as K

from policies.policy_BaseDQN import BaseDQN, SQUARE, NO, FLIP, SPATIAL, FULL
from policies.policy_PriorityBaseDQN import PriorBaseDQN


class ConvDuelDQN(PriorBaseDQN):
    """
    A Dueling Deep Q-Learning implementation with a Convolutional Network as the Q estimator
    """

    def _additional_args(self, policy_args):
        self.flatten = NO

        self.epsilon_decay = 0.98
        self.min_epsilon = 0.1

        self.state_radius = 6
        self.step_forward = True
        self.state_rep = SQUARE
        self.doubleDQN = True
        self.gamma = 0.75

        # for self
        self.batch_norm = False
        return policy_args

    def _build_model(self):
        inputs = Input(self.input_shape)
        net = Conv2D(4, 1, strides=1, activation='relu')(inputs)
        net = Conv2D(32, 4, strides=2, activation='relu', padding='same')(net)
        if self.batch_norm:
            net = BatchNormalization()(net)
        net = Conv2D(45, 4, strides=2, activation='relu')(net)
        if self.batch_norm:
            net = BatchNormalization()(net)
        net = Flatten()(net)

        advt = Dense(128, activation='relu')(net)
        if self.batch_norm:
            advt = BatchNormalization()(advt)
        Advantage = Dense(3)(advt)

        value = Dense(128, activation='relu')(net)
        if self.batch_norm:
            value = BatchNormalization()(value)
        Value = Dense(1)(value)

        advt = Lambda(lambda advt: advt - K.mean(advt, axis=-1, keepdims=True))(Advantage)
        value = Lambda(lambda value: K.tile(value, [1, 3]))(Value)
        Q_out = Add()([value, advt])

        return Model(inputs=[inputs], outputs=[Q_out])
