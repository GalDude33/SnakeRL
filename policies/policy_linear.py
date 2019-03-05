from keras import Sequential
from keras.layers import Dense, Flatten

from policies.policy_BaseDQN import BaseDQN, SQUARE, DIAMOND, NO, FLIP, SPATIAL, FULL, NUM_PER_TYPE


class LinearQL(BaseDQN):
    """
    A Deep Q-Learning implementation with a simple linear network as the Q estimator
    """

    def _additional_args(self, policy_args):
        self.flatten = FULL
        self.state_radius = 4
        self.step_forward = True
        self.state_rep = SQUARE
        self.batch_size = 96
        self.save_model_round = 200
        policy_args['min_epsilon'] = 0.3
        policy_args['epsilon'] = 1.0
        policy_args['gamma'] = 0.5
        return policy_args

    def _build_model(self):
        model = Sequential()
        model.add(Dense(input_shape=self.input_shape, units=3, activation=None, use_bias=True))
        return model
