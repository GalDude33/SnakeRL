from abc import abstractmethod

import keras
import numpy as np
from keras.optimizers import Adam
from sklearn.utils.extmath import softmax

from nn_utils.losses import huber_loss
from policies import base_policy as bp
from utils.ReplayBuffer import ReplayBuffer
from state_utils.State import SquareAroundHeadState, DiamondAroundHeadState, RadarState, \
    NO, FLIP, SPATIAL, FULL, NUM_PER_TYPE, augment_after_normaliztion, DoubleStateWrapper
from keras import Model

EPSILON = 2.0
MIN_EPSILON = 0.33
GAMMA = 0.9
BUFFER_SIZE = 6000

SQUARE = 1
DIAMOND = 2
RADAR = 3
RADAR_PLUS = 4


class RadarBaseDQN(bp.Policy):
    """
    A different version of the abstract base class that implements Deep Q-Learning and allows for customization -
    to be extended by other policies that we wrote. This version was changed to support our attempt at implementing
    a network that uses Radar + Square state representations together
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        self.huber_loss = False
        self.use_softmax_sampling = True

        self.epsilon_decay = 0.90
        self.min_epsilon = MIN_EPSILON

        self.learning_rate = 1e-4
        self.batch_size = 96

        self.state_radius = 5
        self.state_rep = SQUARE
        self.step_forward = True
        self.flatten = FULL

        self.doubleDQN = False

        self.save_model_round = 250

        self.augment_after_normaliztion = False
        policy_args = self._additional_args(policy_args)

        return policy_args

    def _save_model(self):
        self.old_model.set_weights(self.model.get_weights())

    def init_run(self):
        self.log("Starting init")
        self.r_sum = 0

        if self.state_rep == SQUARE:
            self.state_proc = SquareAroundHeadState(radius=self.state_radius,
                                                    step_forward=self.step_forward, flatten=self.flatten)
        elif self.state_rep == DIAMOND:
            self.state_proc = DiamondAroundHeadState(radius=self.state_radius,
                                                     step_forward=self.step_forward, flatten=self.flatten)
        elif self.state_rep == RADAR:
            self.state_proc = RadarState(num_per_type=NUM_PER_TYPE)

        elif self.state_rep == RADAR_PLUS:
            self.state_proc = DoubleStateWrapper(
                SquareAroundHeadState(radius=self.state_radius, step_forward=self.step_forward, flatten=self.flatten),
                RadarState(num_per_type=NUM_PER_TYPE))

        self.input_shape = self.state_proc.get_shape()

        self.model = self._build_model()
        self.model.summary()

        if self.huber_loss:
            loss = huber_loss
        else:
            loss = 'mse'

        opt = Adam(self.learning_rate)
        self.model.compile(loss=loss, optimizer=opt)

        self.old_model = keras.models.clone_model(self.model)
        self._save_model()

        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.log("Init finished!")

        self.num_of_samples = 0
        self.sum_of_loss = 0

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: {}, eps={:.2f}, "
                             "db_size={}".format(str(self.r_sum), self.epsilon, len(self.memory)), 'VALUE')
                else:
                    total_loss = self.sum_of_loss / self.num_of_samples
                    self.num_of_samples = self.sum_of_loss = 0
                    self.log("Rewards in last 100 rounds: {}, eps={:.2f}, db_size={}, loss={:.3f}".format(
                        str(self.r_sum), self.epsilon, len(self.memory), total_loss), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward
        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

        prev, actions, rewards, new = self.memory.sample(self.batch_size)

        new = [np.stack(new.T[0]), np.stack(new.T[1])]
        prev = [np.stack(prev.T[0]), np.stack(prev.T[1])]
        if self.doubleDQN:
            target = rewards + self.gamma * self.old_model.predict(new)[range(len(new[0])),
                                                                        np.argmax(self.model.predict(prev), axis=1)]
        else:
            target = rewards + self.gamma * np.amax(self.old_model.predict(new), axis=1)
        target_f = self.model.predict(prev)

        try:
            target_f[range(len(actions)), actions] = target
            hist = self.model.fit(prev, target_f, epochs=1, verbose=0, batch_size=len(prev), shuffle=True)
            self.sum_of_loss += np.sum(hist.history['loss'])
            self.num_of_samples += len(hist.history['loss'])
        except Exception as e:
            print(e)

        if round % self.save_model_round == 0 and round > 0:
            self._save_model()
        if round % 200 == 0 and round > 0 and self.epsilon > 0:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if round > self.game_duration - self.score_scope:
            # cancel exploration during "money-time"
            self.use_softmax_sampling = False
            self.epsilon = 0

        new_state_repr = self.state_proc.get_state_repr(new_state)

        if prev_state is not None:
            prev_state_repr = self.state_proc.get_state_repr(prev_state)
            self.memory.record(prev_state_repr, bp.Policy.ACTIONS.index(prev_action), reward, new_state_repr)
            if self.augment_after_normaliztion and prev_state[1][1] == new_state[1][1]:
                self.memory.record(*augment_after_normaliztion(
                    prev_state_repr, prev_state[1][1], bp.Policy.ACTIONS.index(prev_action),
                    reward, new_state_repr, new_state[1][1], self.state_radius))

        if self.use_softmax_sampling:
            return np.random.choice(bp.Policy.ACTIONS,
                                    p=softmax(self.model.predict(
                                        [new_state_repr[0][np.newaxis],
                                         new_state_repr[1][np.newaxis]]) / self.epsilon).squeeze())
        else:  # use epsilon-greedy
            if np.random.rand() < self.epsilon:
                return np.random.choice(bp.Policy.ACTIONS)
            else:
                prediction = self.model.predict(new_state_repr[np.newaxis])[0]
                action = bp.Policy.ACTIONS[np.argmax(prediction)]
                return action

    @abstractmethod
    def _build_model(self) -> Model:
        raise NotImplementedError

    @abstractmethod
    def _additional_args(self, policy_args):
        raise NotImplementedError
