import numpy as np
from keras.utils import to_categorical
from policies import base_policy as bp

MAP_VALUES = np.arange(-1, 9 + 1)
NUM_VALUES = len(MAP_VALUES)

NO = 0
FLIP = 1
ALL = 2

SPATIAL = 1
FULL = 2

NUM_PER_TYPE = 2


def normalize(matrix, direction, normalize):
    """
    Normalize direction that snake is facing, either to North only for FULL or to North/East for FLIP, by
    rotating the board/matrix
    """
    if normalize == NO:
        return matrix
    rot = 0  # direction is already 'N'
    if direction == 'E':
        rot = 1 + (1 if normalize == FLIP else 0)
    elif direction == 'S':
        rot = 2
    elif direction == 'W':
        rot = (0 if normalize == FLIP else 3)
    return np.rot90(matrix, rot)


def augment_after_normaliztion(prev_state_repr, prev_state_dir, prev_action, reward, new_state_repr, new_state_dir,
                               radius, normalize=FLIP):
    """ Augment states to visit more states by exploiting symmetries, even though we didn't actually visit them"""
    def flip_state(s, dir):
        orig_shape = s.shape
        if normalize == ALL:
            flip_ax = 1
        else:
            if dir in {'E', 'W'}:
                flip_ax = 0
            else:  # if dir in {'N', 'S'}
                flip_ax = 1
        return np.flip(s.reshape((2*radius+1, 2*radius+1, NUM_VALUES)), axis=flip_ax).reshape(orig_shape)

    def flip_action(a):
        if bp.Policy.ACTIONS[a] == 'R':
            return bp.Policy.ACTIONS.index('L')
        elif bp.Policy.ACTIONS[a] == 'L':
            return bp.Policy.ACTIONS.index('R')
        return a

    return flip_state(prev_state_repr, prev_state_dir), \
           prev_action, \
           reward, \
           flip_state(new_state_repr, new_state_dir)


class SquareAroundHeadState:
    """
    Normalized square around head, l_inf radius
    """

    def __init__(self, radius, step_forward=True, flatten=FULL, normalize=FLIP):
        self.radius = radius
        self.step_forward = step_forward
        self.flatten = flatten
        self.normalize = normalize

    def get_area(self):
        return (self.radius * 2 + 1) ** 2

    def get_shape(self):
        if self.flatten == FULL:
            return (self.get_area() * NUM_VALUES,)
        elif self.flatten == SPATIAL:
            return (self.get_area(), NUM_VALUES)
        else:  # no flattening
            diameter = 1 + 2 * self.radius
            return (diameter, diameter, NUM_VALUES)

    def get_state_repr(self, state):
        board, head = state
        head_pos, direction = head

        if self.step_forward:
            head_pos = head_pos.move(bp.Policy.TURNS[direction]['F'])

        x, y = head_pos

        # slicing along axes
        board = np.take(board, range(x - self.radius, x + self.radius + 1), axis=0, mode='wrap')
        output_state = np.take(board, range(y - self.radius, y + self.radius + 1), axis=1, mode='wrap')

        # normalize
        output_state = normalize(output_state, direction, self.normalize)

        output_state = to_categorical(output_state + 1, num_classes=NUM_VALUES)

        if self.flatten == FULL:
            output_state = output_state.reshape(-1)
        elif self.flatten == SPATIAL:
            output_state = output_state.reshape(-1, NUM_VALUES)

        return output_state


# '''
# l2 radius - euclidean distance
# '''
# class CircleAroundHeadState:
#
#     def __init__(self, radius, step_forward=True):
#         self.radius = radius
#         self.step_forward = step_forward
#
#     def get_state_repr(self, state):
#         board, head = state
#         head_pos, direction = head
#
#         n_x, n_y = board.shape
#
#         if self.step_forward:
#             head_pos = head_pos.move(bp.Policy.TURNS[direction]['F'])
#         x, y = head_pos
#
#         def get_dists(size, ind):
#             all_inds = np.arange(size)
#             return np.min([np.abs((all_inds - size) - ind),
#                            np.abs((all_inds + size) - ind),
#                            np.abs(all_inds - ind)],
#                           axis=0)
#
#         rows = get_dists(n_x, x)
#         cols = get_dists(n_y, y)
#         mask = (rows[:, None] ** 2 + cols ** 2 <= self.radius ** 2).astype(bool)
#         return to_categorical(board[mask] + 1, num_classes=NUM_VALUES).reshape(-1)


class DiamondAroundHeadState:
    """
    Normalized diamond around head, l_1 radius
    """

    def __init__(self, radius, step_forward=True, flatten=FULL, normalize=FLIP):
        self.radius = radius
        self.step_forward = step_forward
        self.flatten = flatten
        self.square_state = SquareAroundHeadState(radius=radius, step_forward=step_forward, flatten=NO,
                                                  normalize=normalize)

    def get_area(self):
        return round((self.radius * 2 + 1) ** 2 // 2 + 1)

    def get_shape(self):
        if self.flatten == FULL:
            return (self.get_area() * NUM_VALUES,)
        elif self.flatten == SPATIAL:
            return (self.get_area(), NUM_VALUES)

    def get_state_repr(self, state):
        board = self.square_state.get_state_repr(state)

        output_state = []
        for d in range(-self.radius, self.radius + 1):
            col_radius = np.abs(self.radius - np.abs(d))
            output_state += [board[d + self.radius][(self.radius) - col_radius: (self.radius + 1) + col_radius]]

        output_state = np.vstack(output_state).squeeze()

        if self.flatten == FULL:
            output_state = output_state.reshape(-1)
        elif self.flatten == SPATIAL:
            output_state = output_state.reshape(-1, NUM_VALUES)

        return output_state


def get_dists(size, ind):
    return np.abs(get_directions(size, ind))


def get_directions(size, ind):
    all_inds = np.arange(size)
    all_dirs = np.array([all_inds - size - ind, all_inds + size - ind, (all_inds - ind)])
    dirs = all_dirs[np.argmin(np.abs(all_dirs), axis=0), range(size)]
    return dirs


class RadarState:
    """
    Sort of a radar around the snake's head. Returns directions and distances to num_per_type objects of each type
    """
    def __init__(self, num_per_type, polars=True, normalize=FLIP):
        self.num_per_type = num_per_type
        self.polars = polars

        self.normalize = normalize

    def get_shape(self):
        return [NUM_VALUES, self.num_per_type, 3]

    def get_state_repr(self, state):
        board, head = state
        head_pos, direction = head

        board = normalize(board, direction, self.normalize)

        x, y = head_pos

        n_x, n_y = board.shape
        rows_dists = get_dists(n_x, x)
        cols_dists = get_dists(n_y, y)
        dists = (rows_dists[:, None] + cols_dists)

        rows_dirs = get_directions(n_x, x)
        cols_dirs = get_directions(n_y, y)

        output_state = np.ones(self.get_shape())
        for i, v in enumerate(MAP_VALUES):
            inds = np.where(board == v)
            min_inds = dists[inds].argsort()[:self.num_per_type]
            min_inds_x = inds[0][min_inds]
            min_inds_y = inds[1][min_inds]

            curr_dirs = np.stack([rows_dirs[min_inds_x], cols_dirs[min_inds_y]], axis=1) / max(n_x, n_y)
            curr_dirs = np.hstack([curr_dirs, np.linalg.norm(curr_dirs, axis=1)[..., np.newaxis]])

            # if self.polars:
            #     r = np.linalg.norm(curr_dirs, axis=1)
            #     t = np.arctan2(curr_dirs[..., 0], curr_dirs[..., 1])
            #     curr_dirs[..., 0] = r
            #     curr_dirs[..., 1] = t
            output_state[i, range(len(min_inds))] = curr_dirs

        return output_state


class DoubleStateWrapper:
    """
    State wrapper that encapsulates a combination of Square and Radar state representations
    """
    def __init__(self, state_square: SquareAroundHeadState, state_radar : RadarState):
        self.state_radar = state_radar
        self.state_square = state_square

    def get_shape(self):
        return (self.state_square.get_shape(), self.state_radar.get_shape())

    def get_state_repr(self, state):
        return [self.state_square.get_state_repr(state), self.state_radar.get_state_repr(state)]
