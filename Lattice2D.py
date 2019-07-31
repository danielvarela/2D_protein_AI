# -*- coding: utf-8 -*-

"""
Implements the 2D Lattice Environment
"""
# Import gym modules
import sys
import math
from math import floor
from collections import OrderedDict

import gym
from gym import (spaces, utils, logger)
import numpy as np
from six import StringIO


from gym_lattice.envs import RenderProt

# Human-readable
ACTION_TO_STR = {
    0 : 'L', 1 : 'D',
    2 : 'U', 3 : 'R'}

POLY_TO_INT = {
    'H' : 1, 'P' : -1
}

class Residue():
    def __init__(self, res_pos, poly_, new_coords):
        self.idx = res_pos
        self.poly = poly_
        self.coords = new_coords

    def set_coords(self, new_coords):
        self.coords = new_coords
    

class Lattice2D(gym.Env):
    """A 2-dimensional lattice environment from Dill and Lau, 1989
    [dill1989lattice]_.

    It follows an absolute Cartesian coordinate system, the location of
    the polymer is stated independently from one another. Thus, we have
    four actions (left, right, up, and down) and a chance of collision.

    The environment will first place the initial polymer at the origin. Then,
    for each step, agents place another polymer to the lattice. An episode
    ends when all polymers are placed, i.e. when the length of the action
    chain is equal to the length of the input sequence minus 1. We then
    compute the reward using the energy minimization rule while accounting
    for the collisions and traps.

    Attributes
    ----------
    seq : str
        Polymer sequence describing a particular protein.
    state : OrderedDict
        Dictionary of the current polymer chain with coordinates and
        polymer type (H or P).
    actions : list
        List of actions performed by the model.
    collisions : int
        Number of collisions incurred by the model.
    trapped : int
        Number of times the agent was trapped.
    grid_length : int
        Length of one side of the grid.
    midpoint : tuple
        Coordinate containing the midpoint of the grid.
    grid : numpy.ndarray
        Actual grid containing the polymer chain.

    .. [dill1989lattice] Lau, K.F., Dill, K.A.: A lattice statistical
    mechanics model of the conformational and se quence spaces of proteins.
    Marcromolecules 22(10), 3986–3997 (1989)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, seq, collision_penalty=-2, trap_penalty=0.5):
        """Initializes the lattice

        Parameters
        ----------
        seq : str, must only consist of 'H' or 'P'
            Sequence containing the polymer chain.
        collision_penalty : int, must be a negative value
            Penalty incurred when the agent made an invalid action.
            Default is -2.
        trap_penalty : float, must be between 0 and 1
            Penalty incurred when the agent is trapped. Actual value is
            computed as :code:`floor(length_of_sequence * trap_penalty)`
            Default is -2.

        Raises
        ------
        AssertionError
            If a certain polymer is not 'H' or 'P'
        """
        try:
            if not set(seq.upper()) <= set('HP'):
                raise ValueError("%r (%s) is an invalid sequence" % (seq, type(seq)))
            self.seq = seq.upper()
        except AttributeError:
            logger.error("%r (%s) must be of type 'str'" % (seq, type(seq)))
            raise

        try:
            if collision_penalty >= 0:
                raise ValueError("%r (%s) must be negative" %
                                 (collision_penalty, type(collision_penalty)))
            if not isinstance(collision_penalty, int):
                raise ValueError("%r (%s) must be of type 'int'" %
                                 (collision_penalty, type(collision_penalty)))
            self.collision_penalty = collision_penalty
        except TypeError:
            logger.error("%r (%s) must be of type 'int'" %
                         (collision_penalty, type(collision_penalty)))
            raise

        try:
            if not 0 < trap_penalty < 1:
                raise ValueError("%r (%s) must be between 0 and 1" %
                                 (trap_penalty, type(trap_penalty)))
            self.trap_penalty = trap_penalty
        except TypeError:
            logger.error("%r (%s) must be of type 'float'" %
                         (trap_penalty, type(trap_penalty)))
            raise

        
        # Grid attributes
        self.grid_length = int(2 * (len(seq) + 1))
        self.midpoint = (int(len(seq)), int(len(seq)) )
        self.grid = np.zeros(shape=(self.grid_length, self.grid_length), dtype=int)

        # Automatically assign first element into grid
        self.grid[self.midpoint] = POLY_TO_INT[self.seq[0]]

        res_0 = Residue(0, self.seq[0], (0,0))
        # Define action-observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-2, high=1,
                                            shape=(self.grid_length, self.grid_length),
                                            dtype=int)

        #self.render_server = RenderProt.RenderProt(self.seq)

    def _compute_state(self, actions):
        state = OrderedDict({0 : Residue(0, self.seq[0], (0,0) )})
        for i in range(0, len(actions)):
            action = actions[i]
            (x, y) = state[list(state.keys())[-1]].coords
            # Get all adjacent coords and next move based on action
            adj_coords = self._get_adjacent_coords((x, y))
            next_move = adj_coords[action]
            #state.update({next_move : self.seq[i+1]})
            state.update({i+1 : Residue(i+1, self.seq[i+1], next_move)})
            
        reward, energy_info = self._compute_reward(state)
        return state, reward, energy_info
 
    def reset(self):
        """Resets the environment"""
        self.grid = np.zeros(shape=(self.grid_length, self.grid_length), dtype=int)
        # Automatically assign first element into grid
        self.grid[self.midpoint] = POLY_TO_INT[self.seq[0]]
        return self.grid

    def render(self, actions, last_action = None, mode='human'):
        state,reward,energy_info = self._compute_state(actions) 
        self.grid = self._draw_grid(state)
        """Renders the environment"""
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        desc = self.grid.astype(str)
        # Convert everything to human-readable symbols
        desc[desc == '0'] = '*'
        desc[desc == '1'] = 'H'
        desc[desc == '-1'] = 'P'
        # Obtain all x-y indices of elements
        x_free, y_free = np.where(desc == '*')
        x_h, y_h = np.where(desc == 'H')
        x_p, y_p = np.where(desc == 'P')

        all_coords = [res.coords for i, res in state.items()] 
        points = [(x,y,0) for (x,y) in all_coords]
        #self.render_server.render( points, 0)
        # Decode if possible
        desc.tolist()
        try:
            desc = [[c.decode('utf-8') for c in line] for line in desc]
        except AttributeError:
            pass

        # All unfilled spaces are gray
        for x, y in zip(x_free, y_free):
            desc[x][y] = utils.colorize(desc[x][y], "gray")


        # midpoint, aka the N-terminus point, is marked as red 
        desc[self.midpoint[0]][self.midpoint[1]] = utils.colorize(desc[self.midpoint[0]][self.midpoint[1]], "red", bold=True) 
        # All hydrophobic molecules are bold-green
        for x, y in zip(x_h, y_h):
            if not((x == self.midpoint[0]) and (y == self.midpoint[1])):
                desc[x][y] = utils.colorize(desc[x][y], "green", bold=True)

        # All polar molecules are cyan
        for x,y in zip(x_p, y_p):
            desc[x][y] = utils.colorize(desc[x][y], "cyan")

        # Provide prompt for last action
        if last_action is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Up", "Right"][last_action]))
        else:
            outfile.write("\n")

        # Draw desc
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile

    def _get_adjacent_coords(self, coords):
        """Obtains all adjacent coordinates of the current position

        Parameters
        ----------
        coords : 2-tuple
            Coordinates (X-y) of the current position

        Returns
        -------
        dictionary
            All adjacent coordinates
        """
        x, y = coords
        adjacent_coords = {
            0 : (x - 1, y),
            1 : (x, y - 1),
            2 : (x, y + 1),
            3 : (x + 1, y),
        }

        return adjacent_coords

    def _get_grid(self, chain, res_pos):
        lim = 10
        grid = np.zeros(shape=(lim, lim), dtype=int)
        new_midpoint = ( int(lim/2), int(lim/2) )
        main_x, main_y = tuple(sum(x) for x in zip(new_midpoint,
                                                   list(chain.items())[res_pos][1].coords ))
        for i, res in chain.items():
            coord = res.coords
            poly = res.poly
            trans_x, trans_y = tuple(sum(x) for x in zip(new_midpoint, coord))
            # Recall that a numpy array works by indexing the rows first
            # before the columns, that's why we interchange.
            if (abs(trans_x) < lim) and (abs(trans_y) < lim):
                grid[(trans_y, trans_x)] = POLY_TO_INT[poly]

        return np.flipud(grid)


    def _draw_grid(self, chain):
        """Constructs a grid with the current chain

        Parameters
        ----------
        chain : OrderedDict
            Current chain/state

        Returns
        -------
        numpy.ndarray
            Grid of shape :code:`(n, n)` with the chain inside
        """
        self.grid = np.zeros(shape=(self.grid_length, self.grid_length), dtype=int)
        for i, res in chain.items():
            coord = res.coords
            poly = res.poly
            trans_x, trans_y = tuple(sum(x) for x in zip(self.midpoint, coord))
            # Recall that a numpy array works by indexing the rows first
            # before the columns, that's why we interchange.
            self.grid[(trans_y, trans_x)] = POLY_TO_INT[poly]

        return np.flipud(self.grid)

    def _compute_reward(self, state):
        """Computes the reward for a given time step

        For every timestep, we compute the reward using the following function:

        .. code-block:: python

            reward_t = state_reward 
                       + collision_penalty
                       + actual_trap_penalty

        The :code:`state_reward` is only computed at the end of the episode
        (Gibbs free energy) and its value is :code:`0` for every timestep
        before that.

        The :code:`collision_penalty` is given when the agent makes an invalid
        move, i.e. going to a space that is already occupied.

        The :code:`actual_trap_penalty` is computed whenever the agent
        completely traps itself and has no more moves available. Overall, we
        still compute for the :code:`state_reward` of the current chain but
        subtract that with the following equation:
        :code:`floor(length_of_sequence * trap_penalty)`
        try:

        Parameters
        ----------
        is_trapped : bool
            Signal indicating if the agent is trapped.
        done : bool
            Done signal
        collision : bool
            Collision signal

        Returns
        -------
        int
            Reward function
        """
        #state_reward = self._compute_free_energy(state) if not done else 0
        is_trapped = False
        done = False
        collision = False
        state_reward = self._compute_free_energy(state)
        all_coords = [res.coords for i, res in state.items()]
        set_all_coords = set(all_coords)
        has_collision = False
        collision_penalty = 0
        if (len(set_all_coords) < len(all_coords)):
            has_collision = True
            collision_penalty = ( len(all_coords) - len(set_all_coords))
        collision_penalty = self.collision_penalty * collision_penalty if has_collision else 0
        #actual_trap_penalty = -floor(len(self.seq) * self.trap_penalty) if is_trapped else 0
        #actual_trap_penalty = 0
        # pen = 0
        # for a in range(1, len(self.actions) ):
        #     pen += 1 if self.actions[a] == self.actions[a-1] else 0
        actual_trap_penalty = 0
        #actual_trap_penalty = min(-2, actual_trap_penalty)
        # Compute reward at timestep, the state_reward is originally
        # negative (Gibbs), so we invert its sign.
        #reward = - state_reward + collision_penalty + actual_trap_penalty
        reward = state_reward - collision_penalty * 20 - actual_trap_penalty
        energy_info = {
            "gibbs_energy": state_reward,
            "collisions" : ( len(all_coords) - len(set_all_coords)),
            "collision_penalty" : collision_penalty,
            "trap_penalty" : actual_trap_penalty,
            "done" : done
        }
        return reward, energy_info

    def _compute_free_energy(self, chain):
        """Computes the Gibbs free energy given the lattice's state

        The free energy is only computed at the end of each episode. This
        follow the same energy function given by Dill et. al.
        [dill1989lattice]_

        Recall that the goal is to find the configuration with the lowest
        energy.

        .. [dill1989lattice] Lau, K.F., Dill, K.A.: A lattice statistical
        mechanics model of the conformational and se quence spaces of proteins.
        Marcromolecules 22(10), 3986–3997 (1989)

        Parameters
        ----------
        chain : OrderedDict
            Current chain in the lattice

        Returns
        -------
        int
            Computed free energy
        """
        h_polymers = [ (x,k) for x,k in chain.items() if k.poly == 'H']
        h_pairs = [(val_1.coords, val_2.coords) for x,val_1 in h_polymers for
                   y,val_2 in h_polymers if (y > x)]
        # Compute distance between all hydrophobic pairs
        h_adjacent = []
        for pair in h_pairs:
            #dist = np.linalg.norm(np.subtract(pair[0], pair[1]))
            #dist = math.hypot((pair[0][0] - pair[1][0]) , (pair[0][1]-pair[1][1]) )
            dist = math.sqrt((pair[0][0] - pair[1][0]) ** 2 + (pair[0][1]-pair[1][1]) ** 2)
            if dist == 1.0: # adjacent pairs have a unit distance
                h_adjacent.append(pair)

        # Get the number of consecutive H-pairs in the string,
        # these are not included in computing the energy
        h_consecutive = 0
        for i in range(1, len(chain)):
            if (self.seq[i] == 'H') and (self.seq[i] == self.seq[i-1]):
                h_consecutive += 1

        # Remove duplicate pairs of pairs and subtract the
        # consecutive pairs
        nb_h_adjacent = len(h_adjacent)
        gibbs_energy = nb_h_adjacent
        if (nb_h_adjacent > 0):
            gibbs_energy = nb_h_adjacent - h_consecutive
        #gibbs_energy = nb_h_adjacent 
        reward = - gibbs_energy
        return int(reward)
