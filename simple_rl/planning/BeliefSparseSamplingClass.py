from math import log
import numpy as np
import copy
from collections import defaultdict
import random
import pdb

from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.planning.ValueIterationClass import ValueIteration
from simple_rl.pomdp.BeliefMDPClass import BeliefMDP, BeliefState


class BeliefSparseSampling(object):
    '''
    A Sparse Sampling Algorithm for Near-Optimal Planning in Large Markov Decision Processes (Kearns et al)

    Assuming that you don't have access to the underlying transition dynamics, but do have access to a naiive generative
    model of the underlying MDP, this algorithm performs on-line, near-optimal planning with a per-state running time
    that has no dependence on the number of states in the MDP.
    '''
    def __init__(self, gen_model, gamma, tol, max_reward, state, name="bss"):
        '''
        Args:
             gen_model (BeliefMDP): Model of our MDP -- we tell it what action we are performing from some state s
             and it will return what our next state is
             gamma (float): MDP discount factor
             tol (float): Most expected difference between optimal and computed value function
             max_reward (float): Upper bound on the reward you can get for any state, action
             state (State): This is the current state, and we need to output the action to take here
        '''
        self.tol = tol
        self.gamma = gamma
        self.max_reward = max_reward
        self.gen_model = gen_model
        self.current_state = state
        self.horizon = self._horizon
        self.width = self._width

        print 'Horizon = {} \t Width = {}'.format(self.horizon, self.width)

        self.name = name
        self.root_level_qvals = defaultdict()
        self.nodes_by_horizon  = defaultdict(lambda: defaultdict(float))

    @property
    def _horizon(self):
        '''
        Returns:
            _horizon (int): The planning horizon; depth of the recursive tree created to determined the near-optimal
            action to take from a given state
        '''
        return int(log((self._lam / self._vmax), self.gamma))

    @property
    def _width(self):
        '''

        Returns:
             _width (int): The number of times we ask the generative model to give us a next_state sample for each
             state, action pair
        '''
        part1 = (self._vmax ** 2) / (self._lam ** 2)
        part2 = 2 * self._horizon * log(self._horizon * (self._vmax ** 2) / (self._lam ** 2))
        part3 = log(self.max_reward / self._lam)
        return int(part1 * (part2 + part3))

    @property
    def _lam(self):
        return (self.tol * (1.0 - self.gamma) ** 2) / 4.0

    @property
    def _vmax(self):
        return float(self.max_reward) / (1 - self.gamma)

    def _get_width_at_height(self, height):
        '''

        Args:
            height (int): the current depth in the MDP recursive tree measured from top
        Returns:
            width (int): the decayed branching factor for a state, action pair

        The branching factor of the tree is decayed according to this formula as suggested by the BSS paper
        '''
        c = int(self.width * (self.gamma ** (2 * height)))
        return c if c > 1 else 1

    def _estimate_qs(self, state, horizon):
        qvalues = np.zeros(len(self.gen_model.actions))
        for action_idx, action in enumerate(self.gen_model.actions):
            if horizon <= 0:
                qvalues[action_idx] = 0.0
            else:
                qvalues[action_idx] = self._sampled_q_estimate(state, action, horizon)
        return qvalues

    def _sampled_q_estimate(self, state, action, horizon):
        '''
        Args:
            state (State): current state in MDP
            action (str): action to take from `state`
            horizon (int): planning horizon / depth of recursive tree

        Returns:
            average_reward (float): measure of how good (s, a) would be
        '''
        sum_ = 0.0
        width = self._get_width_at_height(self.horizon - horizon)
        for _ in range(width):
            next_state = self.gen_model.transition_func(state, action)
            sum_ += self.gen_model.reward_func(state, action) + (self.gamma * self._estimate_v(next_state, horizon-1))
        return sum_ / float(width)

    def _estimate_v(self, state, horizon):
        '''
        Args:
            state (State): current state
            horizon (int): time steps in future you want to use to estimate V*

        Returns:
            V(s) (float)
        '''
        if state in self.nodes_by_horizon:
            if horizon in self.nodes_by_horizon[state]:
                return self.nodes_by_horizon[state][horizon]

        if self.gen_model.is_in_goal_state():
            self.nodes_by_horizon[state][horizon] = self.gen_model.reward_func(state, random.choice(self.gen_model.actions))
        else:
            self.nodes_by_horizon[state][horizon] = np.max(self._estimate_qs(state, horizon))

        return self.nodes_by_horizon[state][horizon]

    def plan_from_state(self, state):
        '''
        Args:
            state (State): the current state in the MDP

        Returns:
            action (str): near-optimal action to perform from state
        '''
        if state in self.root_level_qvals:
            qvalues = self.root_level_qvals[state]
        else:
            init_horizon = self.horizon
            qvalues = self._estimate_qs(state, init_horizon)
        action_idx = np.argmax(qvalues)
        self.root_level_qvals[state] = qvalues
        return self.gen_model.actions[action_idx]

    def run(self, num_episodes=5, verbose=True):
        final_scores, policies = [], []
        for episode in range(num_episodes):
            discounted_sum_rewards = 0.0
            num_iter = 0
            self.gen_model.reset()
            state = self.gen_model.init_state
            policy = defaultdict()
            if verbose: print 'Episode {}: '.format(episode)
            while not self.gen_model.is_in_goal_state():
                action = self.plan_from_state(state)
                reward, next_state = self.gen_model.execute_agent_action(action)
                policy[state] = action
                discounted_sum_rewards += ((self.gamma ** num_iter) * reward)
                if verbose: print '({}, {}, {}) -> {} | {}'.format(state, action, next_state, reward, discounted_sum_rewards)
                state = copy.deepcopy(next_state)
                num_iter += 1
            final_scores.append(discounted_sum_rewards)
            policies.append(policy)
        return final_scores, policies

def plan_with_vi(gamma=0.99):
    '''
    Args:
        gamma (float): discount factor

    Running value iteration on the problem to test the correctness of the policy returned by BSS
    '''
    mdp = GridWorldMDP(gamma=gamma, goal_locs=[(4, 3)], slip_prob=0.0)
    value_iter = ValueIteration(mdp, sample_rate=5)
    value_iter.run_vi()

    action_seq, state_seq = value_iter.plan(mdp.get_init_state())

    print "[ValueIteration] Plan for {}".format(mdp)
    for i in range(len(action_seq)):
        print 'pi({}) --> {}'.format(state_seq[i], action_seq[i])


if __name__ == '__main__':
    from simple_rl.tasks.maze_1d.Maze1DPOMDPClass import Maze1DPOMDP
    discount_factor = 0.6
    plan_with_vi(gamma=discount_factor)
    pomdp = Maze1DPOMDP()
    model = BeliefMDP(pomdp)
    bss = BeliefSparseSampling(gen_model=model, gamma=discount_factor, tol=1.0, max_reward=1.0, state=model.init_state)
    scores, policies = bss.run(num_episodes=3, verbose=True)
