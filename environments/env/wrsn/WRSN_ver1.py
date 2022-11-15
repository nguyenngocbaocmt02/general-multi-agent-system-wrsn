import copy
import sys
from abc import ABC
from collections import namedtuple
import numpy as np
from environments.agent.MobileCharger import MobileCharger
from environments.env.MultiAgentEnv import MultiAgentEnv
from environments.iostream.NetworkIO import NetworkIO


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def scale_down(location, frame):
    return [(location[0] - frame[0]) / (frame[1] - frame[0]),
            (location[1] - frame[2]) / (frame[3] - frame[2])]


def scale_up(ma, frame):
    return [ma[0] * (frame[1] - frame[0]) + frame[0],
            ma[1] * (frame[3] - frame[2]) + frame[2]]


class ActionSpace(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high


class WRSN(MultiAgentEnv, ABC):
    def __init__(self, kwargs):
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        self.netIO = NetworkIO(self.args.scenario, args)
        self.n_mcs = self.args.n_mcs
        self.mc_specification = args.mc_specification

        self.degree = min(self.args.degree, 20)
        self.grid_size = [10, 10]
        self.cons_series = [1]
        self.series = []
        self.frame = None
        para = 1
        self.series.append(1)
        self.env, self.net = self.netIO.makeNetwork()
        for i in range(1, self.degree):
            para = para * -1 / i / (2 * i + 1) * (2 * i - 1)
            self.cons_series.append(para)

        x_min, x_max, y_min, y_max = sys.float_info.max, sys.float_info.min, sys.float_info.max, sys.float_info.min
        for node in self.net.listNodes:
            if node.level != -1:
                x_min = min(x_min, node.location[0])
                y_min = min(y_min, node.location[1])
                x_max = max(x_max, node.location[0])
                y_max = max(y_max, node.location[1])
        self.frame = [x_min, x_max, y_min, y_max]

        np.random.seed(args.seed)
        self.episode_limit = args.episode_limit
        self.action_space = ActionSpace(low=-1, high=1)
        if self.args.mode == "distributed":
            self.n_actions = 1
            self.n_agents = self.n_mcs
        elif self.args.mode == "decentralized":
            self.n_actions = self.n_mcs * 3
            self.n_agents = 1

        agents_obs, state = self.reset()
        self.obs_size = agents_obs[0].shape[0]
        self.state_size = state.shape[0]

    def series_cul(self, x0, a, b):
        para1 = (a - x0) * (a - x0)
        para2 = (b - x0) * (b - x0)
        res = 0
        tmp1 = a - x0
        tmp2 = b - x0
        for i in range(self.degree):
            res += (tmp1 - tmp2) * self.cons_series[i]
            tmp1 *= para1
            tmp2 *= para2
        return res

    def reset(self):
        self.env, self.net = self.netIO.makeNetwork()
        self.mcs = [MobileCharger( [self.net.baseStation.location[0], self.net.baseStation.location[1]],
                    convert(self.mc_specification))
                    for i in range(self.n_mcs)]
        tmp = 0
        for mc in self.mcs:
            mc.id = tmp
            tmp += 1
            mc.net = self.net
            mc.env = self.env
        p = self.env.process(self.net.operate(t=1, max_time=100))
        self.env.run(until=p)
        return self.get_obs(), self.get_state()

    def get_state(self):
        state = [[0 for j in range(self.grid_size[0])] for i in range(self.grid_size[1])]
        for node in self.net.listNodes:
            pos = scale_down(node.location, self.frame)
            if node.status == 0:
                break
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    value1 = self.series_cul(pos[0], 1.0 / self.grid_size[0] * i, 1.0 / self.grid_size[0] * (i + 1))
                    value2 = self.series_cul(pos[1], 1.0 / self.grid_size[1] * j, 1.0 / self.grid_size[1] * (j + 1))
                    alpha = node.energyCS / (node.energy / node.capacity)
                    state[i][j] += value1 * value2 * alpha
        print(state)
        return np.array(state).flatten()

    def get_avail_actions(self):
        """return available actions for all agents
        """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return np.expand_dims(np.array(avail_actions), axis=0)

    def get_avail_agent_actions(self, agent_id):
        """ return the available actions for agent_id
        """
        return [1]

    def get_obs(self):
        x = self.get_state()
        return [copy.deepcopy(x) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        return copy.deepcopy(self.get_state())

    def get_state_size(self):
        return self.state_size

    def get_num_of_agents(self):
        return self.n_agents

    def get_obs_size(self):
        return self.obs_size

    def get_total_actions(self):
        return self.n_actions

    def step(self, actions):
        p = self.env.process(self.net.operate(t=1, max_time=50 + self.env.now))
        self._take_action(actions)
        self.env.run(until=p)
        terminal = False
        if self.net.check_targets() == 0:
            terminal = True
        return 1, terminal, self.net

    def _take_action(self, actions):
        tmp = actions.squeeze()
        tmp = tmp.tolist()
        for i, mc in enumerate(self.mcs):
            if tmp[3 * i] < 1.0 / 3.0:
                self.env.process(mc.charge(50))
            elif tmp[3 * i] < 2.0 / 3.0:
                self.env.process(mc.move([tmp[3 * i + 1], tmp[3 * i + 2]], 50, False))
            else:
                self.env.process(mc.move(vector=[self.net.baseStation.location[j] - mc.location[j]
                                                 for j in range(len(mc.location))],
                                         moving_time=50, to_base=True))
