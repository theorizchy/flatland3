import sys
from pathlib import Path

import numpy as np

from reinforcement_learning.policy import Policy

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from utils.observation_utils import split_tree_into_feature_groups, min_gt


class OrderedPolicy(Policy):
    def __init__(self):
        self.action_size = 5

    def act(self, state, eps=0.):
        _, distance, _ = split_tree_into_feature_groups(state, 1)
        distance = distance[1:]
        min_dist = min_gt(distance, 0)
        min_direction = np.where(distance == min_dist)
        if len(min_direction[0]) > 1:
            return min_direction[0][-1] + 1
        return min_direction[0] + 1

    def step(self, state, action, reward, next_state, done):
        return

    def save(self, filename):
        return

    def load(self, filename):
        return
