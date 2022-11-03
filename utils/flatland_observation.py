from typing import Optional, List

import numpy as np
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from utils.fast_tree_obs import FastTreeObs
from utils.observation_utils import normalize_observation


class FlatlandTreeObservation(ObservationBuilder):
    def __init__(self, max_depth: int):
        super(FlatlandTreeObservation, self).__init__()
        self.max_depth = max_depth
        self.predictor = ShortestPathPredictorForRailEnv(self.max_depth)
        self.tree_obs = TreeObsForRailEnv(max_depth=max_depth, predictor=self.predictor)
        n_features_per_node = self.tree_obs.observation_dim
        n_nodes = sum([np.power(4, i) for i in range(max_depth + 1)])
        self.tree_obs_observation_dim = n_features_per_node * n_nodes
        self.observation_dim = self.tree_obs_observation_dim

    def reset(self):
        self.tree_obs.reset()

    def set_env(self, env: Environment):
        super().set_env(env)
        self.tree_obs.set_env(env)

    def get_many(self, handles: Optional[List[int]] = None):
        tree_observation = self.tree_obs.get_many(handles)

        final_observations = {}
        if handles is None:
            handles = []
        for h in handles:
            final_observations[h] = self.get_normalized_tree_observation(tree_observation[h],
                                                                         self.max_depth,
                                                                         10)
        return final_observations

    def get_normalized_tree_observation(self, observation, tree_depth: int, observation_radius=0):
        if observation is None:
            normalized_tree_obs = np.zeros(self.tree_obs_observation_dim) - 1
        else:
            normalized_tree_obs = normalize_observation(observation, tree_depth,
                                                        observation_radius)
            if len(normalized_tree_obs) != self.tree_obs_observation_dim:
                normalized_tree_obs = np.zeros(self.tree_obs_observation_dim) - 1
        return normalized_tree_obs


class FlatlandObservation(FlatlandTreeObservation):
    def __init__(self, max_depth: int):
        super(FlatlandObservation, self).__init__(max_depth)
        self.fast_tree_obs = FastTreeObs()
        self.fast_tree_obs_observation_dim = self.fast_tree_obs.observation_dim
        self.observation_dim += self.fast_tree_obs_observation_dim

    def reset(self):
        super(FlatlandObservation, self).reset()
        self.fast_tree_obs.reset()

    def set_env(self, env: Environment):
        super(FlatlandObservation, self).set_env(env)
        self.fast_tree_obs.set_env(env)

    def get_many(self, handles: Optional[List[int]] = None):
        observation = self.fast_tree_obs.get_many(handles)
        tree_observation = self.tree_obs.get_many(handles)

        final_observations = {}
        if handles is None:
            handles = []
        for h in handles:
            final_observations[h] = np.concatenate((observation[h],
                                                    self.get_normalized_tree_observation(tree_observation[h],
                                                                                         self.max_depth,
                                                                                         10)))
        return final_observations


class FlatlandFastTreeObservation(ObservationBuilder):
    def __init__(self):
        super(FlatlandFastTreeObservation, self).__init__()
        self.fast_tree_obs = FastTreeObs()
        self.fast_tree_obs_observation_dim = self.fast_tree_obs.observation_dim
        self.observation_dim = self.fast_tree_obs_observation_dim

    def reset(self):
        self.fast_tree_obs.reset()

    def set_env(self, env: Environment):
        super(FlatlandFastTreeObservation, self).set_env(env)
        self.fast_tree_obs.set_env(env)

    def get_many(self, handles: Optional[List[int]] = None):
        return self.fast_tree_obs.get_many(handles)
