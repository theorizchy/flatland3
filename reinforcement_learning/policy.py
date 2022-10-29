from flatland.envs.rail_env import RailEnv


class DummyMemory:
    def __init__(self):
        self.memory = []

    def __len__(self):
        return 0


class Policy:
    def step(self, handle, state, action, reward, next_state, done):
        raise NotImplementedError

    def act(self, handle, state, eps=0.):
        raise NotImplementedError

    def shape_reward(self, handle, action, state, reward, done, deadlocked=None):
        return reward

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError

    def start_step(self, train):
        pass

    def get_agent_handles(self, env):
        return range(env.get_num_agents())

    def end_step(self, train):
        pass

    def start_episode(self, train):
        pass

    def end_episode(self, train):
        pass

    def load_replay_buffer(self, filename):
        pass

    def test(self):
        pass

    def reset(self, env: RailEnv):
        pass

    def clone(self):
        return self


class HeuristicPolicy(Policy):
    def __init__(self):
        super(HeuristicPolicy).__init__()


class LearningPolicy(Policy):
    def __init__(self):
        super(LearningPolicy).__init__()


class HybridPolicy(Policy):
    def __init__(self):
        super(HybridPolicy).__init__()
