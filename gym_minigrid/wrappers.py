import math
import operator
from functools import reduce

import numpy as np
import gym
from gym import error, spaces, utils
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
from gym_minigrid.minigrid import CELL_PIXELS

import json

class ReseedWrapper(gym.core.Wrapper):
    """{{
    Wrapper to always regenerate an environment with the same set of seeds.
    This can be used to force an environment to always keep the same
    configuration when reset.
    """

    def __init__(self, env, seeds=[0], seed_idx=0):
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        super().__init__(env)

    def reset(self, **kwargs):
        seed = self.seeds[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        self.env.seed(seed)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

class ActionBonus(gym.core.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class StateBonus(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = (tuple(env.agent_pos))

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']

class OneHotPartialObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space['image'].shape

        # Number of bits per cell
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1], num_bits),
            dtype='uint8'
        )

    def observation(self, obs):
        img = obs['image']
        out = np.zeros(self.observation_space.shape, dtype='uint8')

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        return {
            'mission': obs['mission'],
            'image': out
        }

class RGBImgObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width*tile_size, self.env.height*tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )

        return {
            'mission': obs['mission'],
            'image': rgb_img
        }


class RGBImgPartialObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as the only observation output
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space['image'].shape
        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img_partial = env.get_obs_render(
            obs['image'],
            tile_size=self.tile_size,
            mode='rgb_array'
        )

        return {
            'mission': obs['mission'],
            'image': rgb_img_partial
        }

class FullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        return {
            'mission': obs['mission'],
            'image': full_grid
        }

class FlatObsWrapper(gym.core.ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 27

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, imgSize + self.numCharCodes * self.maxStrLen),
            dtype='uint8'
        )

        self.cachedStr = None
        self.cachedArray = None

    def observation(self, obs):
        image = obs['image']
        mission = obs['mission']

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert len(mission) <= self.maxStrLen, 'mission string too long ({} chars)'.format(len(mission))
            mission = mission.lower()

            strArray = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype='float32')

            for idx, ch in enumerate(mission):
                if ch >= 'a' and ch <= 'z':
                    chNo = ord(ch) - ord('a')
                elif ch == ' ':
                    chNo = ord('z') - ord('a') + 1
                assert chNo < self.numCharCodes, '%s : %d' % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs

class ViewSizeWrapper(gym.core.Wrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        # Override default view size
        env.unwrapped.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(agent_view_size, agent_view_size, 3),
            dtype='uint8'
        )

        # Override the environment's observation space
        self.observation_space = spaces.Dict({
            'image': observation_space
        })

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


class FrameStackerWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, n_stack=4):
        """
        Stack last frames to give the agent a bit a memory

        old observation shape is : (width, height, channel)
        new observation is : (N_STACK, width, height, channel)
        """
        super().__init__(env=env)

        # Modify observation space to contains stacking, pre-pending a dimension
        obs_space = {}
        for key in env.observation_space.spaces.keys():
            obs_space[key] = env.observation_space.spaces[key]

        low, high, shape = obs_space["image"].low, obs_space["image"].high, obs_space["image"].shape
        new_shape = (n_stack, *shape)
        obs_space["image"] = spaces.Box(low=low, high=high, shape=new_shape)

        self.observation_space = spaces.Dict(obs_space)
        self.last_frames = []
        self.n_stack = n_stack

    def reset(self):
        obs = self.env.reset()
        last_frame = obs["image"]
        self.last_frames = [last_frame]*self.n_stack
        obs["image"] = np.stack(self.last_frames)

    def step(self, action):
        obs = self.env.step(action)
        self.last_frames.append(obs["image"])
        self.last_frames.pop(0)
        return np.stack(self.last_frames)

class LessActionAndObsWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(4)

        # Modify observation space to contains the correct number of channel
        obs_space = {}
        for key in env.observation_space.spaces.keys():
            obs_space[key] = env.observation_space.spaces[key]

        n_channel = obs_space["image"].shape[-1]
        old_shape = obs_space["image"].shape[:-1]

        # Change the last dimension, whatever the number of dimension
        obs_space["image"] = spaces.Box(0, 255, (*old_shape,n_channel-1))
        self.observation_space = spaces.Dict(obs_space)

    def step(self, action):
        assert self.action_space.contains(action), "Action not available in LessActionAndObsWrapper. Action : {}".format(action)
        obs, reward, done, info = self.env.step(action)

        # Reduce complexity by removing open/close, not useful in many env
        obs["image"] = obs["image"][:,:,:-1] # Remove last element in state
        return obs, reward, done, info

class TextWrapper(gym.core.ObservationWrapper):

    def __init__(self, env, vocab_file_str="gym-minigrid/gym_minigrid/envs/missions/vocab_fetch.json"):
        """
        Load vocabulary from file, path in str format is given as input to TextWrapper
        """

        super().__init__(env)

        # Word => Index
        vocab_dict = json.load(open(vocab_file_str, 'r'))
        self.w2i = vocab_dict["vocabulary"]

        self.max_vocabulary = len(self.w2i)
        self.sentence_max_length = vocab_dict["sentence_max_length"]

        # Index => Word
        self.i2w = list(range(len(self.w2i)))
        for word, index in self.w2i.items():
            self.i2w[index] = word

        obs_space = {}
        for key in env.observation_space.spaces.keys():
            obs_space[key] = env.observation_space.spaces[key]

        obs_space["mission"] = spaces.Box(0, self.max_vocabulary, (self.sentence_max_length,))
        self.observation_space = spaces.Dict(obs_space)

    def reset(self):
        obs = self.env.reset()
        self.current_mission = [self.w2i(word) for word in obs["mission"]]

        # Pad
        mission_len = len(self.current_mission)
        n_padding = self.sentence_max_length - mission_len
        if n_padding > 0:
            self.current_mission.extend([0]*n_padding)

        self.raw_mission = obs["mission"]

        obs["mission"] = self.current_mission
        obs["raw_mission"] = self.raw_mission
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["mission"] = self.current_mission
        obs["raw_mission"] = self.raw_mission

        return obs, reward, done, info


if __name__ == "__main__":

    from gym_minigrid.envs.fetch_attr import FetchAttrEnv
    missons_file_str = "envs/missions/fetch_train_missions_10_percent.json"

    env = FetchAttrEnv(size=8,
                       numObjs=10,
                       missions_file_str=missons_file_str)

    env = LessActionAndObsWrapper(TextWrapper(env=env, vocab_file_str="envs/missions/vocab_fetch.json"))
    print(env.observation_space)

