import math
import operator
from functools import reduce

import numpy as np
import gym
from gym import error, spaces, utils
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, \
    SHADE_TO_IDX, SIZE_TO_IDX
from gym_minigrid.minigrid import CELL_PIXELS

import json
import torch

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

        low, high, shape = int(obs_space["image"].low.min()), int(obs_space["image"].high.max()), obs_space["image"].shape
        new_shape = (n_stack, *shape)
        obs_space["image"] = spaces.Box(low=low, high=high, shape=new_shape)

        self.observation_space = spaces.Dict(obs_space)
        self.last_frames = []
        self.n_stack = n_stack

    def reset(self):
        obs = self.env.reset()
        last_frame = obs["image"]
        obs["last_state"] = last_frame
        self.last_frames = [last_frame]*self.n_stack
        obs["image"] = np.stack(self.last_frames)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        obs["last_state"] = obs["image"]
        self.last_frames.append(obs["image"])
        self.last_frames.pop(0)
        obs["image"] = np.stack(self.last_frames)

        return obs, reward, done, info


class RemoveUselessActionWrapper(gym.core.Wrapper):
    def __init__(self, env, n_action=4):
        super().__init__(env)
        self.action_space = spaces.Discrete(4)
    def step(self, act):
        assert self.action_space.contains(act), "Action not in action space"
        return self.env.step(act)

class RemoveUselessChannelWrapper(gym.core.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        # Modify observation space to contains the correct number of channel
        obs_space = {}
        for key in env.observation_space.spaces.keys():
            obs_space[key] = env.observation_space.spaces[key]

        n_channel = obs_space["image"].shape[-1]
        old_shape = obs_space["image"].shape[:-1]

        # Change the last dimension, whatever the number of dimension
        obs_space["image"] = spaces.Box(0, 255, (*old_shape,n_channel-1))
        self.observation_space = spaces.Dict(obs_space)

    def observation(self, obs):
        # Reduce complexity by removing open/close, not useful in many env
        obs["image"] = obs["image"][:,:,:-1] # Remove last element in state
        return obs

class Word2IndexWrapper(gym.core.ObservationWrapper):

    def __init__(self, env, vocab_file_str=""):
        """
        Load vocabulary from file, path in str format is given as input to TextWrapper
        """

        super().__init__(env)

        # Word => Index
        vocab_dict = json.load(open(vocab_file_str, 'r'))
        self.w2i = vocab_dict["vocabulary"]

        self.max_vocabulary = len(self.w2i)
        self.sentence_max_length = vocab_dict["sentence_max_length"]

        assert self.w2i["<BEG>"] == 0
        assert self.w2i["<END>"] == 1
        assert self.w2i["<PAD>"] == 2


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

        self.raw_mission = obs["mission"]
        self.current_mission =  [self.w2i["<BEG>"]] + [self.w2i[word] for word in obs["mission"].split(" ")] + [self.w2i["<END>"]]
        self.len_mission = len(self.current_mission)

        # No Padding, done in model
        # mission_len = len(self.current_mission)
        # n_padding = self.sentence_max_length - mission_len
        # if n_padding > 0:
        #     self.current_mission.extend([0]*n_padding)

        obs["mission"] = self.current_mission
        obs["raw_mission"] = self.raw_mission
        obs["mission_length"] = self.len_mission

        return obs

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        obs["mission"] = self.current_mission
        obs["raw_mission"] = self.raw_mission
        obs["mission_length"] = self.len_mission

        if "hindsight_mission" in obs:
            obs["hindsight_raw_mission"] = obs["hindsight_mission"]
            obs["hindsight_mission"] = [self.w2i[word] for word in obs["hindsight_mission"].split(" ")]

        return obs, reward, done, info


class CartPoleWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        obs_space = dict()
        obs_space["image"] = gym.spaces.Box(0,1,(4,))
        obs_space["mission"] = gym.spaces.Box(0,1,(2,))
        self.observation_space = spaces.Dict(obs_space)

    def observation(self, observation):
        obs = dict()
        obs["image"] = observation
        obs["mission"] = [1,1]
        obs["mission_length"] = [2]
        return obs


class TorchWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, device='cpu'):
        super().__init__(env)
        self.device = device
        if isinstance(env.observation_space, gym.spaces.Dict) :
            self.observation = self._dict_space_to_torch
        elif isinstance(env.observation_space, gym.spaces.Box):
            self.observation = self._box_space_to_torch
        else:
            raise NotImplementedError("Can only deal with box and dict")

    def _dict_space_to_torch(self, obs):
        for key in obs.keys():
            obs[key] = torch.tensor(obs[key]).float().unsqueeze(0).to(self.device)
        return obs

    def _box_space_to_torch(self, obs):
        return torch.tensor(obs).unsqueeze(0).to(self.device)


class VisionObjectiveWrapper(gym.core.ObservationWrapper):

    def __init__(self, env, with_distractor=True):
        super().__init__(env)
        obs_space = {}
        for key in env.observation_space.spaces.keys():
            obs_space[key] = env.observation_space.spaces[key]

        # The objective mission is being added
        h, w, c = obs_space["image"].shape
        obs_space["image"] = spaces.Box(0, 255, shape=(h,w, 2*c-1))
        self.observation_space = gym.spaces.Dict(obs_space)

        # Do you want to remove distractor object (otherobject present)
        self.with_distractor = with_distractor

    def observation(self, obs):
        if self.with_distractor:
            state_obj = obs["final_state"]
        else:
            state_obj = obs["final_state_wo_distractor"]
        assert obs["image"].shape == state_obj.shape
        state_obj = state_obj[:,:,:-1]

        obs["image"] = np.concatenate((state_obj, obs["image"]), axis=2)
        return obs


class MinigridTorchWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, device='cpu'):
        super().__init__(env)
        self.device = device

        obs_space = {}
        for key in env.observation_space.spaces.keys():
            obs_space[key] = env.observation_space.spaces[key]

        h, w, c = obs_space["image"].shape[-3:]
        im_space_shape = list(obs_space["image"].shape)
        im_space_shape[-3:] = c, h, w
        obs_space["image"] = spaces.Box(0, 255, shape=im_space_shape)

        self.observation_space = gym.spaces.Dict(obs_space)

    def observation(self, obs):
        obs["image"] = torch.Tensor(obs["image"])
        if obs["image"].dim() == 4:
            obs["image"] = obs["image"].permute(0,3,1,2)
        else:
            assert obs["image"].dim() == 3, "Image should be of dim 4 (stacked) or 3 (single image)"
            obs["image"] = obs["image"].permute(2, 0, 1)

        obs["image"] = obs["image"].unsqueeze(0).to(self.device)
        obs["mission"] = torch.LongTensor(obs["mission"]).unsqueeze(0).to(self.device)
        obs["mission_length"] = torch.LongTensor([obs["mission_length"]]).to(self.device)
        return obs

def wrap_env_from_list(env, wrappers_json):

    str2wrap = {
        "directionwrapper" : DirectionWrapper,
        "visionobjectivewrapper" : VisionObjectiveWrapper,
        "framestackerwrapper" : FrameStackerWrapper,
        "word2indexwrapper": Word2IndexWrapper,
        "removeuselessactionwrapper" : RemoveUselessActionWrapper,
        "removeuselesschannelwrapper" : RemoveUselessChannelWrapper,
        "minigridtorchwrapper": MinigridTorchWrapper,
        "vizdoom2minigrid" : Vizdoom2Minigrid
    }

    for wrap_dict in wrappers_json:
        current_wrap = str2wrap[wrap_dict["name"].lower()]
        env = current_wrap(env, **wrap_dict["params"])

    return env

class DirectionWrapper(gym.core.ObservationWrapper):
    """
    Add direction as 2 features map in the image space
    if image is (7,7,3), new state will be (7,7,5)
    """
    def __init__(self, env):
        super().__init__(env)

        obs_space = {}
        for key in env.observation_space.spaces.keys():
            obs_space[key] = env.observation_space.spaces[key]

        new_shape = list(obs_space["image"].shape)
        #add 2 channels
        new_shape[2] += 2
        obs_space["image"] = spaces.Box(0, 255, shape=new_shape)

    def observation(self, obs):
        assert 'direction' in obs, "Direction not present in observation"

        h,w = obs["image"].shape[:2]
        direction = obs["direction"]

        direction_channels = np.zeros((h, w, 2))

        if direction == 0: # East
            direction_channels[:,:,0] = 1
        elif direction == 1: # South
            direction_channels[:,:,1] = 1
        elif direction == 2: # West
            direction_channels[:,:,0] = -1
        elif direction == 3: # North
            direction_channels[:,:,1] = -1

        obs["image"] = np.concatenate((direction, obs["image"]), axis=2)
        return obs


class FullyObsWrapperAndState(gym.core.ObservationWrapper):
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
            SHADE_TO_IDX['very_light'],
            SIZE_TO_IDX['tiny'],
            env.agent_dir
        ])

        return {
            'mission': obs['mission'],
            'image_full': full_grid,
            'image': env.gen_obs()['image']
        }


class Vizdoom2Minigrid(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def reset(self):
        raise NotImplementedError("Need to add <BEG>, <END> into vizdoom")
        (image, instruction), reward, is_done, info = self.env.reset()
        wordidx = [self.env.word_to_idx[word] for word in instruction.split()]
        self.mission = wordidx
        self.mission_raw = instruction
        self.mission_length = len(wordidx)
        return {
            'mission_raw': self.mission_raw,
            'mission': self.mission,
            'image': image,
            "mission_length": self.mission_length
        }

    def step(self, action):
        (image, instruction), reward, done, info = self.env.step(action)
        return {
            'mission_raw': self.mission_raw,
            'mission': self.mission,
            'image': image,
            "mission_length": self.mission_length
        }, reward, done, info


if __name__ == "__main__":

    from gym_minigrid.envs.fetch_attr import FetchAttrEnv
    missons_file_str = "envs/missions/fetch_train_missions_10_percent.json"

    env = FetchAttrEnv(size=8,
                       numObjs=10,
                       missions_file_str=missons_file_str)

    env = LessActionAndObsWrapper(Word2IndexWrapper(env=env, vocab_file_str="envs/missions/vocab_fetch.json"))
    print(env.observation_space)

