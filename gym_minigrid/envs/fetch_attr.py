from gym_minigrid.minigrid import *
from gym_minigrid.register import register

from gym_minigrid.minigrid_extension import GridAttr
import json
import random
from os import path

import pickle as pkl

from gym_minigrid.envs.fetch import FetchEnv
from os import path

class FetchAttrEnv(MiniGridEnv):
    """
    Environment in which the agent has to fetch a random object
    named using English text strings

    Additions made :
    ============================
    - Adding more attributes to the object to have more complex task
    - Choose missions among list, send via json (clear train/test separations)
    """

    def __init__(
            self,
            size=8,
            numObjs=10,
            max_steps=40,
            missions_file_str="gym-minigrid/gym_minigrid/envs/missions/fetch_train_missions_10_percent.json",
            single_mission=False,
            n_step_between_test=0,
            n_step_test=0,
            seed=42
    ):
        self.numObjs = numObjs
        self.missions_list = json.load(open(missions_file_str, 'r'))
        self.single_mission = single_mission

        # If the env is used as test, those variable will help
        self.n_step_between_test=n_step_between_test
        self.n_step_test=n_step_test

        super().__init__(
            grid_size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            see_through_walls=True,
            seed=seed
        )

        obs_space = {"image" : gym.spaces.Box(0, 255, (7,7,5))}
        self.observation_space = gym.spaces.Dict(obs_space)

        self.max_steps = min(self.max_steps, 50)

    def _gen_grid(self, width, height):
        # Adding new attributes here !
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.grid = GridAttr(width=width,
                             height=height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width-1, 0)

        types = ['key', 'ball']

        objs = []

        # First object, always the same
        if self.single_mission:
            objColor, objType, objSize, objShade = self.get_first_mission()
            if objType == 'key':
                obj = Key(objColor, objShade, objSize)
            elif objType == 'ball':
                obj = Ball(objColor, objShade, objSize)

            self.place_obj(obj)
            objs.append(obj)

        # For each object to be generated
        while len(objs) < self.numObjs:
            rand_object = self.sample_object()
            objColor = rand_object[0]
            objType = rand_object[1]
            objSize = rand_object[2]
            objShade = rand_object[3]

            if objType == 'key':
                obj = Key(objColor, objShade, objSize)
            elif objType == 'ball':
                obj = Ball(objColor, objShade, objSize)
            else:
                raise NotImplementedError("Wrong object type. type = {}".format(objType))

            self.place_obj(obj)
            objs.append(obj)

        # Randomize the player start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        if self.single_mission:
            target = objs[0]
        else:
            target = objs[self._rand_int(0, len(objs))]
            self.create_visual_mission_from_features(target)

        self.targetType = target.type
        self.targetColor = target.color
        self.targetShade = target.shade
        self.targetSize = target.size

        self.mission = self.generate_random_goal_string(self.targetColor, self.targetType, self.targetShade, self.targetSize)
        assert hasattr(self, 'mission')

    def generate_random_goal_string(self, target_color, target_type, target_shade, target_size):
        descStr = '%s %s %s %s' % (target_color, target_type, target_shade, target_size)

        # Generate the mission string
        idx = self._rand_int(0, 5)
        if idx == 0:
            mission = 'get a %s' % descStr
        elif idx == 1:
            mission = 'go get a %s' % descStr
        elif idx == 2:
            mission = 'fetch a %s' % descStr
        elif idx == 3:
            mission = 'go fetch a %s' % descStr
        elif idx == 4:
            mission = 'you must fetch a %s' % descStr
        return mission


    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.carrying:
            if self.carrying.color == self.targetColor \
                and self.carrying.type == self.targetType \
                and self.carrying.shade == self.targetShade \
                and self.carrying.size == self.targetSize:

                reward = self._reward()
                done = True
            else:
                #hindsight from env, in info
                hindsight_target = {
                    "color": self.carrying.color,
                    "type": self.carrying.type,
                    "shade": self.carrying.shade,
                    "size": self.carrying.size
                }

                obs["hindsight_target"] = hindsight_target
                obs["hindsight_mission"] = self.generate_random_goal_string(target_color=self.carrying.color,
                                                                             target_type=self.carrying.type,
                                                                             target_shade=self.carrying.shade,
                                                                             target_size=self.carrying.size)
                reward = 0
                done = True

        return obs, reward, done, info

    def sample_object(self):
        return random.choice(self.missions_list)
    def get_first_mission(self):
        return self.missions_list[0]
    def create_visual_mission_from_features(self, target):
        return None

class FetchAttrDictLoaded(FetchAttrEnv):
    def __init__(self, size=8, numObjs=3, dict_mission_str="MiniGrid-Fetch-6x6-N2-v0_all_missions.pkl"):

        gym_mission_path = "gym-minigrid/gym_minigrid/envs/missions/"
        full_dict_path = path.join(gym_mission_path, dict_mission_str)
        self.mission_dict = pkl.load(open(full_dict_path, 'rb'))
        super().__init__(size=size, numObjs=numObjs)

    def gen_obs(self):
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image': image,
            'direction': self.agent_dir,
            'mission': self.mission,
            'final_state' : self.mission_vision,
            'final_state_wo_distractor': self.mission_vision_no_distractor
        }

        return obs

    def sample_object(self):
        mission = random.sample(list(self.mission_dict), 1)[0]
        return mission

    def get_first_mission(self):
        for key in self.mission_dict.keys():
            mission = key
        return mission

    def create_visual_mission_from_features(self, target):
        key = (target.color, target.type, target.size, target.shade)

        # Sample a random visual goal (random distractor)
        self.mission_vision = random.choice(self.mission_dict[key]["final_state"])
        self.mission_vision_no_distractor = self.mission_dict[key]["final_state_wo_distraction"][0]

# class FetchEnv5x5N2(FetchEnv):
#     def __init__(self):
#         super().__init__(size=5, numObjs=2)
#
# class FetchEnv6x6N2(FetchEnv):
#     def __init__(self):
#         super().__init__(size=6, numObjs=2)
#
# register(
#     id='MiniGrid-Fetch-5x5-N2-v0',
#     entry_point='gym_minigrid.envs:FetchEnv5x5N2'
# )
#
# register(
#     id='MiniGrid-Fetch-6x6-N2-v0',
#     entry_point='gym_minigrid.envs:FetchEnv6x6N2'
# )
#
# register(
#     id='MiniGrid-Fetch-8x8-N3-v0',
#     entry_point='gym_minigrid.envs:FetchEnv'
# )
