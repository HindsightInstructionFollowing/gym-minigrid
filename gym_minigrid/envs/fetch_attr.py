from gym_minigrid.minigrid import *
from gym_minigrid.register import register

from gym_minigrid.minigrid_extension import GridAttr
import json
import random
from os import path

class FetchAttrEnv(MiniGridEnv):
    """
    Environment in which the agent has to fetch a random object
    named using English text strings

    Additions made :
    ============================
    - Hindsight available in info
    - Adding more attributes to the object to have more complex task
    - Choose missions among list, send via json (clear train/test separations)
    """

    def __init__(
        self,
        size=8,
        numObjs=10,
        missions_file_str="gym-minigrid/gym_minigrid/envs/missions/fetch_train_missions_10_percent.json"
    ):
        self.numObjs = numObjs

        self.missions_list = json.load(open(missions_file_str, 'r'))

        super().__init__(
            grid_size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            see_through_walls=True
        )

        obs_space = {"image" : gym.spaces.Box(0, 255, (7,7,5))}
        self.observation_space = gym.spaces.Dict(obs_space)

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

        # For each object to be generated
        while len(objs) < self.numObjs:
            rand_mission = random.choice(self.missions_list)
            objColor = rand_mission[0]
            objType = rand_mission[1]
            objSize = rand_mission[2]
            objShade = rand_mission[3]

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
        target = objs[self._rand_int(0, len(objs))]
        self.targetType = target.type
        self.targetColor = target.color
        self.targetShade = target.shade
        self.targetSize = target.size

        descStr = '%s %s %s %s' % (self.targetColor, self.targetType, self.targetShade, self.targetSize)

        # Generate the mission string
        idx = self._rand_int(0, 5)
        if idx == 0:
            self.mission = 'get a %s' % descStr
        elif idx == 1:
            self.mission = 'go get a %s' % descStr
        elif idx == 2:
            self.mission = 'fetch a %s' % descStr
        elif idx == 3:
            self.mission = 'go fetch a %s' % descStr
        elif idx == 4:
            self.mission = 'you must fetch a %s' % descStr
        assert hasattr(self, 'mission')

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

                info["hindsight_target"] = hindsight_target
                reward = 0
                done = True

        return obs, reward, done, info

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
