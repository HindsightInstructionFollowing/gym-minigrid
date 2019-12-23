from gym_minigrid.minigrid import *
from gym_minigrid.register import register

from gym_minigrid.minigrid_extension import GridAttr
import json
import random
from os import path

import pickle as pkl

from gym_minigrid.envs.fetch import FetchEnv
from os import path

class RelationnalFetch(MiniGridEnv):
    """
    Environment in which the agent has to fetch an object

    The mission is an indirect description of the object
    such as :
    "Fetch the object on the left of the black key"

    Additions made :
    ============================
    - Adding more attributes to the object to have more complex task
    """

    def __init__(
            self,
            size=8,
            numObjs=10,
            missions_file_str="gym-minigrid/gym_minigrid/envs/missions/fetch_3_attr_70.json",
    ):
        self.numObjs = numObjs

        if missions_file_str[-4:] == "json":
            self.missions_list = json.load(open(missions_file_str, 'r'))
            self.create_visual_mission_from_features = lambda x : None
            self.sample_object = self._sample_object_list
        elif missions_file_str[-3:] == "pkl":
            self.missions_list = pkl.load(open(missions_file_str, 'rb'))
            self.sample_object = self._sample_object_dict
        else:
            raise NotImplementedError("Cannot use other type of file, file is {}", missions_file_str)

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

        objs = []
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

        self.objs = objs

        # Randomize the player start position and orientation
        self.place_agent()

        # Choose a random object to be the anchor
        anchor = objs[self._rand_int(0, len(objs))]
        self.create_visual_mission_from_features(anchor)

        self.generate_mission(anchor=anchor)
        assert hasattr(self, 'mission')


    def generate_mission(self, anchor, direction=None):
        correct_direction = False
        while not correct_direction:
            direction = random.choice(['north', 'south', 'west', 'east'])
            n_potential_obj = self.count_relation(self.objs, anchor, direction)
            if n_potential_obj > 0:
                correct_direction = True
                self.n_possible_object =  n_potential_obj
                self.objective_direction = direction
                self.anchor = anchor

        # Generate the mission string
        idx = self._rand_int(0, 5)
        if idx == 0:
            self.mission = 'get an object {} of the {} {} {} {}'
        elif idx == 1:
            self.mission = 'go get an object {} of the {} {} {} {}'
        elif idx == 2:
            self.mission = 'fetch an object {} of the {} {} {} {}'
        elif idx == 3:
            self.mission = 'go fetch an object {} of the {} {} {} {}'
        elif idx == 4:
            self.mission = 'you must fetch an object {} of the {} {} {} {}'

        self.mission = self.mission.format(direction,
                                           anchor.color,
                                           anchor.type,
                                           anchor.shade,
                                           anchor.size)

    def count_relation(self, obj_list, anchor, direction):
        def exist_superior(objs, value, axis):
            count = 0
            for obj in objs:
                if obj.cur_pos[axis] > value:
                    count +=1
            return count
        def exist_inferior(objs, value, axis):
            count = 0
            for obj in objs:
                if obj.cur_pos[axis] < value:
                    count += 1
            return count

        if direction == 'north':
            axis=1
            return exist_inferior(obj_list, anchor.cur_pos[axis], axis)
        if direction == 'south':
            axis=1
            return exist_superior(obj_list, anchor.cur_pos[axis], axis)
        if direction == 'east':
            axis = 0
            return exist_superior(obj_list, anchor.cur_pos[axis], axis)
        if direction == 'west':
            axis = 0
            return exist_inferior(obj_list, anchor.cur_pos[axis], axis)

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.carrying:
            if self.count_relation([self.carrying], self.anchor, self.objective_direction):
                reward = self._reward()
                done = True
            else:
                reward = 0
                done = True

        return obs, reward, done, info

    def _sample_object_list(self):
        return random.choice(self.missions_list)

    def _sample_object_dict(self):

        validated_mission = False
        while not validated_mission:
            mission = random.sample(list(self.missions_list), 1)[0]
            direction, objective_count = mission
            for obj in self.objs:
                mission_count = self.count_relation(obj_list=self.objs,
                                                    anchor=obj,
                                                    direction=direction)
                if mission_count == objective_count:
                    validated_mission = True

        return mission

    def create_visual_mission_from_features(self, target):
        key = (target.color, target.type, target.size, target.shade)

        # Sample a random visual goal (random distractor)
        self.mission_vision = random.choice(self.missions_list[key]["final_state"])
        self.mission_vision_no_distractor = self.missions_list[key]["final_state_wo_distraction"][0]


if __name__ == "__main__":

    env = RelationnalFetch(numObjs=3, size=6)
    while True:
        env.reset()
        env.render()
        print(env.n_possible_object)
        print(env.mission)
        pass

