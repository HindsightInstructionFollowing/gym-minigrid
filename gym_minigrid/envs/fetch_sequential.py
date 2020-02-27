from gym_minigrid.minigrid import *
from gym_minigrid.register import register

from gym_minigrid.minigrid_extension import GridAttr
import json
import random
from os import path

import pickle as pkl

from gym_minigrid.envs.fetch_attr import FetchAttrEnv
from os import path

class FetchAttrSequentialEnv(FetchAttrEnv):
    """
    Environment in which the agent has to fetch a a series of random object
    named using English text strings
    """

    def __init__(
            self,
            size=8,
            numObjs=10,
            max_steps=50,
            missions_file_str="gym-minigrid/gym_minigrid/envs/missions/fetch_train_missions_10_percent.json",
            single_mission=False,
            n_step_between_test=0,
            n_step_test=0,
            seed=42,
            n_objective=2,
            ordered_pickup=False,
    ):
        self.n_objective = n_objective
        self.ordered_pickup = ordered_pickup

        super().__init__(size=size,
                         numObjs=numObjs,
                         max_steps=max_steps,
                         missions_file_str=missions_file_str,
                         single_mission=single_mission,
                         n_step_between_test=n_step_between_test,
                         n_step_test=n_step_test,
                         seed=seed)



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
            for objective_num in range(self.n_objective):
                objColor, objType, objSize, objShade = self.missions_list[objective_num]
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
        targets = []
        if self.single_mission:
            targets = [objs[i] for i in range(self.n_objective)]
        else:
            targets_id = {}
            while len(targets) < self.n_objective:
                target_id = self._rand_int(0, len(objs))
                if target_id not in targets_id:
                    targets.append(objs[target_id])

        self.targets = targets
        self.mission = self.generate_random_goal_string(targets)
        self.object_picked = []
        assert hasattr(self, 'mission')

    def generate_random_goal_string(self, targets):

        for num_target, target in enumerate(targets):
            descStr = '%s %s %s %s' % (target.color, target.type, target.shade, target.size)

            if num_target == 0:
                # Generate the mission string
                idx = self._rand_int(0, 5)
                if idx == 0:
                    mission = 'get a %s ' % descStr
                elif idx == 1:
                    mission = 'go get a %s ' % descStr
                elif idx == 2:
                    mission = 'fetch a %s ' % descStr
                elif idx == 3:
                    mission = 'go fetch a %s ' % descStr
                elif idx == 4:
                    mission = 'you must fetch a %s ' % descStr
            else:
                # If you need to pick object in order use 'then' otherwise 'and' is used
                mission += 'then' if self.ordered_pickup else 'and'
                mission += ' a ' + descStr

        return mission


    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.carrying:

            self.object_picked.append(self.carrying)
            self.carrying = None
            if len(self.object_picked) >= self.n_objective:

                objective_complete = 0
                for obj in self.object_picked:
                    for target_obj in self.targets:
                        if obj.color == target_obj.color \
                                and obj.type == target_obj.type \
                                and obj.shade == target_obj.shade \
                                and obj.size == target_obj.size:
                            objective_complete += 1
                            break

                        if self.ordered_pickup:
                            break

                if objective_complete == self.n_objective:
                    reward = self._reward()
                    done = True

                else:
                    #hindsight from env, in info
                    hindsight_target = []
                    for obj in self.object_picked:
                        hindsight_target.append({
                            "color": obj.color,
                            "type": obj.type,
                            "shade": obj.shade,
                            "size": obj.size
                        })

                    obs["hindsight_target"] = hindsight_target
                    obs["hindsight_mission"] = self.generate_random_goal_string(self.object_picked)
                    reward = 0
                    done = True

        return obs, reward, done, info

if __name__ == "__main__":

    fetch = FetchAttrSequentialEnv()

    done = False
    while not done :
        fetch.render('human')
        act = int(input())
        obs, reward, done, info = fetch.step(act)

    print(reward)
