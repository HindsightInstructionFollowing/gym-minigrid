import gym
import gym_minigrid
import pickle
import numpy as np

from gym_minigrid.envs.fetch_attr import FetchAttrEnv
from gym_minigrid.envs.relationnal import RelationnalFetch
from gym_minigrid.minigrid import check_integrity

from gym_minigrid.wrappers import FullyObsWrapperAndState

from gym_minigrid.minigrid import COLOR_TO_IDX, SHADE_TO_IDX, OBJECT_TO_IDX

def remove_distractor(state):
    state_copy = np.copy(state)
    for i in range(state_copy.shape[0]):
        for j in range(state_copy.shape[1]):
            if (i,j) != (3,5):
                state_copy[i, j] = (1, 0, 0, 0, 0)
    return state_copy

env_name = "relationnal_fetch_3A_N3_70"
#env = gym.make(env_name)
# env = FetchAttrEnv(numObjs=2,
#                    size=6,
#                    missions_file_str="gym-minigrid/gym_minigrid/envs/missions/fetch_3_attr_70.json")

env = RelationnalFetch(numObjs=3,
                       size=6,
                       missions_file_str="gym-minigrid/gym_minigrid/envs/missions/fetch_3_attr_70.json")

save_dict = dict()
n_episodes = 10000

env = FullyObsWrapperAndState(env)

for i in range(n_episodes):
    if i % 1000 == 0:
        print("#episodes : {}".format(i))
    obs = env.reset()
    grid = env.grid
    mission = env.mission
    done = False

    while not done:
        action = env.action_space.sample() #int(input())
        #env.render()
        next_obs, reward, done, info = env.step(action)

        if done:
            break
        else:
            obs = next_obs

    # END OF EP, goes here
    # Store objective
    if reward > 0:
        mission_id = (env.objective_direction, env.n_possible_object)

        if mission_id in save_dict:
            save_dict[mission_id]["final_state"].append(np.copy(obs["image"]))
            save_dict[mission_id]["final_state_full"].append(np.copy(obs["image_full"]))
            save_dict[mission_id]["final_raw_state"].append(env.render('rgb_array'))

        else:
            save_dict[mission_id] = dict()
            save_dict[mission_id]["str"] = mission
            save_dict[mission_id]["final_state"] = [np.copy(obs["image"])]
            save_dict[mission_id]["final_state_full"] = [np.copy(obs["image_full"])]

            save_dict[mission_id]["final_raw_state"] = [env.render('rgb_array')]
            # save_dict[mission_id]["final_state_wo_distraction"] = [np.copy(remove_distractor(obs["image"]))]

        # assert np.equal(next_obs["image"][3,6], obs["image"][3,5]).all()
        # check_integrity(save_dict[mission_id]["final_state"][-1], mission_id)
        # check_integrity(save_dict[mission_id]["final_state_wo_distraction"][-1], mission_id)


pickle.dump(save_dict, open("gym-minigrid/gym_minigrid/envs/missions/{}.pkl".format(env_name), 'wb'))