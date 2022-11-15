import torch as th
import os
import argparse
import yaml
from tensorboardX import SummaryWriter

from models.model_registry import Model, Strategy
from environments.env.wrsn.WRSN_ver1 import WRSN
from utilities.util import convert, dict2str
from utilities.trainer import PGTrainer

parser = argparse.ArgumentParser(description="Train rl agent.")
parser.add_argument("--save-path", type=str, nargs="?", default="./", help="Please enter the directory of saving model.")
parser.add_argument("--alg", type=str, nargs="?", default="maddpg", help="Please enter the alg name.")
parser.add_argument("--env", type=str, nargs="?", default="wrsn", help="Please enter the env name.")
parser.add_argument("--alias", type=str, nargs="?", default="", help="Please enter the alias for exp control.")
parser.add_argument("--mode", type=str, nargs="?", default="decentralized", help="Please enter the mode: distributed or decentralised.")
parser.add_argument("--scenario", type=str, nargs="?", default="test.yaml", help="Please input the valid name of an node scenario.")
parser.add_argument("--render", action="store_true", default= False, help="Activate the rendering of the environment.")
parser.add_argument("--n-mcs", type=int, nargs="?", default=2, help="Please input the number off agents")
parser.add_argument("--degree", type=int, nargs="?", default=10, help="Please input the degree of Taylor")
parser.add_argument("--test-mode", type=str, nargs="?", default="single", help="Please input the valid test mode: single or batch.")
argv = parser.parse_args()
# load env args
with open("./args/env_args/"+argv.env+".yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]
scenario = env_config_dict["scenario"].split("/")
scenario[-1] = argv.scenario
env_config_dict["scenario"] = "/".join(scenario)
env_config_dict["n_mcs"] = argv.n_mcs
env_config_dict["degree"] = argv.degree

assert argv.mode in ['distributed', 'decentralized'], "Please input the correct mode, e.g. distributed or decentralised."
env_config_dict["mode"] = argv.mode

# load default args
with open("./args/default.yaml", "r") as f:
    default_config_dict = yaml.safe_load(f)

# load alg args
with open("./args/alg_args/" + argv.alg + ".yaml", "r") as f:
    alg_config_dict = yaml.safe_load(f)["alg_args"]
    alg_config_dict["action_scale"] = env_config_dict["action_scale"]
    alg_config_dict["action_bias"] = env_config_dict["action_bias"]
log_name = "-".join([argv.env, argv.mode, argv.alg, argv.alias])
alg_config_dict = {**default_config_dict, **alg_config_dict}

# define envs
print(env_config_dict)
env = WRSN(env_config_dict)

alg_config_dict["agent_num"] = env.get_num_of_agents()
alg_config_dict["obs_size"] = env.get_obs_size()
alg_config_dict["action_dim"] = env.get_total_actions()
args = convert(alg_config_dict)

# define the save path
if argv.save_path[-1] == "/":
    save_path = argv.save_path
else:
    save_path = argv.save_path+"/"

# create the save folders
if "model_save" not in os.listdir(save_path):
    os.mkdir(save_path + "model_save")
if "tensorboard" not in os.listdir(save_path):
    os.mkdir(save_path + "tensorboard")
if log_name not in os.listdir(save_path + "model_save/"):
    os.mkdir(save_path + "model_save/" + log_name)
if log_name not in os.listdir(save_path + "tensorboard/"):
    os.mkdir(save_path + "tensorboard/" + log_name)
else:
    path = save_path + "tensorboard/" + log_name
    for f in os.listdir(path):
        file_path = os.path.join(path,f)
        if os.path.isfile(file_path):
            os.remove(file_path)

# create the logger
logger = SummaryWriter(save_path + "tensorboard/" + log_name)

model = Model[argv.alg]

strategy = Strategy[argv.alg]

print (f"{args}\n")

if strategy == "pg":
    train = PGTrainer(args, model, env, logger)
elif strategy == "q":
    raise NotImplementedError("This needs to be implemented.")
else:
    raise RuntimeError("Please input the correct strategy, e.g. pg or q.")

with open(save_path + "tensorboard/" + log_name + "/log.txt", "w+") as file:
    alg_args2str = dict2str(alg_config_dict, 'alg_params')
    env_args2str = dict2str(env_config_dict, 'env_params')
    file.write(alg_args2str + "\n")
    file.write(env_args2str + "\n")
for i in range(args.train_episodes_num):
    stat = {}
    train.run(stat, i)
    train.logging(stat)
    if i%args.save_model_freq == args.save_model_freq-1:
        train.print_info(stat)
        th.save({"model_state_dict": train.behaviour_net.state_dict()}, save_path + "model_save/" + log_name + "/model.pt")
        print("The model is saved!\n")

logger.close()
