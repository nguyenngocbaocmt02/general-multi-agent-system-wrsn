import torch
import argparse
import yaml
import pickle

from models.model_registry import Model, Strategy
from environments.env.wrsn.WRSN_ver1 import WRSN
from utilities.util import convert
from utilities.tester import PGTester

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

# for one-day test
env_config_dict["episode_limit"] = 240

# load default args
with open("./args/default.yaml", "r") as f:
    default_config_dict = yaml.safe_load(f)
default_config_dict["max_steps"] = 240
env = WRSN(env_config_dict)
# load alg args
with open("./args/alg_args/"+argv.alg+".yaml", "r") as f:
    alg_config_dict = yaml.safe_load(f)["alg_args"]
    alg_config_dict["action_scale"] = env_config_dict["action_scale"]
    alg_config_dict["action_bias"] = env_config_dict["action_bias"]

log_name = "-".join([argv.env, argv.node, argv.target, argv.mode, argv.alg, argv.alias])
alg_config_dict = {**default_config_dict, **alg_config_dict}

# define envs


alg_config_dict["agent_num"] = env.get_num_of_agents()
alg_config_dict["obs_size"] = env.get_obs_size()
alg_config_dict["action_dim"] = env.get_total_actions()

alg_config_dict["cuda"] = False
args = convert(alg_config_dict)

# define the save path
if argv.save_path[-1] == "/":
    save_path = argv.save_path
else:
    save_path = argv.save_path+"/"

LOAD_PATH = save_path+log_name+"/model.pt"

model = Model[argv.alg]

strategy = Strategy[argv.alg]

if args.target:
    target_net = model(args)
    behaviour_net = model(args, target_net)
else:
    behaviour_net = model(args)
#checkpoint = torch.load(LOAD_PATH, map_location='cpu') if not args.cuda else torch.load(LOAD_PATH)
#behaviour_net.load_state_dict(checkpoint['model_state_dict'])

print(f"{args}\n")

if strategy == "pg":
    test = PGTester(args, behaviour_net, env, argv.render)
elif strategy == "q":
    raise NotImplementedError("This needs to be implemented.")
else:
    raise RuntimeError("Please input the correct strategy, e.g. pg or q.")

if argv.test_mode == 'single':
    # record = test.run(199, 23, 2) # (day, hour, 3min)
    # record = test.run(730, 23, 2) # (day, hour, 3min)
    record = test.run()
    with open('test_record_'+log_name+f'_day{argv.test_day}'+'.pickle', 'wb') as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
elif argv.test_mode == 'batch':
    record = test.batch_run(10)
    with open('test_record_'+log_name+'_'+argv.test_mode+'.pickle', 'wb') as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
