import copy
import random
import simpy
import yaml

envv = 0
with open(r'D:\gym-tc-wrsn\environments\data\test.yaml', 'r') as file:
    envv = yaml.safe_load(file)

print(envv)





