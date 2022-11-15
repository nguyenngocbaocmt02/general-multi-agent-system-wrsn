import copy
import random
import simpy
import yaml

base = [500, 500]
nodes = []
targets = []
n_nodes = 1000
n_targets = 0
for i in range(1000):
    a, b = random.uniform(0.0, 1000.0), random.uniform(0, 1000)
    nodes.append([a, b])
    if random.random() < 0.1:
        targets.append([a, b])
        n_targets += 1

envv = {"base_station" : base, "nodes" : nodes, "targets" : targets}
with open(r'D:\gym-tc-wrsn\environments\data\test.yaml', 'w') as file:
    documents = yaml.dump(envv, file, default_flow_style=False)







