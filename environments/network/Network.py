class Network:
    def __init__(self, env, listNodes, baseStation, listTargets):
        self.env = env

        self.listNodes = listNodes
        self.baseStation = baseStation
        self.listTargets = listTargets

        self.targets_active = [1 for i in range(len(self.listTargets))]
        baseStation.env = self.env
        baseStation.net = self
        it = 0
        for node in self.listNodes:
            node.env = self.env
            node.net = self
            node.id = it
            it += 1
        it = 0
        for target in listTargets:
            target.id = it
            it += 1

    def setLevels(self):
        for node in self.listNodes:
            node.level = -1
        tmp1 = []
        tmp2 = []
        for node in self.baseStation.direct_nodes:
            if node.status == 1:
                node.level = 1
                tmp1.append(node)
        for i in range(len(self.targets_active)):
            self.targets_active[i] = 0
        while True:
            if len(tmp1) == 0:
                break
            for node in tmp1:
                for target in node.listTargets:
                    self.targets_active[target.id] = 1
                for neighbor in node.neighbors:
                    if neighbor.status == 1 and neighbor.level == -1:
                        tmp2.append(neighbor)
                        neighbor.level = node.level + 1
            tmp1 = tmp2[:]
            tmp2.clear()
        return

    def operate(self, t=1, max_time=1):
        for node in self.listNodes:
            self.env.process(node.operate(t=t))
        self.env.process(self.baseStation.operate(t=t))
        while True:
            yield self.env.timeout(t / 10.0)
            self.setLevels()
            alive = self.check_targets()
            yield self.env.timeout(9.0 * t / 10.0)
            if alive == 0 or self.env.now >= max_time:
                break
            tmp = 0
            for node in self.listNodes:
                if node.status == 0:
                    tmp += 1
        return

    def check_targets(self):
        return min(self.targets_active)
