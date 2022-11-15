import random
import numpy as np
from environments.Extension import *
from environments.network.Package import Package


class Node:

    def __init__(self, location, energy, threshold, capacity, phy_spe):
        self.env = None
        self.net = None

        self.location = location
        self.energy = energy
        self.threshold = threshold
        self.capacity = capacity

        self.node_com_range = phy_spe.node_com_range
        self.node_prob_gp = phy_spe.node_prob_gp
        self.package_size = phy_spe.package_size
        self.er = phy_spe.er
        self.et = phy_spe.et
        self.efs = phy_spe.efs
        self.emp = phy_spe.emp

        self.energyRR = 0
        self.energyCS = 0
        self.id = None
        self.level = None
        self.status = 1
        self.neighbors = []
        self.listTargets = []
        self.log = []
        self.log_energy = 0
        self.check_status()

    def operate(self, t=1):
        """
        The operation of a node
        :param t:
        :returns yield t(s) to time management system every t(s)
        """
        self.probe_targets()
        self.probe_neighbors()
        while True:
            self.log_energy = 0
            yield self.env.timeout(t * 0.5)
            if self.status == 0:
                break
            self.energy = min(self.energy + self.energyRR * t * 0.5, self.capacity)
            if random.random() < self.node_prob_gp:
                self.generate_packages()
            yield self.env.timeout(t * 0.5)

            if self.status == 0:
                break
            self.energy = min(self.energy + self.energyRR * t * 0.5, self.capacity)

            len_log = len(self.log)
            if len_log < 50:
                self.log.append(self.log_energy)
                self.energyCS = (self.energyCS * len_log + self.log_energy) / (len_log + 1)
            else:
                self.energyCS = (self.energyCS * len_log - self.log[0] + self.log_energy) / len_log
                del self.log[0]
                self.log.append(self.log_energy)
        return

    def probe_neighbors(self):
        self.neighbors.clear()
        for node in self.net.listNodes:
            if self != node and euclideanDistance(node.location, self.location) <= self.node_com_range:
                self.neighbors.append(node)

    def probe_targets(self):
        self.listTargets.clear()
        for target in self.net.listTargets:
            if euclideanDistance(self.location, target.location) <= self.node_com_range:
                self.listTargets.append(target)

    def find_receiver(self):
        candidates = [node for node in self.neighbors
                      if node.level < self.level and node.status == 1]
        if len(candidates) > 0:
            distances = [euclideanDistance(candidate.location, self.location) for candidate in
                         candidates]
            return candidates[np.argmin(distances)]
        else:
            return None

    def generate_packages(self):
        for target in self.listTargets:
            self.send_package(Package(target.id, self.package_size))

    def send_package(self, package):
        d0 = (self.efs / self.emp) ** 0.5
        if euclideanDistance(self.location, self.net.baseStation.location) > self.node_com_range:
            receiver = self.find_receiver()
        else:
            receiver = self.net.baseStation
        if receiver is not None:
            d = euclideanDistance(self.location, receiver.location)
            e_send = ((self.et + self.efs * d ** 2) if d <= d0
                      else (self.et + self.emp * d ** 4)) * package.package_size
            if self.energy - self.threshold < e_send:
                self.energy = self.threshold
            else:
                self.energy -= e_send
                receiver.receive_package(package)
                self.log_energy += e_send
        self.check_status()

    def receive_package(self, package):
        e_receive = self.er * package.package_size
        if self.energy - self.threshold < e_receive:
            self.energy = self.threshold
        else:
            self.energy -= e_receive
            self.send_package(package)
            self.log_energy += e_receive
        self.check_status()

    def charger_connection(self, mc):
        if self.status == 0:
            return
        tmp = mc.alpha / (euclideanDistance(self.location, mc.location) + mc.beta) ** 2
        self.energyRR += tmp
        mc.chargingRate += tmp

    def charger_disconnection(self, mc):
        if self.status == 0:
            return
        tmp = mc.alpha / (euclideanDistance(self.location, mc.location) + mc.beta) ** 2
        self.energyRR -= tmp
        mc.chargingRate -= tmp

    def check_status(self):
        if self.energy <= self.threshold:
            self.status = 0
            self.energyCS = 0
