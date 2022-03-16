import random
import numpy as np
from copy import deepcopy
from CompilerQC import Energy


class MC:
    """
    Find optimal configuration of qbits given
    an energy_object
    """

    def __init__(
        self,
        energy_object: Energy,
        delta: float = 1,
        chi_0: float = 0.8,
        repetition_rate: int = None,
        recording: bool = False,
    ):
        """
        repetition rate: how many moves with constant temperature
        delta: how large are the jumps in temperature change
        T_0: initial temperture, can be later set with initial_temperature()
        """
        self.energy = energy_object
        self.n_total_steps = 0
        self.chi_0 = chi_0
        self.delta = delta
        self.temperature_kirkpatrick = False
        self.repetition_rate = repetition_rate
        if self.repetition_rate is None:
            # number of repetition rate from Eq. (4.17) Simulated Annealing and Boltzmann Machines
            # size of neighbourhood approx bei K
            self.repetition_rate = self.energy.polygon_object.qbits.graph.K
        # compute total energy once at the beginning
        self.total_energy = self.energy(self.energy.polygon_object.qbits)
        # record
        self.recording = recording
        if self.recording: 
            self.record_temperature = []
            self.record_total_energy = []
            self.record_mean_energy = []
            self.record_variance_energy = []
            self.record_good_moves = []
            self.record_bad_moves = []
            self.record_rejected_moves = []
            self.record_acc_probability = []
            self.record_delta_energy = []

    def select_move(self):
        """
        move random qbit to random coord
        if coord is already occupied, swap qbits
        """
        qbit = self.energy.polygon_object.qbits.random()
        target_coord = self.generate_coord()
        if target_coord in self.energy.polygon_object.qbits.coords:
            target_qbit = self.energy.polygon_object.qbits.qbit_from_coord(target_coord)
            return qbit, target_qbit
        else:
            return qbit, target_coord

    def apply_move(self, qbit, coord_or_qbit):
        """
        move qbit to coord
        if coord is a qbit (occupied coord), swap qbits
        """
        if type(coord_or_qbit) == tuple:
            qbit.coord = coord_or_qbit
        else:
            self.energy.polygon_object.qbits.swap_qbits(qbit, coord_or_qbit)

    def generate_coord(self):
        """
        generate random coord in grid
        ###change generate coord only every e.g. 5 steps
        """
        possible_coords = self.energy.polygon_object.envelop_rect()
        return random.choice(possible_coords)

    def step(
        self,
        operation: str,
    ):
        """
        move qbit or swap qbits and return energy of old and new configuration
        to expand this function for g(x) > 1, select_move() shouls use random.choices(k>1)
        """
        self.current_qbits = deepcopy(self.energy.polygon_object.qbits)
        qbits_to_move = self.select_move()
        # self.qbits_to_move for analysis
        self.qbits_to_move = qbits_to_move
        current_energy = self.energy(qbits_to_move)
        self.apply_move(*qbits_to_move)
        new_energy = self.energy(qbits_to_move)
        return current_energy, new_energy

    def initial_temperature(self, size_of_S=100):
        """
        estimate the initial Temperature T_0,
        chi_0 is the inital acceptance rate of bad moves
        (chi_0 should be close to 1 at the beginning)
        size of S is the number of states -> bad states,
        should converge with increasing size of S
        """
        positive_transitions = postive_transitions(
            self.energy.polygon_object.qbits.graph, size_of_S
        )
        return T_0_estimation(positive_transitions, self.chi_0)
    
    @property
    def temperature(self):
        if self.n_total_steps % self.repetition_rate == 0:
            
            if self.temperature_kirkpatrick:
                alpha = 0.8
                new_temperature = alpha * self.current_temperature
                self.current_temperature = new_temperature   
            
            else:
                new_temperature = self.current_temperature / (
                    1
                    + self.current_temperature
                    * np.log(1 + self.delta)
                    / (3 * np.sqrt(self.variance_energy) + 1e-3)
                )
                self.current_temperature = new_temperature
                
        return self.current_temperature

    def update_mean_and_variance(self):
        """
        update mean and variance on the fly according to Welford algorithm on wikipedia
        m = n - 1"""
        if self.n_total_steps % self.repetition_rate == 0:
            self.mean_energy, self.variance_energy = self.total_energy, 0
        self.mean_energy, self.variance_energy = self.update_mean_variance(
            self.n_total_steps,
            self.mean_energy,
            self.variance_energy,
            self.total_energy,
        )
        if self.recording:
            self.record_total_energy.append(self.total_energy)
            self.record_mean_energy.append(self.mean_energy)
            self.record_variance_energy.append(self.variance_energy)

    def update_mean_variance(self, n, µ_m, s_m2, x_n):
        """
        update mean and variance on the fly according to Welford algorithm on wikipedia
        m = n - 1"""
        n = n % self.repetition_rate + 1
        µ_n = µ_m + (x_n - µ_m) / (n + 1)
        s_n2 = s_m2 + (x_n - µ_m) ** 2 / (n + 1) - s_m2 / n
        return µ_n, s_n2

    def metropolis(self, current_energy, new_energy):
        self.n_total_steps += 1
        temp = self.temperature
        if self.recording: self.record_temperature.append(temp)
        delta_energy = new_energy - current_energy
        if self.recording: self.record_delta_energy.append(delta_energy)
        if delta_energy < 0:
            self.total_energy += delta_energy
            if self.recording:
                self.record_good_moves.append(self.n_total_steps)
                self.record_acc_probability.append(1)
            return
        acceptance_probability = np.exp(-delta_energy / temp)
        if self.recording: self.record_acc_probability.append(acceptance_probability)
        if random.random() < acceptance_probability:
            self.total_energy += delta_energy
            if self.recording: self.record_bad_moves.append(self.n_total_steps)
            return
        else:
            if self.recording: self.record_rejected_moves.append(self.n_total_steps)
            self.energy.polygon_object.qbits = self.current_qbits

    def apply(
        self,
        operation,
        n_steps,
    ):
        for i in range(n_steps):
            if (
                self.energy.polygon_object.number_of_plaqs
                == self.energy.polygon_object.graph.C
            ):
                return 0  # ? what is the best value np.nan, None, ?
            current_energy, new_energy = self.step(operation)
            self.update_mean_and_variance()
            self.metropolis(current_energy, new_energy)

        return new_energy - current_energy

    def optimization_schedule(
        self,
    ):
        """
        schedule is a dict = {operation:n_steps}
        set schedule for the optimization
        """
        for operation, n_steps in self.operation_schedule.items():
            self.apply(operation, n_steps)


from CompilerQC import Qbits, Polygons, Energy


def postive_transitions(graph, size_of_S=100, E=[]):
    """
    create states and for each state, it takes one bad neighbour (state with higher energy)
    return list: [[state, bad_neighbour_state]] of length size_of_S
    """
    for i in range(size_of_S):
        # initialise state
        qbits = Qbits.init_qbits_randomly(graph)
        polygon_object = Polygons(qbits)
        energy = Energy(polygon_object, scaling_for_plaq3=0, scaling_for_plaq4=0)
        mc = MC(energy)
        mc_min = deepcopy(mc)
        mc_min.energy(mc_min.energy.polygon_object.qbits)

        # find bad neighbour state
        not_found = True
        while not_found:
            current_energy, new_energy = mc.step("")
            delta_energy = new_energy - current_energy
            if delta_energy > 0:
                mc_max = mc
                not_found = False
            else:
                mc_max = deepcopy(mc_min)

        # append energy of state and bad neighbour state
        E.append(
            [
                mc_min.energy(mc_min.energy.polygon_object.qbits),
                mc_max.energy(mc_max.energy.polygon_object.qbits),
            ]
        )
    return np.array(E)


def chi(T, E):
    """formula (5) from https://doi.org/10.1023/B:COAP.0000044187.23143.bd"""
    E_min, E_max = E.T
    return np.exp(-E_max / T).sum() / np.exp(-E_min / T).sum()


def new_T(chi_0, T, E, p=0.5):
    """formula (6) from https://doi.org/10.1023/B:COAP.0000044187.23143.bd"""
    return T * (np.log(chi(T, E)) / np.log(chi_0)) ** 1 / p


def T_0_estimation(E, chi_0=0.8, T_1=1000, epsilon=0.0001):
    """Computing the temperature of simulated annealing"""
    difference = epsilon + 1
    while difference > epsilon:
        T_n = new_T(chi_0, T_1, E)
        difference = np.abs(chi(T_n, E) - chi(T_1, E))
        T_1 = T_n
    return T_1
