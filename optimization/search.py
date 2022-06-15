import random
import numpy as np
from CompilerQC import *
import cProfile

"""
number of repetition rate from Eq. (4.17)
Simulated Annealing and Boltzmann Machines,
size of neighbourhood approx bei K
"""


class MC:
    """
    Find optimal configuration of qbits given
    an energy_object
    """

    def __init__(
        self,
        energy_object: Energy,
        delta: float = 0.05,
        chi_0: float = 0.8,
        alpha: float = 0.95,
        repetition_rate_factor: float = 1,
        recording: bool = False,
    ):
        """
        repetition rate: how many moves with constant temperature
        delta: how large are the jumps in temperature change
        T_0: initial temperature, can be later set with initial_temperature()
        alpha: decreasing paramter for kirkpatrick temperature schedule
        """
        # self.prof = cProfile.Profile()

        self.energy = energy_object
        self.n_total_steps = 0
        
        # choose qbits
        self.isolation_weight = False
        self.sparse_plaquette_density_weight = False
        self.random_qbit = False
        
        # choose coords
        self.neighbour_coords = False
        self.finite_grid_size = False
        self.padding_finite_grid = 0
        self.infinite_grid_size = False
        
        # initial temperature
        self.init_T_by_std = False
        self.T_1 = False
        self.T_0_estimation = False
        
        # choose temperature
        self.chi_0 = chi_0
        self.delta = delta
        self.alpha = alpha
        self.repetition_rate_factor = repetition_rate_factor

        self.temperature_adaptive = False
        self.temperature_kirkpatrick = False
        self.temperature_kirkpatrick_sigma = False
        self.linear_in_moves = False
        self.temperature_C = False
        self.temperature_linearC = False
        
        # compute total energy once at the beginning
        self.total_energy, self.number_of_plaquettes = self.energy(
            self.energy.polygon_object.qbits
        )

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
            
    def init_from_yaml(self):
        pass

    def reset(self, current_temperature, with_core: bool):
        """ reset complete MC search"""
        if not with_core:
            Qbits.remove_core(self.energy.polygon_object.qbits)
        else:
            Qbits.reinit_coords(self.energy.polygon_object.qbits, with_core=True)
        self.n_total_steps = 0
        self.total_energy, self.number_of_plaquettes = self.energy(
            self.energy.polygon_object.qbits
        )   
        self.current_temperature = current_temperature

    def select_move(self, qbit=None, target_coord=None):
        """
        move random qbit to random coord
        if coord is already occupied, swap qbits
        """
        # select qbit
        if self.isolation_weight:
            isolation_weight = self.energy.penalty_for_isolated_qbits()
            qbit = self.energy.polygon_object.qbits.random_weighted_shell_qbit(weights=isolation_weight)
        if self.sparse_plaquette_density_weight:
            sparse_plaquette_density_weight = self.energy.penalty_for_sparse_plaquette_density()
            qbit = self.energy.polygon_object.qbits.random_weighted_shell_qbit(weights=sparse_plaquette_density_weight)
        if self.random_qbit:
            qbit = self.energy.polygon_object.qbits.random_shell_qbit()
        # select coord
        if self.neighbour_coords:
            target_coord = self.coord_from_neighbour_coords()
        if self.finite_grid_size:
            target_coord = self.coord_from_finite_grid()
        if self.infinite_grid_size:
            target_coord = self.coord_from_infinite_grid()
        assert qbit is not None and target_coord is not None, "oh! there is no qbit or targetcoord"
        # return  
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
        # move
        if type(coord_or_qbit) == tuple:
            self.old_coord = qbit.coord
            qbit.coord = coord_or_qbit
        # swap
        else:
            self.energy.polygon_object.qbits.swap_qbits(qbit, coord_or_qbit)
            self.old_coord = None

    def reverse_move(self, qbit, coord_or_qbit):
        """
        reverse move from apply_move()
        """
        # move back
        if type(coord_or_qbit) == tuple:
            qbit.coord = self.old_coord
        # swack back
        else:
            assert self.old_coord == None, "something went wrong"
            self.energy.polygon_object.qbits.swap_qbits(coord_or_qbit, qbit)
        # reset number of plaquettes
        self.number_of_plaquettes = self.current_number_of_plaquettes

    
    def coord_from_neighbour_coords(self):
        """
        return random coord from neighbour coords
        """
        neighbour_coords = [coord for qbit in self.energy.polygon_object.qbits
                for coord in Qbits.neighbour_coords(qbit.coord)]
        self.possible_coords = list(
                set(neighbour_coords)
                - set(self.energy.polygon_object.qbits.core_qbit_coords)
            )
        return random.choice(self.possible_coords)
    
    @property
    def repetition_rate(self):
        """lenght of markov chain"""
        return self.repetition_rate_factor * self.energy.polygon_object.qbits.graph.K    
    
    def coord_from_infinite_grid(self):
        """
        generate random coord in grid
        sample from envelop rectengular of compiled graph
        for running time reasons: only every repetition_rate times
        """
        if self.n_total_steps % self.repetition_rate == 0:
            possible_coords = self.energy.polygon_object.envelop_rect()
            self.possible_coords = list(
                set(possible_coords)
                - set(self.energy.polygon_object.qbits.core_qbit_coords)
            )
        return random.choice(self.possible_coords)
    
    def coord_from_finite_grid(self):
        if self.n_total_steps == 0:
            possible_coords = self.energy.polygon_object.envelop_rect(self.padding_finite_grid)
            self.possible_coords = list(
                set(possible_coords)
                - set(self.energy.polygon_object.qbits.core_qbit_coords)
            )
        return random.choice(self.possible_coords)        

    def qbits_of_coord(self, coord, xy):
        """
        the core is defined by the core qbits, 
        but also by its corners. the corners form 
        an envelop rectengular, with a side x and a side y,
        we can access each row (y) or column (x) by 
        the coordinate coord.
        TODO: instead of corner, we could consider only 
        the corner of the part we want to swap
        """
        corner = self.energy.polygon_object.core_corner
        return [qbit for qbit in self.energy.polygon_object.qbits
                if qbit.coord[xy] == coord
                and corner[(xy+1)%2][0] <= qbit.coord[(xy+1)%2] <= corner[(xy+1)%2][1]]

    def select_lines_in_core(self):
        """
        sample first the side of the core (x or y)
        and then two coords on this side,
        determine the qbits belonging to these coords,
        note: the corner is coming from Polygons.corner_of_core()
        it returns the minx, maxx, miny, maxy from the rect_envelop
        of the core
        """
        corner = self.energy.polygon_object.core_corner
        xy = random.randint(0,1)
        i, j = random.sample(range(corner[xy][0], corner[xy][1] + 1), 2)
        qbits_i = self.qbits_of_coord(i, xy)
        qbits_j = self.qbits_of_coord(j, xy)  
        return qbits_i, qbits_j, i, j, xy

    def swap_lines_in_core(self, i, j, xy):
        """
        given a side of the core (x or y), 
        and two coordinates (i and j)
        this function will swap
        these coords with each other
        """ 
        for qbit in self.qbits_i:
            coord = list(qbit.coord)
            coord[xy] = j
            qbit.coord = tuple(coord)
        for qbit in self.qbits_j:
            coord = list(qbit.coord)
            coord[xy] = i
            qbit.coord = tuple(coord)
    
    def reverse_swap_lines_in_core(self, i, j, xy):
        self.swap_lines_in_core(j, i, xy)

    def step(self, operation: str = None):
        """
        move qbit or swap qbits and return energy of old and new configuration
        to expand this function for g(x) > 1, select_move() shouls use random.choices(k>1)
        """
        
        if operation == 'move':
            self.qbits_to_move = self.select_move()
            current_energy, current_n_plaqs = self.energy(self.qbits_to_move)
            self.apply_move(*self.qbits_to_move)
            new_energy, new_n_plaqs = self.energy(self.qbits_to_move)

        if operation == 'swap':
            self.qbits_i, self.qbits_j, i, j, xy = self.select_lines_in_core()
            qbits_to_move = self.qbits_i + self.qbits_j
            current_energy, current_n_plaqs = self.energy(qbits_to_move)
            self.lines_in_core_to_swap = (i, j, xy)
            self.swap_lines_in_core(*self.lines_in_core_to_swap)
            new_energy, new_n_plaqs = self.energy(qbits_to_move)
            
        self.current_number_of_plaquettes = self.number_of_plaquettes
        self.number_of_plaquettes += new_n_plaqs - current_n_plaqs
        self.n_total_steps += 1
        return current_energy, new_energy


    def metropolis(self, current_energy, new_energy, operation):
        temperature = self.temperature
        delta_energy = new_energy - current_energy
        if self.recording:
            self.record_temperature.append(temperature)
        if delta_energy < 0:
            self.total_energy += delta_energy
            if self.recording:
                self.record_good_moves.append(self.n_total_steps)
                self.record_acc_probability.append(1)
            return
        acceptance_probability = np.exp(-delta_energy / (temperature + 1e-2))
        if self.recording:
            self.record_acc_probability.append(acceptance_probability)
        if random.random() < acceptance_probability:
            self.total_energy += delta_energy
            if self.recording:
                self.record_bad_moves.append(self.n_total_steps)
            return
        else:
            if operation == 'move':
                self.reverse_move(*self.qbits_to_move)
            if operation == 'swap':
                self.reverse_swap_lines_in_core(*self.lines_in_core_to_swap)

            if self.recording:
                self.record_rejected_moves.append(self.n_total_steps)

    def apply(
        self, operation, n_steps,
    ):
        # self.prof.enable()
        C = self.energy.polygon_object.graph.C
        for i in range(n_steps):
            if self.number_of_plaquettes == C:
                return 0  # ? what is the best value np.nan, None, ?
            self.update_mean_and_variance()
            current_energy, new_energy = self.step(operation)
            self.metropolis(current_energy, new_energy, operation)

        # self.prof.disable()
        return new_energy - current_energy

    def optimization_schedule(self,):
        """
        schedule is a dict = {operation:n_steps}
        set schedule for the optimization
        """
        for operation, n_steps in self.operation_schedule.items():
            self.apply(operation, n_steps)

    def initial_temperature(self, chi_0: float=None, size_of_S=50):
        """
        estimate the initial Temperature T_0,
        chi_0 is the inital acceptance rate of bad moves
        (chi_0 should be close to 1 at the beginning)
        size of S is the number of states -> bad states,
        should converge with increasing size of S
        comment: empirically one can see the the initial temperature
        is already estimated quiet good by size_of_S=1
        benchmark: just T_1, T_0_estimation, or estimate_standard_deviation() / ln(chi0)
        """
        if chi_0 is None:
            chi_0 = self.chi_0
        if self.init_T_by_std:
            return - estimate_standard_deviation(self) / np.log(chi_0)
        
        energy_values = positive_transitions(
            self, size_of_S)
        E_start, E_plus = energy_values.T
        # to slightly accelerate the T_0_estimation process, we use Eq. 7 from the paper
        T_1 = - (E_plus - E_start).mean() / np.log(chi_0)
        if self.T_1:
            return T_1
        if self.T_0_estimation:
            return T_0_estimation(energy_values, chi_0=chi_0, T_1=T_1)
        else:
            print('No initial temperature has been choosen')

    @property
    def temperature(self):
        """
        call this function 
        after n_total_steps > 1,
        since welford needs it like this,
        + 1e-2 to avoid warning in np.exp
        """
        if self.n_total_steps % self.repetition_rate == 0:

            if self.temperature_linearC:
                C = self.energy.polygon_object.qbits.graph.C
                p = self.number_of_plaquettes
                self.current_temperature = self.T_0 * (C - p) / C
                return self.current_temperature

            if self.temperature_C:
                self.current_temperature = (
                    self.T_0
                    * np.exp(
                        self.number_of_plaquettes
                        / (
                            self.energy.polygon_object.qbits.graph.C
                            - self.number_of_plaquettes
                            + 1e-2
                        )
                    )
                 )
                return self.current_temperature

            if self.linear_in_moves:
                self.current_temperature = (
                    self.T_0 * (self.n_moves - self.n_total_steps) / self.n_moves
                )
                return self.current_temperature


            if self.temperature_kirkpatrick:
                new_temperature = self.alpha * self.current_temperature
                self.current_temperature = new_temperature
                return self.current_temperature

            sigmoid = lambda x: 1 - 1 / (1 + np.exp(-x))
            if self.temperature_kirkpatrick_sigma:
                new_temperature = self.alpha * self.current_temperature
                self.current_temperature = new_temperature + sigmoid(
                    self.variance_energy
                )
                return self.current_temperature

            if self.temperature_adaptive:
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


from copy import deepcopy
from CompilerQC import Qbits, Polygons, Energy


def positive_transitions(mc_object, size_of_S=50):
    """
    create states and for each state, it takes one bad neighbour (state with higher energy)
    return list: [[state, bad_neighbour_state]] of length size_of_S
    """
    E=[]
    mc = deepcopy(mc_object)
    for i in range(size_of_S):
        E_start, _ = mc.energy(mc.energy.polygon_object.qbits)
        # find bad neighbour state
        not_found = True
        while not_found:
            current_energy, new_energy = mc.step()
            delta_energy = new_energy - current_energy
            if delta_energy > 0:
                E_plus = E_start + delta_energy
                not_found = False
                Qbits.remove_core(mc.energy.polygon_object.qbits)
            else:
                mc.reverse_move(*mc.qbits_to_move)
        # append energy of state and bad neighbour state
        E.append([E_start, E_plus])
    return np.array(E)


def chi(T, E):
    """formula (5) from https://doi.org/10.1023/B:COAP.0000044187.23143.bd"""
    E_min, E_max = E.T
    return np.exp(-E_max / T).sum() / np.exp(-E_min / T).sum()


def new_T(chi_0, T, E, p=2):
    """formula (6) from https://doi.org/10.1023/B:COAP.0000044187.23143.bd"""
    return T * (np.log(chi(T, E)) / np.log(chi_0)) ** (1 / p)


def T_0_estimation(E, chi_0, T_1, epsilon=0.01):
    """Computing the temperature of simulated annealing"""
    difference = np.abs(chi(T_1, E) - chi_0)
    while difference > epsilon:
        T_n = new_T(chi_0, T_1, E)
        difference = np.abs(chi(T_n, E) - chi_0)
        T_1 = T_n
    return T_1

def estimate_standard_deviation(mc_object, number_of_samples: int=100):
    """return the standard deviation of the energies of number_of_samples
    randomly generated configurations"""
    mc = deepcopy(mc_object)
    energies = []
    for i in range(number_of_samples):
        Qbits.remove_core(mc.energy.polygon_object.qbits)
        energies.append(mc.energy(mc.energy.polygon_object.qbits)[0])
    return np.std(energies)

class MC_core(MC):
    
    def __init__(
        self,
        energy_object: Energy,
        delta: float = 0.05,
        chi_0: float = 0.8,
        alpha: float = 0.95,
        repetition_rate_factor: float = 1,
        recording: bool = False,
    ):
        self.energy = energy_object
        self.n_total_steps = 0
        
        # choose temperature
        self.chi_0 = chi_0
        self.delta = delta
        self.alpha = alpha
        self.repetition_rate_factor = repetition_rate_factor

        self.temperature_adaptive = False
        self.temperature_kirkpatrick = False
        self.temperature_kirkpatrick_sigma = False
        self.linear_in_moves = False
        self.temperature_C = False
        self.temperature_linearC = False
        
        # compute total energy once at the beginning
        self.total_energy, self.number_of_plaquettes = self.energy(
            self.energy.polygon_object.qbits
        )
        
        self.swap_symmetric = True
        
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
 
    def reset(self, current_temperature):
        """ reset complete MC search,
        first part sets qbit coords new, but keeps other qbits 
        attributes"""
        self.energy.polygon_object.nodes_object = (
            Nodes(self.energy.polygon_object.nodes_object.qbits)
        )
        self.n_total_steps = 0
        self.total_energy, self.number_of_plaquettes = self.energy(
            self.energy.polygon_object.qbits
        )   
        self.current_temperature = current_temperature
        
    def reverse_swap(self, node_i, node_j):
        self.energy.polygon_object.nodes_object.swap_nodes(node_j, node_i)

    def step(self, operation: str = None):
        """
        move qbit or swap qbits and return energy of old and new configuration
        to expand this function for g(x) > 1, select_move() shouls use random.choices(k>1)
        """
        node_i, node_j = self.energy.polygon_object.nodes_object.nodes_to_swap()
        self.nodes_to_move = node_i, node_j
        qbits_to_move = self.energy.polygon_object.nodes_object.qbits_of_nodes(self.nodes_to_move)
        current_energy, current_n_plaqs = self.energy(qbits_to_move)
        self.energy.polygon_object.nodes_object.swap_nodes(*self.nodes_to_move)
        new_energy, new_n_plaqs = self.energy(qbits_to_move)
        self.current_number_of_plaquettes = self.number_of_plaquettes
        self.number_of_plaquettes += new_n_plaqs - current_n_plaqs
        self.n_total_steps += 1
        return current_energy, new_energy
    
    def metropolis(self, current_energy, new_energy, operation):
        temperature = self.temperature
        delta_energy = new_energy - current_energy
        if self.recording:
            self.record_temperature.append(temperature)
        if delta_energy < 0:
            self.total_energy += delta_energy
            if self.recording:
                self.record_good_moves.append(self.n_total_steps)
                self.record_acc_probability.append(1)
            return
        acceptance_probability = np.exp(-delta_energy / (temperature + 1e-2))
        if self.recording:
            self.record_acc_probability.append(acceptance_probability)
        if random.random() < acceptance_probability:
            self.total_energy += delta_energy
            if self.recording:
                self.record_bad_moves.append(self.n_total_steps)
            return
        else:
            self.reverse_swap(*self.nodes_to_move)
            self.number_of_plaquettes = self.current_number_of_plaquettes
            if self.recording:
                self.record_rejected_moves.append(self.n_total_steps)

       
    def initial_temperature(self, chi_0: float=None, number_of_samples=25):
        """
        estimate the initial Temperature T_0,
        chi_0 is the inital acceptance rate of bad moves
        (chi_0 should be close to 1 at the beginning)
        size of S is the number of states -> bad states,
        should converge with increasing size of S
        comment: empirically one can see the the initial temperature
        is already estimated quiet good by size_of_S=1
        benchmark: just T_1, T_0_estimation, or estimate_standard_deviation() / ln(chi0)
        """
        if chi_0 is None:
            chi_0 = self.chi_0   
        energies = []
        for i in range(number_of_samples):
            energies.append(self.total_energy)
            self.reset(current_temperature=None)
        return - np.std(energies) / np.log(chi_0)
   