import random
import numpy as np
from CompilerQC import *
import cProfile
# for convolution in cluster search
from scipy import signal, misc
from scipy.ndimage.measurements import label
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
        self.qbit_with_same_node = False
        
        # choose coords
        self.same_node_coords = False
        self.finite_grid_size = False
        self.padding_finite_grid = 0
        self.infinite_grid_size = False
        self.shell_search = False
        self.min_plaquette_density_in_softcore = 0.75
        self.corner = None
        # TODO: only coords which have node as neighbozr -> form a line
        
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
        
        # search schedule
        self.swap_probability = 0
        
        # compute total energy once at the beginning
        self.total_energy, self.number_of_plaquettes = self.energy(
            self.energy.polygon_object.nodes_object.qbits
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

    def reset(self, current_temperature, keep_core: bool):
        """ reset complete MC search"""
        if not keep_core:
            Qbits.remove_core(self.energy.polygon_object.nodes_object.qbits)
        else:
            Qbits.reinit_coords(self.energy.polygon_object.nodes_object.qbits, with_core=True)
        self.n_total_steps = 0
        self.total_energy, self.number_of_plaquettes = self.energy(
            self.energy.polygon_object.nodes_object.qbits
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
            qbit = self.energy.polygon_object.nodes_object.qbits.random_weighted_shell_qbit(weights=isolation_weight)
        if self.sparse_plaquette_density_weight:
            sparse_plaquette_density_weight = self.energy.penalty_for_sparse_plaquette_density()
            qbit = self.energy.polygon_object.nodes_object.qbits.random_weighted_shell_qbit(weights=sparse_plaquette_density_weight)
        if self.random_qbit:
            qbit = self.energy.polygon_object.nodes_object.qbits.random_shell_qbit()
        # select coord
        if self.qbit_with_same_node:
            target_coord = self.coord_from_softcore()
            qbit = self.qbit_with_same_node_as_neighbour(target_coord)
        if self.qbit_with_same_node_around_core:
            target_coord = self.coord_around_core()
            qbit = self.qbit_with_same_node_as_neighbour_around_core(target_coord)
        if self.same_node_coords:
            qbit, possible_coords = self.coord_from_same_node(qbit)
            target_coord = random.choice(possible_coords)
        if self.finite_grid_size:
            target_coord = self.coord_from_finite_grid()
        if self.shell_search:
            target_coord = self.coord_from_softcore()
        if self.shell_search_around_core:
            target_coord = self.coord_around_core()
        if self.infinite_grid_size:
            target_coord = self.coord_from_infinite_grid()
        assert qbit is not None and target_coord is not None, "oh! there is no qbit or targetcoord"
        # return  
        if target_coord in self.energy.polygon_object.nodes_object.qbits.coords:
            target_qbit = self.energy.polygon_object.nodes_object.qbits.qbit_from_coord(target_coord)
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
            self.energy.polygon_object.nodes_object.qbits.swap_qbits(qbit, coord_or_qbit)
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
            self.energy.polygon_object.nodes_object.qbits.swap_qbits(coord_or_qbit, qbit)
        # reset number of plaquettes
        self.number_of_plaquettes = self.current_number_of_plaquettes

    def qbit_with_same_node_as_neighbour(self, target_coord):
        neighbour_qbits_in_envelop = list(map(self.energy.polygon_object.nodes_object.qbits.coord_to_qbit_dict.get, set(self.coords_in_softcore()[0]).intersection(self.energy.polygon_object.nodes_object.qbits.neighbour_coords_without_origin(target_coord))))
        neighbour_nodes_in_envelop = [node for qbit in neighbour_qbits_in_envelop for node in qbit.qubit if qbit is not None]
        potential_nodes = [node for node in neighbour_nodes_in_envelop if neighbour_nodes_in_envelop.count(node) < 4 ]
        qbits_to_propose = ([qbit.qubit for qbit in self.energy.polygon_object.nodes_object.qbits.shell_qbits if set(qbit.qubit).intersection(potential_nodes)])
        if qbits_to_propose == []:
            return self.energy.polygon_object.nodes_object.qbits.random_shell_qbit()
        else:
            return random.choice(qbits_to_propose)
        
    def coord_from_same_node(self, qbit_to_move):
        """
        return random coord from neighbour coords of qbits which 
        are from the same node as qbit_to_move
        """
        """
        softcore = core + plaquettes around it from shell qbits
        """
        if self.corner is None:
            self.corner = self.energy.polygon_object.core_corner
        # check every few hundert steps if shell should be expanded
        if self.n_total_steps % (self.energy.polygon_object.nodes_object.qbits.graph.K * 10) == 0:
            if self.plaquette_density_in_softcore() > self.min_plaquette_density_in_softcore:
                 # if enough plaquettes in this area, expand area
                (min_x, max_x), (min_y, max_y) = self.corner
                self.corner = (min_x - 1, max_x + 1), (min_y - 1, max_y + 1)
            possible_coords = self.coords_in_softcore()[0]
            self.possible_coords = (
                set(possible_coords)
                - set(self.energy.polygon_object.nodes_object.qbits.core_qbit_coords)
            )                

        neighbour_coords = [coord for qbit in self.energy.polygon_object.nodes_object.qbits for coord in Qbits.neighbour_coords_without_origin(qbit.coord) if qbit.qubit[0] in qbit_to_move.qubit
                        or qbit.qubit[1] in qbit_to_move.qubit]
        possible_coords = list(self.possible_coords.intersection(neighbour_coords))
        if possible_coords == []:
            qbit_to_move = self.energy.polygon_object.nodes_object.qbits.random_shell_qbit()
            qbit_to_move, possible_coords = self.coord_from_same_node(qbit_to_move)

        return qbit_to_move, possible_coords

    @property
    def repetition_rate(self):
        """lenght of markov chain"""
        return self.repetition_rate_factor * self.energy.polygon_object.nodes_object.qbits.graph.K    
    
    def coord_from_infinite_grid(self):
        """
        generate random coord in grid
        sample from envelop rectengular of compiled graph
        for running time reasons: only every repetition_rate times
        """
        if self.n_total_steps % self.repetition_rate == 0:
            possible_coords = self.energy.polygon_object.nodes_object.qbits.envelop_rect()
            self.possible_coords = list(
                set(possible_coords)
                - set(self.energy.polygon_object.nodes_object.qbits.core_qbit_coords)
            )
        return random.choice(self.possible_coords)
    
    def coord_from_finite_grid(self):
        if self.n_total_steps == 0:
            possible_coords = self.energy.polygon_object.nodes_object.qbits.envelop_rect(self.padding_finite_grid)
            self.possible_coords = list(
                set(possible_coords)
                - set(self.energy.polygon_object.nodes_object.qbits.core_qbit_coords)
            )
        return random.choice(self.possible_coords)        

    def coords_around_core(self):
        """
        designing a plaquette density function for this function is not trivial?
        """
        coords_around_core = []
        core_qbit_coords = self.energy.polygon_object.nodes_object.qbits.core_qbit_coords
        for coord in core_qbit_coords:
            coords_around_core.extend(Qbits.neighbour_coords_without_origin(coord))
        return list(set(coords_around_core) - set(core_qbit_coords))
 
    def qbit_with_same_node_as_neighbour_around_core(self, target_coord):
        neighbour_qbits_in_envelop = list(map(self.energy.polygon_object.nodes_object.qbits.coord_to_qbit_dict.get, set(self.coords_around_core()[0]).intersection(self.energy.polygon_object.nodes_object.qbits.neighbour_coords_without_origin(target_coord))))
        neighbour_nodes_in_envelop = [node for qbit in neighbour_qbits_in_envelop for node in qbit.qubit if qbit is not None]
        potential_nodes = [node for node in neighbour_nodes_in_envelop if neighbour_nodes_in_envelop.count(node) < 4 ]
        qbits_to_propose = ([qbit.qubit for qbit in self.energy.polygon_object.nodes_object.qbits.shell_qbits if set(qbit.qubit).intersection(potential_nodes)])
        if qbits_to_propose == []:
            return self.energy.polygon_object.nodes_object.qbits.random_shell_qbit()
        else:
            return random.choice(qbits_to_propose)
    
    def coords_in_softcore(self):
        (min_x, max_x), (min_y, max_y) = self.corner
        coords_in_softcore = [(i, j) for i in range(min_x, max_x + 1) for j in range(min_y, max_y + 1)]
        number_of_cells_in_softcore = (max_x - min_x) * (max_y - min_y)
        
        max_x_and_y_coords = ([(max_x, y) for y in range(min_y, max_y + 1)] + [(max_y, x) for x in range(min_x, max_x)])
        return coords_in_softcore, number_of_cells_in_softcore, max_x_and_y_coords
    
    def plaquette_density_in_softcore(self):
        coords_in_softcore, number_of_cells_in_softcore, max_x_and_y_coords = self.coords_in_softcore()
        number_of_plaquettes_in_softcore = len(self.energy.polygon_object.nodes_object.qbits.found_plaquettes_around_coords(
            list(set(coords_in_softcore) - set(max_x_and_y_coords))))
        plaquette_density = number_of_plaquettes_in_softcore / number_of_cells_in_softcore       
        return plaquette_density
    
    def edge_coords_of_envelop(self):
        """ returns the coords of the edge of corner
        this function is currently not used"""
        (min_x, max_x), (min_y, max_y) = self.corner
        return ([(X, y) for y in range(min_y, max_y + 1) for X in [min_x, max_x]]
+ [(Y, x) for x in range(min_x + 1, max_x) for Y in [min_y, max_y]])
            
    def coord_from_softcore(self):
        """
        softcore = core + plaquettes around it from shell qbits
        """
        if self.corner is None:
            self.corner = self.energy.polygon_object.core_corner
            # if no core has been set, use square around the center as start for shell search
            if self.corner is None:
                x, y = self.energy.polygon_object.center_of_coords(self.energy.polygon_object.nodes_object.qbits.coords)
                self.corner = (int(x - 0.5), int(x + 0.5)), (int(y - 0.5), int(y + 0.5))
            possible_coords = self.coords_in_softcore()[0]
            self.possible_coords = list(
                set(possible_coords)
                - set(self.energy.polygon_object.nodes_object.qbits.core_qbit_coords)
            )        
            
        # check every few hundert steps if shell should be expanded
        if self.n_total_steps % (self.energy.polygon_object.nodes_object.qbits.graph.K * 10) == 0:
            if self.plaquette_density_in_softcore() > self.min_plaquette_density_in_softcore:
                 # if enough plaquettes in this area, expand area
                (min_x, max_x), (min_y, max_y) = self.corner
                self.corner = (min_x - 1, max_x + 1), (min_y - 1, max_y + 1)
                possible_coords = self.coords_in_softcore()[0]
                self.possible_coords = list(
                    set(possible_coords)
                    - set(self.energy.polygon_object.nodes_object.qbits.core_qbit_coords)
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
        return [qbit for qbit in self.energy.polygon_object.nodes_object.qbits
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
        
        # choose line, and upper or lower / lefter or righter neighbour line
        i = random.choice(range(corner[xy][0], corner[xy][1] + 1))
        if i == corner[xy][0]:
            j = i + 1
        elif i == corner[xy][1]:
            j = i - 1
        else:
            j = i + (-1) ** random.randint(0, 1)
        
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

    def step(self):
        """
        move qbit or swap qbits and return energy of old and new configuration
        to expand this function for g(x) > 1, select_move() shouls use random.choices(k>1)
        """
        
        # swap adjacent lines with swap_probability
        if random.random() < self.swap_probability:
            operation = 'swap'
            self.qbits_i, self.qbits_j, i, j, xy = self.select_lines_in_core()
            qbits_to_move = self.qbits_i + self.qbits_j
            current_energy, current_n_plaqs = self.energy(qbits_to_move)
            self.lines_in_core_to_swap = (i, j, xy)
            self.swap_lines_in_core(*self.lines_in_core_to_swap)
            new_energy, new_n_plaqs = self.energy(qbits_to_move)
            
        # if not swapping, move a qbit
        else:
            operation = 'move'
            self.qbits_to_move = self.select_move()
            current_energy, current_n_plaqs = self.energy(self.qbits_to_move)
            self.apply_move(*self.qbits_to_move)
            new_energy, new_n_plaqs = self.energy(self.qbits_to_move)
          
        self.current_number_of_plaquettes = self.number_of_plaquettes
        self.number_of_plaquettes += new_n_plaqs - current_n_plaqs
        self.n_total_steps += 1
        return current_energy, new_energy, operation


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
        self, n_steps,
    ):
        # self.prof.enable()
        for _ in range(n_steps):
            if self.number_of_plaquettes == self.energy.polygon_object.nodes_object.qbits.graph.C:
                return 0  # ? what is the best value np.nan, None, ?
            self.update_mean_and_variance()
            current_energy, new_energy, operation = self.step()
            self.metropolis(current_energy, new_energy, operation)

        # self.prof.disable()
        return new_energy - current_energy

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
                C = self.energy.polygon_object.nodes_object.qbits.graph.C
                p = self.number_of_plaquettes
                self.current_temperature = self.T_0 * (C - p) / C
                return self.current_temperature

            if self.temperature_C:
                self.current_temperature = (
                    self.T_0
                    * np.exp(
                        self.number_of_plaquettes
                        / (
                            self.energy.polygon_object.nodes_object.qbits.graph.C
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
    
    
    def update_qbits_from_dict(self, core_qbit_to_core_dict, assign_to_core: bool=True):
        self.energy.polygon_object.nodes_object.qbits.update_qbits_from_dict(core_qbit_to_core_dict, assign_to_core=True)
        self.total_energy, self.number_of_plaquettes = self.energy(
            self.energy.polygon_object.nodes_object.qbits
        )

        
    def add_ancillas(self, ancillas, only_four_cycles=False):
        """
        ancillas are also conncected via 3cycles, 
        this can be changed with only_four_cycles=True
        """
        self.energy.polygon_object.nodes_object.qbits.graph.update_ancillas(ancillas)
        new_polygons = self.energy.polygon_object.add_ancillas_to_polygons(ancillas, only_four_cycles=False)
        ancilla_qbits = self.energy.polygon_object.nodes_object.qbits.add_ancillas_to_qbits(ancillas, new_polygons)
        self.energy.polygon_object.nodes_object.add_ancillas_to_nodes(ancillas)

        delta_energy, delta_number_of_plaquettes = self.energy(ancilla_qbits)
        self.total_energy += delta_energy
        self.number_of_plaquettes += delta_number_of_plaquettes
        
    def remove_ancillas(self, ancillas):
        delta_energy, delta_number_of_plaquettes = self.energy(
            [self.energy.polygon_object.nodes_object.qbits[ancilla] for ancilla in ancillas])
        self.total_energy -= delta_energy
        self.number_of_plaquettes -= delta_number_of_plaquettes
        
        self.energy.polygon_object.remove_ancillas_from_polygons(ancillas)
        self.energy.polygon_object.nodes_object.remove_ancillas_from_nodes(ancillas)
        self.energy.polygon_object.nodes_object.qbits.remove_ancillas_from_qbits(ancillas)
        # remove ancillas from graph, cycles are not updatet!
        self.energy.polygon_object.nodes_object.qbits.graph.remove_ancillas(ancillas)
        

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
        E_start, _ = mc.energy(mc.energy.polygon_object.nodes_object.qbits)
        # find bad neighbour state
        not_found = True
        while not_found:
            current_energy, new_energy = mc.step()
            delta_energy = new_energy - current_energy
            if delta_energy > 0:
                E_plus = E_start + delta_energy
                not_found = False
                Qbits.remove_core(mc.energy.polygon_object.nodes_object.qbits)
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
        Qbits.remove_core(mc.energy.polygon_object.nodes_object.qbits)
        energies.append(mc.energy(mc.energy.polygon_object.nodes_object.qbits)[0])
    return np.std(energies)

class MC_core(MC):
    
    square_filter = np.array(
        [[1,1],
         [1,1]]) 
    structure = np.array(
        [[1,1,1],
         [1,1,1],
         [1,1,1]])
    
    def __init__(
        self,
        energy_object: Energy,
        delta: float = 0.05,
        chi_0: float = 0.8,
        alpha: float = 0.95,
        repetition_rate_factor: float = 1,
        recording: bool = False,
        cluster_shuffling_probability: float=0.0,
        ancilla_insertion_probability: float=0.0,
        ancilla_deletion_probability: float=0.0,
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
        
        # cluster swap
        self.shape_x = self.shape_y = self.energy.polygon_object.nodes_object.qbits.graph.N
        self.indices = np.indices((self.shape_x + 2, self.shape_y + 2)).T[:,:,]
        self.cluster_shuffling_probability = cluster_shuffling_probability
        
        #ancillas
        self.ancilla_insertion_probability = ancilla_insertion_probability
        self.ancilla_deletion_probability = ancilla_deletion_probability
        self.only_four_cycles_for_ancillas = False
        
        # compute total energy once at the beginning
        self.total_energy, self.number_of_plaquettes = self.energy(
            self.energy.polygon_object.nodes_object.qbits
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
 
    def reset(self, current_temperature):
        """ reset complete MC search,
        first part sets qbit coords new, but keeps other qbits 
        attributes"""
        self.energy.polygon_object.nodes_object = (
            Nodes(self.energy.polygon_object.nodes_object.qbits)
        )
        self.n_total_steps = 0
        self.total_energy, self.number_of_plaquettes = self.energy(
            self.energy.polygon_object.nodes_object.qbits
        )   
        self.current_temperature = current_temperature
        
    def reverse_swap(self, node_i, node_j):
        self.energy.polygon_object.nodes_object.swap_nodes(node_j, node_i)

    def step(self):
        """
        move qbit or swap qbits and return energy of old and new configuration
        to expand this function for g(x) > 1, select_move() shouls use random.choices(k>1)
        """
        if random.random() < self.ancilla_insertion_probability:
            ancillas = self.energy.polygon_object.nodes_object.propose_ancillas(5)
            if ancillas != {}:
                self.add_ancillas(ancillas, only_four_cycles=self.only_four_cycles_for_ancillas)
            
        if random.random() < self.ancilla_deletion_probability:
            ancillas_to_remove = self.energy.polygon_object.nodes_object.propose_ancillas_to_remove()
            if ancillas_to_remove != {}:
                self.remove_ancillas(ancillas_to_remove)
            
        if random.random() < self.cluster_shuffling_probability:
            self.clusters_swap()
            #self.greedy_clusters_swap(number_of_cluster_swaps=10)
        
        node_i, node_j = self.energy.polygon_object.nodes_object.nodes_to_swap()
        self.nodes_to_move = node_i, node_j
        qbits_to_move = self.energy.polygon_object.nodes_object.qbits_of_nodes(self.nodes_to_move)
        current_energy, current_n_plaqs = self.energy(qbits_to_move)
        self.energy.polygon_object.nodes_object.swap_nodes(*self.nodes_to_move)
        new_energy, new_n_plaqs = self.energy(qbits_to_move)
        self.current_number_of_plaquettes = self.number_of_plaquettes
        self.number_of_plaquettes += new_n_plaqs - current_n_plaqs
        self.n_total_steps += 1
        return current_energy, new_energy, 'placeholder'
    
    def metropolis(self, current_energy, new_energy, placeholder):
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
    
    
    @staticmethod
    def consolidate(sets):
        """from stackoverflow, used in unique_clusters()"""
        # http://rosettacode.org/wiki/Set_consolidation#Python:_Iterative
        setlist = [s for s in sets if s]
        for i, s1 in enumerate(setlist):
            if s1:
                for s2 in setlist[i+1:]:
                    intersection = s1.intersection(s2)
                    if intersection:
                        s2.update(s1)
                        s1.clear()
                        s1 = s2
        return [s for s in setlist if s]

    @staticmethod
    def wrapper(seqs):
        """from stackoverflow, used in unique_clusters()"""
        consolidated = MC_core.consolidate(map(set, seqs))
        groupmap = {x: i for i,seq in enumerate(consolidated) for x in seq}
        output = {}
        for seq in seqs:
            target = output.setdefault(groupmap[seq[0]], [])
            target.append(seq)
        return list(output.values())
    
    def get_clusters(self):
        """
        a clusters is simply a list of nodes of a conncected set of plaquettes,
        this function returns all clusters in form of the nodes order,
        a and b are the sides of a cluster
        you can visualize square_plaquette_featuremap.T and labeled.T
        with plt.imshow()
        """
        grid = np.zeros(shape = (self.shape_x + 1, self.shape_y + 1))
        for (i, j) in self.energy.polygon_object.nodes_object.qbits.coords:
            grid[i, j] = 1
        square_plaquette_featuremap = signal.convolve2d(grid, self.square_filter)
        square_plaquette_featuremap[square_plaquette_featuremap<4]=0
        labeled, num = label(square_plaquette_featuremap, self.structure)

        clusters = []
        for i in range(1, num+1):
            a, b = self.indices[labeled.T==i].T
            a, b = list(a), list(b)
            a, b = [min(a)-1]+a, [min(b)-1]+b
            clusters.append([self.energy.polygon_object.nodes_object.order[i] for i in sorted(set(a))])
            clusters.append([self.energy.polygon_object.nodes_object.order[i] for i in sorted(set(b))])
        return clusters
    
    @staticmethod
    def unique_clusters(clusters):
        """
        merge clusters as far as possible
        """
        merged = []
        for idx, group in enumerate(MC_core.wrapper(clusters)):
            # if len is one, nothing can be merged
            if len(group)>1:
                # remove clusters in group which are already subset of other cluster in this group
                for x in group:
                    if (sum([all(i in g for i in x) for g in group]) > 1):
                        group.remove(x)
                j = 0
                # merge clusters in group till group is one large cluster
                while len(group) > 1:
                    """
                    take first element (a) of group of clusters, 
                    check if last element of a is in the next element (b)
                    if so merge them, if not check switch role of a and b,
                    otherwise keep a and take next element b,
                    if successful, remove a and b and append merged list ~ a+b
                    start search from first element again (which is now another
                    since a has been removed)
                    """
                    a, b = group[0], group[(j+1)%len(group)]
                    if a[-1] in b:
                        merged_ab = a+b[b.index(a[-1])+1:]
                    elif b[-1] in a:
                        merged_ab = b+a[a.index(b[-1])+1:]
                    else:
                        j += 1
                        continue
                    group.append(merged_ab)
                    group.pop(0)
                    group.pop((j)%len(group))
                    j=0
            merged.append(group[0])
        return merged
    
    def update_nodes_order(self, new_order):
        for idx, node in enumerate(new_order):
            self.energy.polygon_object.nodes_object.nodes[node].coord = idx
        self.energy.polygon_object.nodes_object.place_qbits_in_lines()
        
    def shuffle_clusters(self, merged_clusters):
            nodes_in_clusters = [i for x in merged_clusters for i in x]
            all_clusters = merged_clusters + [[i]
                                              for i in 
                                              list(set(self.energy.polygon_object.nodes_object.qbits.graph.nodes)
                                                   -set(nodes_in_clusters))]
            np.random.shuffle(all_clusters)
            new_order = [i for x in all_clusters for i in x]
            return new_order
        
    def greedy_clusters_swap(self, number_of_cluster_swaps: int=10):
        """
        swap clusters (swap them in a random way (np.random.shuffle) 
        for number_of_cluster_swaps times, take the orientation with the
        lowest energy. Since it is expensive to determine the clusters,
        we determine them once and then shuffle them several times 
        and update the nodes_order at the end by the minimum 
        order
        """
        order_and_energy = [(self.total_energy, self.energy.polygon_object.nodes_object.order)]    
        clusters = self.get_clusters()
        merged = MC_core.unique_clusters(clusters)
        for _ in range(number_of_cluster_swaps):
            new_order = self.shuffle_clusters(merged[:])
            self.update_nodes_order(new_order)
            new_energy = self.energy(self.energy.polygon_object.nodes_object.qbits)[0]
            order_and_energy.append((new_energy, new_order))

        self.update_nodes_order(min(order_and_energy)[1])
        
    def clusters_swap(self, number_of_cluster_swaps: int=10):
        """
        swap clusters and always accept this move
        """
        clusters = self.get_clusters()
        merged = MC_core.unique_clusters(clusters)
        new_order = self.shuffle_clusters(merged[:])
        self.update_nodes_order(new_order)
        
    def remove_short_lines_in_core_search(self):
        """
        ignore qbit in search if it is part of a node which is very short (length 2 or 3)
        length 1 would mean that only this qbit exists which cannot form any constraints
        """
        qbits_to_ignore = [qbit.qubit for qbit in self.energy.polygon_object.nodes_object.qbits if 2 in qbit.length_of_its_nodes or 3 in qbit.length_of_its_nodes]
        self.remove_ancillas(qbits_to_ignore)
        return qbits_to_ignore
    
import matplotlib.pyplot as plt
import imageio

def create_image_for_step(mc):
    mc.apply(1)
    fig, ax = plt.subplots(figsize=(15,15))
    ax = mc.energy.polygon_object.visualize(ax=ax, core_corner=mc.corner)

    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

def visualize_search_process(mc, number_of_steps):
    
    imageio.mimsave(f'gifs/{number_of_steps}.gif', [create_image_for_step(mc) for _ in range(number_of_steps)], fps=2)