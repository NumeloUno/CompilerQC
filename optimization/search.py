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
        temperatur_schedule: str = "1 / x",
        T_0: float=10.
    ):
        self.energy = energy_object
        self.temperatur_schedule = temperatur_schedule
        self.n_total_steps = 0
        self.n_good_moves = 0
        self.n_bad_moves = 0
        self.check_sum = 0
        self.T_0 = T_0


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
        self.current_qbits = deepcopy(self.energy.polygon_object.qbits)
        qbits_to_move = self.select_move()
        self.qbits_to_move = qbits_to_move
        current_energy = self.energy(qbits_to_move)
        self.apply_move(*qbits_to_move)
        new_energy = self.energy(qbits_to_move)
        return current_energy, new_energy
    
    def initial_temperature(self, chi_0=0.8, size_of_S=100):
        """
        estimate the initial Temperature T_0,
        size of S is the number of states -> bad states,
        should converge with increasing size of S
        """
        positive_transitions = postive_transitions(
            self.energy.polygon_object.qbits.graph, size_of_S)
        return T_0_estimation(positive_transitions, chi_0)
    
    @property
    def temperature(self):
        k = self.n_total_steps
        temperatur = lambda x: eval(str(self.temperatur_schedule))
        return self.T_0 * temperatur(k + 1)      

    def metropolis(self, current_energy, new_energy):
        self.n_total_steps += 1
        delta_energy = new_energy - current_energy
        if delta_energy < 0:
            self.n_good_moves += 1
            return
        acceptance_probability = np.exp(- delta_energy / self.temperature)
        if random.random() < acceptance_probability:
            self.n_bad_moves += 1
            return 
        else:
            self.check_sum += 1
            self.energy.polygon_object.qbits = self.current_qbits

    def apply(
        self,
        operation,
        n_steps,
    ):
        for i in range(n_steps):
            if self.energy.polygon_object.number_of_plaqs == self.energy.polygon_object.graph.C:
                return 0 # ? what is the best value np.nan, None, ?
            current_energy, new_energy = self.step(operation)
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

def postive_transitions(graph, size_of_S = 100, E = []):
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
            current_energy, new_energy = mc.step('')
            delta_energy = new_energy - current_energy
            if delta_energy > 0:
                mc_max = mc
                not_found = False
            else:
                mc_max = deepcopy(mc_min)
                
        # append energy of state and bad neighbour state
        E.append([mc_min.energy(mc_min.energy.polygon_object.qbits), mc_max.energy(mc_max.energy.polygon_object.qbits)])
    return np.array(E)
    
def chi(T, E):
    """ formula (5) from https://doi.org/10.1023/B:COAP.0000044187.23143.bd"""
    E_min, E_max = E.T
    return np.exp(- E_max / T).sum() / np.exp(- E_min / T).sum()

def new_T(chi_0, T, E, p=0.5):
    """ formula (6) from https://doi.org/10.1023/B:COAP.0000044187.23143.bd"""
    return T * (np.log(chi(T, E)) / np.log(chi_0)) ** 1 / p

def T_0_estimation(E, chi_0=0.8, T_1=1000, epsilon=0.0001):
    """Computing the temperature of simulated annealing"""
    difference = epsilon + 1
    while difference > epsilon:
        T_n = new_T(chi_0, T_1, E)
        difference = np.abs(chi(T_n,E) - chi(T_1, E))
        T_1 = T_n
    return T_1


            
       