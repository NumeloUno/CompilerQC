import random
import numpy as np
from copy import deepcopy
from CompilerQC import Energy
import cProfile
            # number of repetition rate from Eq. (4.17) Simulated Annealing and Boltzmann Machines
            # size of neighbourhood approx bei K

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
    ):
        """
        repetition rate: how many moves with constant temperature
        delta: how large are the jumps in temperature change
        T_0: initial temperture, can be later set with initial_temperature()
        """
        #self.prof = cProfile.Profile()

        self.energy = energy_object
        self.n_total_steps = 0
        self.chi_0 = chi_0
        self.delta = delta
        self.temperature_kirkpatrick = False
        self.repetition_rate = repetition_rate
        if self.repetition_rate is None:
            self.repetition_rate = self.energy.polygon_object.qbits.graph.K
        # compute total energy once at the beginning
        self.total_energy, self.number_of_plaquettes = self.energy(self.energy.polygon_object.qbits)


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
            self.old_coord = qbit.coord
            qbit.coord = coord_or_qbit
        else:
            self.energy.polygon_object.qbits.swap_qbits(qbit, coord_or_qbit)
            self.old_coord = None

    def reverse_move(self, qbit, coord_or_qbit):
        """
        reverse move from apply_move()
        """
        if type(coord_or_qbit) == tuple:
            qbit.coord = self.old_coord
        else:
            assert self.old_coord == None, 'something went wrong'
            self.energy.polygon_object.qbits.swap_qbits(coord_or_qbit, qbit)

    def generate_coord(self):
        """
        generate random coord in grid
        sample from envelop rectengular of compiled graph
        for running time reasons: only every repetition_rate times
        """
        if self.n_total_steps % self.repetition_rate == 0:
            self.possible_coords = self.energy.polygon_object.envelop_rect()
        return random.choice(self.possible_coords)

    def step(self, operation: str=None):
        """
        move qbit or swap qbits and return energy of old and new configuration
        to expand this function for g(x) > 1, select_move() shouls use random.choices(k>1)
        """
        self.qbits_to_move = self.select_move()
        current_energy, current_n_plaqs = self.energy(self.qbits_to_move)
        self.apply_move(*self.qbits_to_move)
        new_energy, new_n_plaqs= self.energy(self.qbits_to_move)
        self.current_number_of_plaquettes = self.number_of_plaquettes
        self.number_of_plaquettes += (new_n_plaqs - current_n_plaqs)
        self.n_total_steps += 1
        return current_energy, new_energy

    @property
    def temperature(self):
        """
        call this function 
        after n_total_steps > 1,
        since welford needs it like this
        """
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
                    / 3
                )
                self.current_temperature = new_temperature
                
        return self.current_temperature
 
    def metropolis(self, current_energy, new_energy):
        temperature = self.temperature
        delta_energy = new_energy - current_energy
        if delta_energy < 0:
            self.total_energy += delta_energy
            return
        acceptance_probability = np.exp(-delta_energy / temperature)
        if random.random() < acceptance_probability:
            self.total_energy += delta_energy
            return
        else:
            self.reverse_move(*self.qbits_to_move)
            self.number_of_plaquettes = self.current_number_of_plaquettes

    def apply(
        self,
        operation,
        n_steps,
    ):
        
        #self.prof.enable()
        
        C = self.energy.polygon_object.graph.C
        for i in range(n_steps):
            if self.number_of_plaquettes == C:
                return 0  # ? what is the best value np.nan, None, ?
            current_energy, new_energy = self.step(operation)
            self.metropolis(current_energy, new_energy)
            
        #self.prof.disable()

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

