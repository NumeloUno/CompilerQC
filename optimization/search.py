import random 
import numpy as np
from copy import deepcopy
from CompilerQC import Energy

class MC:
    def __init__(
        self,
        energy_object: Energy,
        temperatur_schedule: str = "1 / x",
    ):
        self.energy = energy_object
        self.temperatur_schedule = temperatur_schedule
        self.n_total_steps = 0
        self.n_good_moves = 0
        self.n_bad_moves = 0
        self.check_sum = 0

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
    
    # TODO: handle warning (overflow encountered)
    @property
    def temperature(self):
        k = self.n_total_steps
        temperatur = lambda x: eval(str(self.temperatur_schedule))
        return temperatur(k + 1)
    
    def initial_temperature(self):
        """
        estimate the initial Temperature T_0
        """
        pass

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
