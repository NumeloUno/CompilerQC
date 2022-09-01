import random
import numpy as np
from CompilerQC import *
import cProfile
import yaml
import CompilerQC

# for convolution in cluster search
from scipy import signal, misc
from scipy.ndimage.measurements import label

"""
number of repetition rate from Eq. (4.17)
Simulated Annealing and Boltzmann Machines
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
        self.number_of_plaquettes_weight = 0
        self.sparse_plaquette_density_weight = 0
        self.length_of_node_weight = 0
        self.random_qbit = False
        self.qbit_with_same_node = False

        # choose coords
        self.min_plaquette_density_in_softcore = 0.75
        self.corner = None
        self.possible_coords_changed = False

        self.finite_grid_size = False
        self.padding_finite_grid = 0
        self.envelop_shell_search = False
        self.shell_search = False
        self.radius = 0
        self.same_node_coords = False

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
        self.decay_rate_of_swap_probability = 1
        self.swap_only_core_qbits_in_line_swaps = False
        self.shell_time = 10

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

    @classmethod
    def init_from_yaml(cls, graph: Graph, path_to_config: str = "Default/default.yaml"):
        with open(paths.parameters_path / path_to_config) as f:
            mc_schedule = yaml.load(f, Loader=yaml.FullLoader)
        energy = CompilerQC.functions_for_benchmarking.init_energy(graph)
        mc = cls(energy)
        mc = CompilerQC.functions_for_benchmarking.update_mc(
            mc, mc_schedule, core=False
        )
        mc.n_moves = int(mc.n_moves * mc.repetition_rate)
        # initialize temperature
        initial_temperature = mc.current_temperature
        if initial_temperature == 0:
            initial_temperature = mc.initial_temperature()
        mc.T_0 = initial_temperature
        mc.current_temperature = initial_temperature
        return mc

    def reset(self, current_temperature, remove_ancillas: bool, keep_core: bool):
        """reset complete MC search"""
        if remove_ancillas:
            self.remove_ancillas(
                {
                    qbit.qubit: None
                    for qbit in self.energy.polygon_object.nodes_object.qbits
                    if qbit.ancilla == True
                }
            )
        if not keep_core:
            Qbits.remove_core(self.energy.polygon_object.nodes_object.qbits)
        else:
            Qbits.reinit_coords(
                self.energy.polygon_object.nodes_object.qbits, with_core=True
            )
        self.n_total_steps = 0
        self.total_energy, self.number_of_plaquettes = self.energy(
            self.energy.polygon_object.nodes_object.qbits
        )
        self.current_temperature = current_temperature

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
            self.energy.polygon_object.nodes_object.qbits.swap_qbits(
                qbit, coord_or_qbit
            )
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
            self.energy.polygon_object.nodes_object.qbits.swap_qbits(
                coord_or_qbit, qbit
            )
        # reset number of plaquettes
        self.number_of_plaquettes = self.current_number_of_plaquettes

    def select_move(self, qbit=None, target_coord=None):
        """
        move random qbit to random coord
        if coord is already occupied, swap qbits
        """
        # select coord
        if self.finite_grid_size:
            target_coord = self.coord_from_finite_grid()
        elif self.envelop_shell_search:  # coord in envelop of softcore
            target_coord = self.coord_from_softcore()
        elif self.shell_search:  # coord from neighbours of core
            target_coord = self.coord_around_core()

        # select qbit
        if self.random_qbit:
            qbit = self.energy.polygon_object.nodes_object.qbits.random_shell_qbit()
        elif self.number_of_plaquettes_weight:
            if random.random() < 1 - self.number_of_plaquettes_weight:
                qbit = self.energy.polygon_object.nodes_object.qbits.random_shell_qbit()
            else:
                number_of_plaquettes_weight = (
                    self.energy.penalty_for_low_number_of_plaquettes()
                )
                qbit = self.energy.polygon_object.nodes_object.qbits.random_weighted_shell_qbit(
                    weights=number_of_plaquettes_weight
                )
        elif self.sparse_plaquette_density_weight:
            if random.random() < 1 - self.sparse_plaquette_density_weight:
                qbit = self.energy.polygon_object.nodes_object.qbits.random_shell_qbit()
            else:
                sparse_plaquette_density_weight = (
                    self.energy.penalty_for_sparse_plaquette_density()
                )
                qbit = self.energy.polygon_object.nodes_object.qbits.random_weighted_shell_qbit(
                    weights=sparse_plaquette_density_weight
                )
        elif self.length_of_node_weight:
            if random.random() < 1 - self.length_of_node_weight:
                qbit = self.energy.polygon_object.nodes_object.qbits.random_shell_qbit()
            else:
                length_of_node_weight = [
                    np.multiply(*qbit.length_of_its_nodes)
                    for qbit in self.energy.polygon_object.nodes_object.qbits.shell_qbits
                ]
                qbit = self.energy.polygon_object.nodes_object.qbits.random_weighted_shell_qbit(
                    weights=length_of_node_weight
                )

        elif (
            self.qbit_with_same_node
        ):  # qbit with same node as target_coord in softcore has
            qbit = self.qbit_with_same_node_as_neighbour(target_coord)
        elif (
            self.same_node_coords
        ):  # random qbit has to be True, choose coord which has qbit with same node as random qbit
            # works only if self.possible_coords has already been created,
            # thus (envelop_)shell_search has to be true, in this case target coord will be overwritten
            qbit, possible_coords = self.coord_from_same_node(qbit)
            target_coord = random.choice(possible_coords)
        assert qbit is not None, "oh! there is no qbit selected"
        assert target_coord is not None, "oh! there is no targetcoord selected"
        # return
        if target_coord in self.energy.polygon_object.nodes_object.qbits.coords:
            target_qbit = self.energy.polygon_object.nodes_object.qbits.qbit_from_coord(
                target_coord
            )
            return qbit, target_qbit
        else:
            return qbit, target_coord

    def coord_from_infinite_grid(self):
        """
        generate random coord in grid
        sample from envelop rectengular of compiled graph
        for running time reasons: only every repetition_rate times
        """
        if (
            self.n_total_steps % (self.repetition_rate * self.shell_time) == 0
            or self.possible_coords_changed
        ):
            possible_coords = (
                self.energy.polygon_object.nodes_object.qbits.envelop_rect()
            )
            self.possible_coords = list(
                set(possible_coords)
                - set(self.energy.polygon_object.nodes_object.qbits.core_qbit_coords)
            )
        return random.choice(self.possible_coords)

    def coord_from_finite_grid(self):
        """
        set or create core corner in the first step (0),
        build coords around and from core corner,
        substract core_qbit_coords, when core_qbit_coords changed(due to line swap)
        change possible_coords
        """
        if self.n_total_steps == 0:
            if self.corner is None:
                self.corner = self.energy.polygon_object.core_corner
                # if no core has been set, use square around the center as start for shell search
                if self.corner is None:
                    x, y = self.energy.polygon_object.center_of_coords(
                        self.energy.polygon_object.nodes_object.qbits.coords
                    )
                    self.corner = (int(x - 0.5), int(x + 0.5)), (int(y - 0.5), int(y + 0.5))
                    
            (min_x, max_x), (min_y, max_y) = self.corner

            x, y = np.meshgrid(
            np.arange(min_x - self.padding_finite_grid, max_x + (self.padding_finite_grid + 1)),
            np.arange(min_y - self.padding_finite_grid, max_y + (self.padding_finite_grid + 1)),
            )

            self.finite_possible_coords = list(zip(x.flatten(), y.flatten()))
            
        if self.n_total_steps == 0 or self.possible_coords_changed:
            self.possible_coords = list(
            set(self.finite_possible_coords)
            - set(self.energy.polygon_object.nodes_object.qbits.core_qbit_coords)
            )
        return random.choice(self.possible_coords)

    def coords_in_softcore(self):
        (min_x, max_x), (min_y, max_y) = self.corner
        coords_in_softcore = [
            (i, j) for i in range(min_x, max_x + 1) for j in range(min_y, max_y + 1)
        ]

        coords_in_softcore = list(
            set(coords_in_softcore)
            - set(self.energy.polygon_object.nodes_object.qbits.core_qbit_coords)
        )
        return coords_in_softcore

    def coord_from_softcore(self):
        """
        softcore = core + plaquettes around it from shell qbits
        """
        if self.corner is None:
            self.corner = self.energy.polygon_object.core_corner
            # if no core has been set, use square around the center as start for shell search
            if self.corner is None:
                x, y = self.energy.polygon_object.center_of_coords(
                    self.energy.polygon_object.nodes_object.qbits.coords
                )
                self.corner = (int(x - 0.5), int(x + 0.5)), (int(y - 0.5), int(y + 0.5))
            self.possible_coords = self.coords_in_softcore()

        # check every few hundert steps if shell should be expanded
        if (
            self.n_total_steps % (self.repetition_rate * self.shell_time) == 0
            or self.possible_coords_changed
        ):
            # even if not enough plaquettes, the shape of the core could have changed
            # thus, the possible coords have to change too
            self.possible_coords = self.coords_in_softcore()
            if (
                self.plaquette_density_in_softcore()
                > self.min_plaquette_density_in_softcore
            ):
                # if enough plaquettes in this area, expand area
                (min_x, max_x), (min_y, max_y) = self.corner
                self.corner = (min_x - 1, max_x + 1), (min_y - 1, max_y + 1)
                self.possible_coords = self.coords_in_softcore()

        return random.choice(self.possible_coords)

    def coords_around_core(self):
        """
        designing a plaquette density function for this function is not trivial?
        """
        coords_around_core = []
        core_qbit_coords = (
            self.energy.polygon_object.nodes_object.qbits.core_qbit_coords
        )
        for coord in core_qbit_coords:
            coords_around_core.extend(
                Qbits.neighbour_coords_without_origin(coord, self.radius)
            )
        return list(set(coords_around_core) - set(core_qbit_coords))

    def coord_around_core(self):
        if (
            self.n_total_steps % (self.repetition_rate * self.shell_time) == 0
            or self.possible_coords_changed
        ):
            self.possible_coords = self.coords_around_core()
            if (
                self.plaquette_density_in_softcore()
                > self.min_plaquette_density_in_softcore
            ):
                self.radius += 1
                self.possible_coords = self.coords_around_core()

        return random.choice(self.possible_coords)

    def qbit_with_same_node_as_neighbour(self, target_coord):
        neighbour_qbits_in_envelop = list(
            map(
                self.energy.polygon_object.nodes_object.qbits.coord_to_qbit_dict.get,
                set(
                    self.possible_coords
                    + self.energy.polygon_object.nodes_object.qbits.core_qbit_coords
                ).intersection(
                    self.energy.polygon_object.nodes_object.qbits.neighbour_coords_without_origin(
                        target_coord
                    )
                ),
            )
        )
        neighbour_nodes_in_envelop = [
            node
            for qbit in list(filter(None, neighbour_qbits_in_envelop))
            for node in qbit.qubit
            if qbit is not None
        ]
        potential_nodes = [
            node
            for node in neighbour_nodes_in_envelop
            if neighbour_nodes_in_envelop.count(node) < 4
        ]
        qbits_to_propose = [
            qbit
            for qbit in self.energy.polygon_object.nodes_object.qbits.shell_qbits
            if set(qbit.qubit).intersection(potential_nodes)
        ]
        if qbits_to_propose == []:
            return self.energy.polygon_object.nodes_object.qbits.random_shell_qbit()
        else:
            return random.choice(qbits_to_propose)

    def plaquette_density_in_softcore(self):
        coords_in_and_around_core = (
            self.possible_coords
            + self.energy.polygon_object.nodes_object.qbits.core_qbit_coords
        )
        left = [(c[0] + 1, c[1]) for c in coords_in_and_around_core]
        left = [
            (c[0] - 1, c[1]) for c in list(set(left) - set(coords_in_and_around_core))
        ]
        top = [(c[0], c[1] + 1) for c in coords_in_and_around_core]
        top = [
            (c[0], c[1] - 1) for c in list(set(top) - set(coords_in_and_around_core))
        ]
        upper_left_edge = top + left
        coords_in_and_around_core_without_upper_left_edge = set(
            coords_in_and_around_core
        ) - set(upper_left_edge)
        plaquette_density = len(
            self.energy.polygon_object.nodes_object.qbits.found_plaquettes_around_coords(
                coords_in_and_around_core_without_upper_left_edge
            )
        ) / len(coords_in_and_around_core_without_upper_left_edge)
        return plaquette_density

    def coord_from_same_node(self, qbit_to_move):
        """
        return random coord from neighbour coords of qbits which
        are from the same node as qbit_to_move
        """
        """
        softcore = core + plaquettes around it from shell qbits
        """
        neighbour_coords = [
            coord
            for qbit in self.energy.polygon_object.nodes_object.qbits
            for coord in Qbits.neighbour_coords_without_origin(qbit.coord)
            if qbit.qubit[0] in qbit_to_move.qubit
            or qbit.qubit[1] in qbit_to_move.qubit
        ]
        possible_coords = list(set(self.possible_coords).intersection(neighbour_coords))
        if possible_coords == []:
            qbit_to_move = (
                self.energy.polygon_object.nodes_object.qbits.random_shell_qbit()
            )
            qbit_to_move, possible_coords = self.coord_from_same_node(qbit_to_move)

        return qbit_to_move, possible_coords

    @property
    def repetition_rate(self):
        """lenght of markov chain"""
        return (
            self.repetition_rate_factor
            * self.energy.polygon_object.nodes_object.qbits.graph.K
        )

    def qbits_of_coord(self, coord, xy):
        """
        the core is defined by the core qbits,
        but also by its corners. the corners form
        an envelop rectengular, with a side x and a side y,
        we can access each row (y) or column (x) by
        the coordinate coord.
        """
        corner = self.energy.polygon_object.core_corner
        if self.swap_only_core_qbits_in_line_swaps:
            return [
                qbit
                for qbit in self.energy.polygon_object.nodes_object.qbits
                if qbit.coord[xy] == coord
                and corner[(xy + 1) % 2][0]
                <= qbit.coord[(xy + 1) % 2]
                <= corner[(xy + 1) % 2][1]
                and qbit.core == True
            ]
        else:
            return [
                qbit
                for qbit in self.energy.polygon_object.nodes_object.qbits
                if qbit.coord[xy] == coord
                and corner[(xy + 1) % 2][0]
                <= qbit.coord[(xy + 1) % 2]
                <= corner[(xy + 1) % 2][1]
            ]

    #     def get_line_clusters(self, xy):
    #         """given the side of the core,
    #         this function will identify clusters
    #         """
    #         (min_x, max_x), (min_y, max_y) = self.energy.polygon_object.core_corner
    #         core_matrix = np.zeros((max_x + 1 - min_x, max_y + 1 - min_y))
    #         for coord in self.energy.polygon_object.nodes_object.qbits.core_qbit_coords:
    #             core_matrix[max_y - coord[1], coord[0] - min_x] = 1
    #         clusters = []
    #         x = 0
    #         while x < core_matrix.shape[xy]:
    #             cluster = [x]
    #             for i in range(1, core_matrix.shape[xy] - x):
    #                 if xy == 0:
    #                     if (core_matrix[:, x] == core_matrix[:, x + i]).all():
    #                         cluster.append(x + i)
    #                     else:
    #                         break
    #                 if xy == 1:
    #                     if (core_matrix[x, :] == core_matrix[x + i, :]).all():
    #                         cluster.append(x + i)
    #                     else:
    #                         break
    #             x = cluster[-1] + 1
    #             clusters.append(cluster)
    #         return [
    #             [node + self.energy.polygon_object.core_corner[xy][0] for node in cluster]
    #             for cluster in clusters
    #         ]
    def get_line_clusters(self, xy):
        """given the side of the core,
        this function will identify clusters
        note: not sure if this, or the commented function above is correct
        """
        (min_x, max_x), (min_y, max_y) = self.energy.polygon_object.core_corner
        core_matrix = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
        for coord in self.energy.polygon_object.nodes_object.qbits.core_qbit_coords:
            core_matrix[coord[0] - min_x, coord[1] - min_y] = 1
        clusters = []
        x = 0
        while x < core_matrix.shape[xy]:
            cluster = [x]
            for i in range(1, core_matrix.shape[xy] - x):
                if xy == 1:
                    if (core_matrix[:, x] == core_matrix[:, x + i]).all():
                        cluster.append(x + i)
                    else:
                        break
                if xy == 0:
                    if (core_matrix[x, :] == core_matrix[x + i, :]).all():
                        cluster.append(x + i)
                    else:
                        break
            x = cluster[-1] + 1
            clusters.append(cluster)
        return [
            [node + self.energy.polygon_object.core_corner[xy][0] for node in cluster]
            for cluster in clusters
        ]

    def select_lines_in_core(self, cluster_swap=True):
        """
        sample first the side of the core (x or y)
        than either swap two clutsters (with 50% prob) in core or
        swap two lines (with 50% prob) in a random cluster (cluster has to be larger then 1)
        return: new mapping:{old_line: new_line}, side of core and qbits involved [node: qbits]
        """
        corner = self.energy.polygon_object.core_corner
        xy = random.randint(0, 1)
        clusters = self.get_line_clusters(xy)
        if random.random() < 0.5 or len(clusters) == 1:
            valid_clusters = [cluster for cluster in clusters if len(cluster) > 1]
            if valid_clusters:
                random_cluster = random.choice(valid_clusters)
                i, j = random.sample(random_cluster, 2)
                new_line_mapping = {i: j, j: i}
            else:
                return None, None, None
        # cluster_swap
        else:
            old_nodes_order = [node for cluster in clusters for node in cluster]
            i, j = random.sample(range(len(clusters)), 2)
            clusters[i], clusters[j] = clusters[j], clusters[i]
            new_nodes_order = [node for cluster in clusters for node in cluster]
            new_line_mapping = {
                new: old
                for old, new in zip(old_nodes_order, new_nodes_order)
                if old != new
            }

        old_qbits = {old: self.qbits_of_coord(old, xy) for old in new_line_mapping}
        return new_line_mapping, xy, old_qbits

    def swap_lines_in_core(self, new_line_mapping, xy, old_qbits):
        """
        given a side of the core (x or y),
        assign new coord to node according to new_line_mapping
        """
        for old, new in new_line_mapping.items():
            for qbit in old_qbits[old]:
                coord = list(qbit.coord)
                coord[xy] = new
                qbit.coord = tuple(coord)

    def reverse_swap_lines_in_core(self, new_line_mapping, xy, old_qbits):
        old_qbits = {old: self.qbits_of_coord(old, xy) for old in new_line_mapping}

        reversed_line_mapping = {old: new for new, old in new_line_mapping.items()}
        self.swap_lines_in_core(reversed_line_mapping, xy, old_qbits)
        self.number_of_plaquettes = self.current_number_of_plaquettes

    def step(self):
        """
        move qbit or swap qbits and return energy of old and new configuration
        to expand this function for g(x) > 1, select_move() shouls use random.choices(k>1)
        """
        move = True
        # swap adjacent lines with swap_probability, if it is the first step, move qbits, dont swap lines 
        # to assure that possible coords and finite possible coords is set properly
        if random.random() < self.swap_probability and self.n_total_steps > 0:
            operation, move = "swap", False
            new_line_mapping, xy, old_qbits = self.select_lines_in_core()
            # if select_lines_in_core failed (no valid clusters e.g.), move qbits instead of swapping lines
            if new_line_mapping is None or self.swap_only_core_qbits_in_line_swaps:
                move = True
            else:
                self.lines_in_core_to_swap = new_line_mapping, xy, old_qbits
                qbits_to_move = [qbit for qbits in old_qbits.values() for qbit in qbits]
                current_energy, current_n_plaqs = self.energy(qbits_to_move)
                self.swap_lines_in_core(*self.lines_in_core_to_swap)
                new_energy, new_n_plaqs = self.energy(qbits_to_move)
                self.possible_coords_changed = True
                self.swap_probability *= self.decay_rate_of_swap_probability
                # if not inside a cluster is swapped, but two clusters are swapped
                # then swap_only_core_qbits_in_line_swaps has to be False, otherwise Non unique coords-error can occur
                # thats why in the if statement it is asked for swap_only_core_...
                
        # if not swapping, move a qbit
        if move:
            operation = "move"
            self.qbits_to_move = self.select_move()
            # even if the move gets rejected, self.possible_coords will be updated, after this update we can
            # set possible_coords_changed to False again, otherwise, there will be an update every step
            self.possible_coords_changed = False
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
            if operation == "move":
                self.reverse_move(*self.qbits_to_move)
            if operation == "swap":
                self.reverse_swap_lines_in_core(*self.lines_in_core_to_swap)
                self.possible_coords_changed = False

            if self.recording:
                self.record_rejected_moves.append(self.n_total_steps)

    def apply(
        self,
        n_steps,
    ):
        # self.prof.enable()
        for _ in range(n_steps):
            if (
                self.number_of_plaquettes
                == self.energy.polygon_object.nodes_object.qbits.graph.C
            ):
                return 0  # ? what is the best value np.nan, None, ?
            self.update_mean_and_variance()
            current_energy, new_energy, operation = self.step()
            self.metropolis(current_energy, new_energy, operation)
        #             assert self.number_of_plaquettes == len(
        #                 self.energy.polygon_object.nodes_object.qbits.found_plaqs()
        #             ) or self.number_of_plaquettes == len(
        #                 [
        #                     p
        #                     for p in self.energy.polygon_object.nodes_object.qbits.found_plaqs()
        #                     if len(p) == 4
        #                 ]
        #             ), "number of plaqs is wrong!"

        # self.prof.disable()
        return new_energy - current_energy

    def initial_temperature(self, chi_0: float = None, size_of_S=50):
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
            return -estimate_standard_deviation(self) / np.log(chi_0)

        energy_values = positive_transitions(self, size_of_S)
        E_start, E_plus = energy_values.T
        # to slightly accelerate the T_0_estimation process, we use Eq. 7 from the paper
        T_1 = -(E_plus - E_start).mean() / np.log(chi_0)
        if self.T_1:
            return T_1
        if self.T_0_estimation:
            return T_0_estimation(energy_values, chi_0=chi_0, T_1=T_1)
        else:
            print("No initial temperature has been choosen")

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
                self.current_temperature = self.T_0 * np.exp(
                    self.number_of_plaquettes
                    / (
                        self.energy.polygon_object.nodes_object.qbits.graph.C
                        - self.number_of_plaquettes
                        + 1e-2
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

    def update_qbits_from_dict(
        self, core_qbit_to_core_dict, assign_to_core: bool = True
    ):
        self.energy.polygon_object.nodes_object.qbits.update_qbits_from_dict(
            core_qbit_to_core_dict, assign_to_core=True
        )
        self.total_energy, self.number_of_plaquettes = self.energy(
            self.energy.polygon_object.nodes_object.qbits
        )

    def add_ancillas(self, ancillas, only_four_cycles=False, update_energy: bool=True):
        """
        ancillas are also conncected via 3cycles,
        this can be changed with only_four_cycles=True
        update_energy: used to prevent error when adding ancillas to graph (non unique coords!)
        """
        # check if ancillas is an empty dictionary
        if ancillas:
            self.energy.polygon_object.nodes_object.qbits.graph.update_ancillas(
                ancillas
            )
            new_polygons = self.energy.polygon_object.add_ancillas_to_polygons(
                ancillas, only_four_cycles=only_four_cycles
            )
            ancilla_qbits = (
                self.energy.polygon_object.nodes_object.qbits.add_ancillas_to_qbits(
                    ancillas, new_polygons
                )
            )
            self.energy.polygon_object.nodes_object.add_ancillas_to_nodes(ancillas)
            if update_energy:
                delta_energy, delta_number_of_plaquettes = self.energy(ancilla_qbits)
                self.total_energy += delta_energy
                self.number_of_plaquettes += delta_number_of_plaquettes

    def remove_ancillas(self, ancillas):
        # check if ancillas is an empty dictionary
        if ancillas:
            delta_energy, delta_number_of_plaquettes = self.energy(
                [
                    self.energy.polygon_object.nodes_object.qbits[ancilla]
                    for ancilla in ancillas
                ]
            )
            self.total_energy -= delta_energy
            self.number_of_plaquettes -= delta_number_of_plaquettes

            self.energy.polygon_object.remove_ancillas_from_polygons(ancillas)
            self.energy.polygon_object.nodes_object.remove_ancillas_from_nodes(ancillas)
            self.energy.polygon_object.nodes_object.qbits.remove_ancillas_from_qbits(
                ancillas
            )
            # remove ancillas from graph, cycles are not updatet!
            self.energy.polygon_object.nodes_object.qbits.graph.remove_ancillas(
                ancillas
            )


from copy import deepcopy
from CompilerQC import Qbits, Polygons, Energy


def positive_transitions(mc_object, size_of_S=50):
    """
    create states and for each state, it takes one bad neighbour (state with higher energy)
    return list: [[state, bad_neighbour_state]] of length size_of_S
    """
    E = []
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


def estimate_standard_deviation(mc_object, number_of_samples: int = 100):
    """return the standard deviation of the energies of number_of_samples
    randomly generated configurations"""
    mc = deepcopy(mc_object)
    energies = []
    for i in range(number_of_samples):
        Qbits.remove_core(mc.energy.polygon_object.nodes_object.qbits)
        energies.append(mc.energy(mc.energy.polygon_object.nodes_object.qbits)[0])
    return np.std(energies)


class MC_core(MC):

    square_filter = np.array([[1, 1], [1, 1]])
    structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    def __init__(
        self,
        energy_object: Energy,
        delta: float = 0.05,
        chi_0: float = 0.8,
        alpha: float = 0.95,
        repetition_rate_factor: float = 1,
        recording: bool = False,
        cluster_shuffling_probability: float = 0.0,
        ancilla_insertion_probability: float = 0.0,
        ancilla_deletion_probability: float = 0.0,
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
        self.shape_x = (
            self.shape_y
        ) = self.energy.polygon_object.nodes_object.qbits.graph.N
        self.indices = np.indices((self.shape_x + 2, self.shape_y + 2)).T[
            :,
            :,
        ]
        self.cluster_shuffling_probability = cluster_shuffling_probability

        # ancillas
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

    @classmethod
    def init_from_yaml(cls, graph: Graph, path_to_config: str = "Default/default.yaml"):
        with open(paths.parameters_path / path_to_config) as f:
            mc_schedule = yaml.load(f, Loader=yaml.FullLoader)
        energy_core = CompilerQC.functions_for_benchmarking.init_energy_core(graph)
        mc_core = cls(energy_core)
        mc_core = CompilerQC.functions_for_benchmarking.update_mc(
            mc_core, mc_schedule, core=True
        )
        mc_core.n_moves = int(mc_core.n_moves * mc_core.repetition_rate)
        # initialize temperature
        initial_temperature = mc_core.current_temperature
        if initial_temperature == 0:
            initial_temperature = mc_core.initial_temperature()
        mc_core.T_0 = initial_temperature
        mc_core.current_temperature = initial_temperature
        return mc_core

    def reset(self, current_temperature, remove_ancillas: bool):
        """reset complete MC search,
        first part sets qbit coords new, but keeps other qbits
        attributes"""
        if remove_ancillas:
            self.remove_ancillas(
                {
                    qbit.qubit: None
                    for qbit in self.energy.polygon_object.nodes_object.qbits
                    if qbit.ancilla == True
                }
            )
        self.energy.polygon_object.nodes_object = Nodes(
            self.energy.polygon_object.nodes_object.qbits
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
                self.add_ancillas(
                    ancillas, only_four_cycles=self.only_four_cycles_for_ancillas
                )

        if random.random() < self.ancilla_deletion_probability:
            ancillas_to_remove = (
                self.energy.polygon_object.nodes_object.propose_ancillas_to_remove()
            )
            if ancillas_to_remove != {}:
                self.remove_ancillas(ancillas_to_remove)

        if random.random() < self.cluster_shuffling_probability:
            self.clusters_swap()
            # self.greedy_clusters_swap(number_of_cluster_swaps=10)

        node_i, node_j = self.energy.polygon_object.nodes_object.nodes_to_swap()
        self.nodes_to_move = node_i, node_j
        qbits_to_move = self.energy.polygon_object.nodes_object.qbits_of_nodes(
            self.nodes_to_move
        )
        current_energy, current_n_plaqs = self.energy(qbits_to_move)
        self.energy.polygon_object.nodes_object.swap_nodes(*self.nodes_to_move)
        new_energy, new_n_plaqs = self.energy(qbits_to_move)
        self.current_number_of_plaquettes = self.number_of_plaquettes
        self.number_of_plaquettes += new_n_plaqs - current_n_plaqs
        self.n_total_steps += 1
        return current_energy, new_energy, "placeholder"

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

    def initial_temperature(self, chi_0: float = None, number_of_samples=25):
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
            self.reset(current_temperature=None, remove_ancillas=False)
        return -np.std(energies) / np.log(chi_0)

    @staticmethod
    def consolidate(sets):
        """from stackoverflow, used in unique_clusters()"""
        # http://rosettacode.org/wiki/Set_consolidation#Python:_Iterative
        setlist = [s for s in sets if s]
        for i, s1 in enumerate(setlist):
            if s1:
                for s2 in setlist[i + 1 :]:
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
        groupmap = {x: i for i, seq in enumerate(consolidated) for x in seq}
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
        grid = np.zeros(shape=(self.shape_x + 1, self.shape_y + 1))
        for (i, j) in self.energy.polygon_object.nodes_object.qbits.coords:
            grid[i, j] = 1
        square_plaquette_featuremap = signal.convolve2d(grid, self.square_filter)
        square_plaquette_featuremap[square_plaquette_featuremap < 4] = 0
        labeled, num = label(square_plaquette_featuremap, self.structure)

        clusters = []
        for i in range(1, num + 1):
            a, b = self.indices[labeled.T == i].T
            a, b = list(a), list(b)
            a, b = [min(a) - 1] + a, [min(b) - 1] + b
            clusters.append(
                [
                    self.energy.polygon_object.nodes_object.order[i]
                    for i in sorted(set(a))
                ]
            )
            clusters.append(
                [
                    self.energy.polygon_object.nodes_object.order[i]
                    for i in sorted(set(b))
                ]
            )
        return clusters

    @staticmethod
    def unique_clusters(clusters):
        """
        merge clusters as far as possible
        """
        merged = []
        for idx, group in enumerate(MC_core.wrapper(clusters)):
            # if len is one, nothing can be merged
            if len(group) > 1:
                # remove clusters in group which are already subset of other cluster in this group
                for x in group:
                    if sum([all(i in g for i in x) for g in group]) > 1:
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
                    a, b = group[0], group[(j + 1) % len(group)]
                    if a[-1] in b:
                        merged_ab = a + b[b.index(a[-1]) + 1 :]
                    elif b[-1] in a:
                        merged_ab = b + a[a.index(b[-1]) + 1 :]
                    else:
                        j += 1
                        continue
                    group.append(merged_ab)
                    group.pop(0)
                    group.pop((j) % len(group))
                    j = 0
            merged.append(group[0])
        return merged

    def update_nodes_order(self, new_order):
        for idx, node in enumerate(new_order):
            self.energy.polygon_object.nodes_object.nodes[node].coord = idx
        self.energy.polygon_object.nodes_object.place_qbits_in_lines()

    def shuffle_clusters(self, merged_clusters):
        nodes_in_clusters = [i for x in merged_clusters for i in x]
        all_clusters = merged_clusters + [
            [i]
            for i in list(
                set(self.energy.polygon_object.nodes_object.qbits.graph.nodes)
                - set(nodes_in_clusters)
            )
        ]
        np.random.shuffle(all_clusters)
        new_order = [i for x in all_clusters for i in x]
        return new_order

    def greedy_clusters_swap(self, number_of_cluster_swaps: int = 10):
        """
        swap clusters (swap them in a random way (np.random.shuffle)
        for number_of_cluster_swaps times, take the orientation with the
        lowest energy. Since it is expensive to determine the clusters,
        we determine them once and then shuffle them several times
        and update the nodes_order at the end by the minimum
        order
        """
        order_and_energy = [
            (self.total_energy, self.energy.polygon_object.nodes_object.order)
        ]
        clusters = self.get_clusters()
        merged = MC_core.unique_clusters(clusters)
        for _ in range(number_of_cluster_swaps):
            new_order = self.shuffle_clusters(merged[:])
            self.update_nodes_order(new_order)
            new_energy = self.energy(self.energy.polygon_object.nodes_object.qbits)[0]
            order_and_energy.append((new_energy, new_order))

        self.update_nodes_order(min(order_and_energy)[1])

    def clusters_swap(self, number_of_cluster_swaps: int = 10):
        """
        swap clusters and always accept this move
        """
        clusters = self.get_clusters()
        merged = MC_core.unique_clusters(clusters)
        new_order = self.shuffle_clusters(merged[:])
        self.update_nodes_order(new_order)
        self.total_energy, self.number_of_plaquettes = self.energy(
            self.energy.polygon_object.nodes_object.qbits
        )


import matplotlib.pyplot as plt
import imageio


def create_image_for_step(mc, envelop_rect):
    mc.apply(1)
    fig, ax = plt.subplots(figsize=(15, 15))
    if not hasattr(mc, 'corner'):
        mc.corner = None
    ax = mc.energy.polygon_object.visualize(ax=ax, core_corner=mc.corner, envelop_rect=envelop_rect)

    # Used to return the plot as an image rray
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def visualize_search_process(mc, name, number_of_images):
    """ fix envelop of image (otherwise gif will wiggle around)"""
    envelop_rect = mc.energy.polygon_object.nodes_object.qbits.envelop_rect()
    imageio.mimsave(
        paths.gifs / f"{name}_nImg_{number_of_images}.gif",
        [create_image_for_step(mc, envelop_rect) for _ in range(number_of_images)],
        fps=2,
    )
