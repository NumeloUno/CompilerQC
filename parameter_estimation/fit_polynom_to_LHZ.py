import numpy as np
from CompilerQC import Polygons
from scipy.special import binom 
import itertools
import os.path
homedir=os.path.expanduser("~/UniInnsbruck/")
pathset=os.path.join(homedir,"CompilerQC/parameter_estimation/parameters")

# this function calculates the energy of LHZ analytically
def scope_of_shapes(x, y, n):
    """
    x, y are the scopes of possible rectengulars you can draw in LHZ
    N=4: xy 11
    N=5: xy 11, 12, 21
    ...

    calculate the scope of the rectengular, the "triangular" and the "cross"
    maybe I will add a sketch 
    """
    n = n-2
    square = 2 * x + 2 * y
    triangular = (
      n - (y - 1)
    + n - (x - 1)
    + np.sqrt(y ** 2 + (n - (y - 1) - x) ** 2)
    + np.sqrt(x ** 2 + (n - (x - 1) - y) ** 2))
    cross = (
      n - (y - 1) - x
    + n - (x - 1) - y
    + np.sqrt(y ** 2 + (n - (y - 1)) ** 2)
    + np.sqrt(x ** 2 + (n - (x - 1)) ** 2))
    return sum([triangular, square, cross]) - 3 * 4

def total_energy_LHZ4(N_):
    """
    loop over all rectengulars one can draw in LHZ,
    calculate the scope_of_shapes() for each (xy) and sum it
    """
    summe = 0
    for N in range(4,N_ + 1):
        summe += (sum(map(lambda x: scope_of_shapes(*x),
                          [(i, j, n) for n in range (4, N + 1)
                           for i in range(1, n) for j in range(1, n)
                           if i + j <= n - 2])))
    return summe

def number_4plaqs(N):
    """
    number of 4plaqs in LHZ
    """
    C = (N / 2 * (N - 1) - N + 1) - number_3plaqs(N)
    return int(C)


def total_energy_LHZ3(N):
    """
    calculates the scope (minus unit triangle scope) of 3er polygons in LHZ 
    and sums over it
    """
    polys = []
    for l in itertools.combinations([i for i in range(0, N)], 3):
         polys.append(Polygons.is_unit_triangle(
             Polygons.get_polygon_from_cycle(list(l))))
    return sum(polys)

def number_3plaqs(N):
    """
    number of 3plaqs in LHZ
    """
    C = N - 2
    return int(C)

def fit_polynom_to_LHZ4(Ns, max_N=20):
    """
    FIT order 5 may be seen by analytical formula,
    or by the fact that they have to scale with
    the number of 4plaqs in LHZ
    """
    Ns_for_fit = np.array([i for i in range(4,max_N)])
    energy_LHZ4_for_fit = [total_energy_LHZ4(N) for N in Ns_for_fit]
    poly_LHZ4 = np.poly1d(np.polyfit(np.array(Ns_for_fit), energy_LHZ4_for_fit, 5))
    return poly_LHZ4(Ns)

def scale_4plaqs(Ns, max_N=20):
    """
    the energy for the 4polygons scale with O(5), the energy for the 4plaqs scale with O(2)
    here is the factor for each N to give the 4plaqs their weight
    """
    poly_LHZ4 = fit_polynom_to_LHZ4(Ns, max_N)
    factors_poly_LHZ4_and_number_4plaqs = poly_LHZ4 / list(map(number_4plaqs, Ns))
    return factors_poly_LHZ4_and_number_4plaqs

def fit_polynom_to_LHZ3(Ns, max_N=20):
    """
    order 4 due to the fact that they have to scale with the
    number of 3plaqs in LHZ
    """
    Ns_for_fit = np.array([i for i in range(4,max_N)])
    energy_LHZ3_for_fit = [total_energy_LHZ3(N) for N in Ns_for_fit]
    poly_LHZ3 = np.poly1d(np.polyfit(Ns_for_fit, energy_LHZ3_for_fit, 4))
    return poly_LHZ3(Ns)

def scale_3plaqs(Ns, max_N=20):
    """
    the energy for the 3polygons scale with O(4), the energy for the 3plaqs scale with O(1)
    here is the factor for each N to give the 3plaqs their weight
    """
    poly_LHZ3 = fit_polynom_to_LHZ3(Ns, max_N)
    factors_poly_LHZ3_and_number_3plaqs = poly_LHZ3 / list(map(number_3plaqs, Ns))
    return factors_poly_LHZ3_and_number_3plaqs

def save_polynom_coeffs():
    """
    saves the coeffs of the fitted polynoms (fitted to the energy of the 
    not satisfied plaqs in LHZ)
    """
    poly_LHZ3 = fit_polynom_to_LHZ3(Ns)
    poly_LHZ4 = fit_polynom_to_LHZ4(Ns)
    np.save('polynomial_coeffs_LHZ4', poly_LHZ4)
    np.save('polynomial_coeffs_LHZ3', poly_LHZ3)
    
def save_scaling_factors(max_N=20):
    """
    saves the factors of scale_4plaqs and scale_3plaqs
    """
    Ns = np.array([i for i in range(4,max_N)])
    factors_poly_LHZ4_and_number_4plaqs = scale_4plaqs(Ns, max_N)
    factors_poly_LHZ3_and_number_3plaqs = scale_3plaqs(Ns, max_N)
    np.save(os.path.join(pathset,"scaling_factors_LHZ4.npy"), factors_poly_LHZ4_and_number_4plaqs)
    np.save(os.path.join(pathset,"scaling_factors_LHZ3.npy"), factors_poly_LHZ3_and_number_3plaqs)

if __name__=='__main__':
    save_scaling_factors()

