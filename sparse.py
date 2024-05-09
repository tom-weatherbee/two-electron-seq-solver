import numpy as np
from scipy.linalg import eigh
from scipy.linalg import eig_banded
from scipy.linalg import eigh_tridiagonal
from scipy.linalg.lapack import dsbev
from scipy.linalg.lapack import ssytrd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from scipy.sparse import diags
from scipy.sparse import kron
from scipy.sparse.linalg import eigsh

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

import time

def calc_r(i, h):
    return 1 / (i + 1) / h

def calc_r_super(i, h, size):
    r1 = i % size + 1
    r2 = i // size + 1
    return 1 / max(r1, r2) / h

def calc_x(i, h):
    return 1 / (i + 1)**2 / h**2

def calc_x_super(i, h, size):
    x1 = i % size + 1
    x2 = i // size + 1
    return 1 / max(x1, x2)**2 / h**2
r = 9
size = 60
z = 2
h = r / size

diagonals = []
offsets = []
diagonals.append(np.ones(size) / h**2 - z * np.array([calc_r(xi, h) for xi in range(size)]))
offsets.append(0)
diagonals.append(-np.ones(size - 1) / 2 / h**2)
offsets.append(1)
diagonals.append(-np.ones(size - 1) / 2 / h**2)
offsets.append(-1)
single = diags(diagonals, offsets)

I = diags([np.ones(size)], [0])

hamiltonian_sparse = kron(I, single) + kron(single, I) + diags([np.array([calc_r_super(xi, h, size) for xi in range(size**2)])], [0])

I = np.identity(size)
second = (np.diag(-2 * np.ones(size)) + np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)) / h**2
r = np.diag(np.array([calc_r(xi, h) for xi in range(size)]))
r_super = np.diag(np.array([calc_r_super(xi, h, size) for xi in range(size**2)]))
single = -second / 2 - z * r

hamiltonian = np.kron(single, I) + np.kron(I, single) + r_super

t0 = time.time()
gs = eigsh(hamiltonian_sparse, k=2, which = 'BE')
print("Sparse: " + str(time.time() - t0) + ", " + str(min(gs[0])))
t0 = time.time()
eigh(hamiltonian, eigvals_only=True)
print("Dense: " + str(time.time() - t0))


#def eigensystem_r(r, size, z, excitation, eigvals_only=False):
#    h = r / size
#
#    I = np.identity(size)
#    second = (np.diag(-2 * np.ones(size)) + np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)) / h**2
#    r = np.diag(np.array([calc_r(xi, h) for xi in range(size)]))
#    r_super = np.diag(np.array([calc_r_super(xi, h, size) for xi in range(size**2)]))
#    single = -second / 2 - z * r
#
#    hamiltonian = np.kron(single, I) + np.kron(I, single) + r_super
#
#    if eigvals_only:
#        return np.partition(eigh(hamiltonian, eigvals_only=True), excitation)[excitation]
#    else:
#        eigenvalues, eigenvectors = eigh(hamiltonian)
#        gs = np.partition(eigenvalues, excitation)[excitation]
#        gs_index = np.where(eigenvalues == gs)[0][0]
#        gs_vec = eigenvectors[:,gs_index]
#        return (gs, gs_vec)
#
#def eigensystem_x(r, size, z, excitation, eigvals_only=False):
#    h = r / size
#
#    I = np.identity(size)
#    second = (np.diag(-2 * np.ones(size)) + np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)) / h**2
#    x2 = np.diag(np.array([calc_x(xi, h) for xi in range(size)]))
#    x2_super = np.diag(np.array([calc_x_super(xi, h, size) for xi in range(size**2)]))
#    single = (-np.matmul(x2, second) / 8) + (3 / 32 * np.matmul(x2, x2)) - (z * x2)
#
#    hamiltonian = np.kron(single, I) + np.kron(I, single) + x2_super
#
#    if eigvals_only:
#        return np.partition(eigh(hamiltonian, eigvals_only=True), excitation)[excitation]
#    else:
#        eigenvalues, eigenvectors = eigh(hamiltonian)
#        gs = np.partition(eigenvalues, excitation)[excitation]
#        gs_index = np.where(eigenvalues == gs)[0][0]
#        gs_vec = eigenvectors[:,gs_index]
#        return (gs, gs_vec)
