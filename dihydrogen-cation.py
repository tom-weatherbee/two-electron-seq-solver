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

def gs(r, size, z):
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
    kinetic = -(np.diag(-2 * np.ones(size)) + np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)) / h**2 / 2
    r = np.diag(np.array([calc_r(xi, h) for xi in range(size)]))
    r_super = np.diag(np.array([calc_r_super(xi, h, size) for xi in range(size**2)]))

    hamiltonian = np.kron(kinetic - z * r, I) + np.kron(I, z * r) + r_super

    t0 = time.time()
    gs = eigsh(hamiltonian_sparse, k=2, which = 'BE')
    gs_index = np.where(gs[0] == min(gs[0]))[0][0]
    gs_vec = gs[1][:,gs_index]
    return (min(gs[0]), gs_vec)

def nevilleAlgo(i, j, spacing, startoffset, r, z, excitation):
    return nevilleAlgoHelper(i, j, spacing, startoffset, r, z, excitation, {}, True)

def nevilleAlgoHelper(i, j, spacing, startoffset, r, z, excitation, dptable, init):
    if (i, j) in dptable:
        return dptable[(i, j)]
    if i == j:
        out = gs(r, (i + startoffset) * spacing, z)
        dptable[(i, j)] = out
        if init:
            return dptable
        return out
    else:
        hi2 = (r / (i + startoffset) / spacing)**2
        hj2 = (r / (j + startoffset) / spacing)**2
        out = (nevilleAlgoHelper(i, j - 1, spacing, startoffset, r, z, excitation, dptable, False) * hj2 - hi2
               * nevilleAlgoHelper(i + 1, j, spacing, startoffset, r, z, excitation, dptable, False)) / (hj2 - hi2)
        dptable[(i, j)] = out
        if init:
            return dptable
        return out

#dp = nevilleAlgo(i=0, j=9, spacing=10, startoffset=5, r=20, z=1, excitation=0)


#for i in range(9):
#    s = ""
#    for j in range(i, 9):
#        s += str(dp[(i, j)]) + ", "
#    print(s.rstrip(" ,"))

def graph_contour(eigenvector, size, excitation):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()

    data = np.zeros((size, size))
    X = np.arange(1, size + 1)
    X, Y = np.meshgrid(X, X)

    for i in range(size**2):
        r1 = i // size
        r2 = i % size
        data[r1, r2] = eigenvector[i]**2

    ax.contour(X, Y, data)
    ax.set_aspect('equal', adjustable='box')
    plt.show()
    #plt.savefig(str(excitation).zfill(3) + '_contour.png', transparent=True)
    #plt.close()

gs_val, gs_vec = gs(20, 100, 1)
print(gs_val)

graph_contour(gs_vec, 100, 0)
