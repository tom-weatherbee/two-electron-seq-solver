import numpy as np
from scipy.linalg import eigh
from scipy.linalg import eig_banded
from scipy.linalg import eigh_tridiagonal
from scipy.linalg.lapack import dsbev
from scipy.linalg.lapack import ssytrd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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

def eigensystem_r(r, size, z, excitation, eigvals_only=False):
    h = r / size

    I = np.identity(size)
    second = (np.diag(-2 * np.ones(size)) + np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)) / h**2
    r = np.diag(np.array([calc_r(xi, h) for xi in range(size)]))
    r_super = np.diag(np.array([calc_r_super(xi, h, size) for xi in range(size**2)]))
    single = -second / 2 - z * r

    hamiltonian = np.kron(single, I) + np.kron(I, single) + r_super

    if eigvals_only:
        return np.partition(eigh(hamiltonian, eigvals_only=True), excitation)[excitation]
    else:
        eigenvalues, eigenvectors = eigh(hamiltonian)
        gs = np.partition(eigenvalues, excitation)[excitation]
        gs_index = np.where(eigenvalues == gs)[0][0]
        gs_vec = eigenvectors[:,gs_index]
        return (gs, gs_vec)

def eigensystem_x(r, size, z, excitation, eigvals_only=False):
    h = r / size

    I = np.identity(size)
    second = (np.diag(-2 * np.ones(size)) + np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)) / h**2
    x2 = np.diag(np.array([calc_x(xi, h) for xi in range(size)]))
    x2_super = np.diag(np.array([calc_x_super(xi, h, size) for xi in range(size**2)]))
    single = (-np.matmul(x2, second) / 8) + (3 / 32 * np.matmul(x2, x2)) - (z * x2)

    hamiltonian = np.kron(single, I) + np.kron(I, single) + x2_super

    if eigvals_only:
        return np.partition(eigh(hamiltonian, eigvals_only=True), excitation)[excitation]
    else:
        eigenvalues, eigenvectors = eigh(hamiltonian)
        gs = np.partition(eigenvalues, excitation)[excitation]
        gs_index = np.where(eigenvalues == gs)[0][0]
        gs_vec = eigenvectors[:,gs_index]
        return (gs, gs_vec)

def nevilleAlgo(i, j, spacing, startoffset, r, z, excitation):
    return nevilleAlgoHelper(i, j, spacing, startoffset, r, z, excitation, {}, True)

def nevilleAlgoHelper(i, j, spacing, startoffset, r, z, excitation, dptable, init):
    if (i, j) in dptable:
        return dptable[(i, j)]
    if i == j:
        out = eigensystem_r(r, (i + startoffset) * spacing, z, excitation, True)
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

#dp = nevilleAlgo(i=0, j=9, spacing=5, startoffset=4, r=9, z=2, excitation=0)

#for i in range(9):
#    s = ""
#    for j in range(i, 9):
#        s += str(dp[(i, j)]) + ", "
#    print(s.rstrip(" ,"))

def graph(r, size, z, excitation):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    gs, eigenvector = eigensystem_r(r, size, z, excitation)

    data = np.zeros((size, size))
    X = np.arange(1, size + 1)
    X, Y = np.meshgrid(X, X)

    for i in range(size**2):
        r1 = i // size
        r2 = i % size
        data[r1, r2] = eigenvector[i]**2

    ax.plot_wireframe(X, Y, data)
    ax.view_init(-150, 225)

    plt.show()

#graph(9, 20, 2, 1)

def calc_r_overlap(i, h, size):
    r1 = i % size + 1
    r2 = i // size + 1
    return (min(r1, r2) / max(r1, r2)**2 / h / 3**0.5)

def eigensystem_l(r, size, z, excitation, eigvals_only=False):
    h = r / size

    I = np.identity(size)
    second = (np.diag(-2 * np.ones(size)) + np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)) / h**2
    r = np.diag(np.array([calc_r(xi, h) for xi in range(size)]))
    r_super = np.diag(np.array([calc_r_overlap(xi, h, size) for xi in range(size**2)]))
    single = -second / 2 - z * r

    hamiltonian = np.kron(single, I) + np.kron(I, single) + r_super

    if eigvals_only:
        return np.partition(eigh(hamiltonian, eigvals_only=True), excitation)[excitation]
    else:
        eigenvalues, eigenvectors = eigh(hamiltonian)
        gs = np.partition(eigenvalues, excitation)[excitation]
        gs_index = np.where(eigenvalues == gs)[0][0]
        gs_vec = eigenvectors[:,gs_index]
        return (gs, gs_vec)



def overlap(radius, size, z, excitation):
    r_super = np.kron(np.diag(np.array([calc_r_overlap(xi, radius / size, size) for xi in range(size**2)])), np.identity(2))

    gsl, vl = eigensystem_l(radius, size, z, excitation)
    gsr, vr = eigensystem_r(radius, size, z, excitation)


    vl = np.kron(vl, np.array([1, 1]).transpose())
    vr = np.kron(vr, np.array([1, 0]).transpose())

    temp = np.matmul(r_super, vr)

    print(gsl)
    print(gsr)
    print(vl.dot(temp))

    return gsl + vl.dot(temp)


print(overlap(9, 20, 2, 0))



def nevilleAlgoTest(i, j, spacing, startoffset, r, z, excitation):
    return nevilleAlgoHelper(i, j, spacing, startoffset, r, z, excitation, {}, True)

def nevilleAlgoHelperTest(i, j, spacing, startoffset, r, z, excitation, dptable, init):
    if (i, j) in dptable:
        return dptable[(i, j)]
    if i == j:
        out = overlap(r, (i + startoffset) * spacing, z, excitation)
        dptable[(i, j)] = out
        if init:
            return dptable
        return out
    else:
        hi2 = (r / (i + startoffset) / spacing)**2
        hj2 = (r / (j + startoffset) / spacing)**2
        out = (nevilleAlgoHelperTest(i, j - 1, spacing, startoffset, r, z, excitation, dptable, False) * hj2 - hi2
               * nevilleAlgoHelperTest(i + 1, j, spacing, startoffset, r, z, excitation, dptable, False)) / (hj2 - hi2)
        dptable[(i, j)] = out
        if init:
            return dptable
        return out

#dp = nevilleAlgoTest(i=0, j=7, spacing=5, startoffset=4, r=9, z=2, excitation=0)
#for i in range(7):
#    s = ""
#    for j in range(i, 7):
#        s += str(dp[(i, j)]) + ", "
#    print(s.rstrip(" ,"))
