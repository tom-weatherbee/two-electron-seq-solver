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

def eigensystem_r(r, size, z, excitation=-1, eigvals_only=False):
    h = r / size

    I = np.identity(size)
    second = (np.diag(-2 * np.ones(size)) + np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)) / h**2
    r = np.diag(np.array([calc_r(xi, h) for xi in range(size)]))
    r_super = np.diag(np.array([calc_r_super(xi, h, size) for xi in range(size**2)]))
    single = -second / 2 - z * r

    hamiltonian = np.kron(single, I) + np.kron(I, single) + r_super

    if eigvals_only:
        if excitation == -1:
            return eigh(hamiltonian, eigvals_only=True)
        return np.partition(eigh(hamiltonian, eigvals_only=True), excitation)[excitation]
    else:
        if excitation == -1:
            return eigh(hamiltonian)
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
    #plt.show()
    plt.savefig(str(excitation).zfill(3) + '_contour.png', transparent=True)
    plt.close()

es, evs = eigensystem_r(r=9, size=60, z=2)

for i in range(100):
    e = np.partition(es, i)[i]
    index = np.where(es == e)[0][0]
    vec = evs[:,index]

    graph_contour(vec, 60, i)

def calc_r_l(z, power, i, h):
    return z / (i + 1)**power / h**power

def calc_r_overlap(constant, k, i, h, size):
    r1 = i % size + 1
    r2 = i // size + 1
    return constant * min(r1, r2)**k / max(r1, r2)**(k + 1) / h

def eigensystem_l1(r, size, z, excitation):
    h = r / size
    
    gs0, phi0 = eigensystem_r(r, size, z, excitation)
    phi_overlap = np.matmul(np.diag(np.array([calc_r_overlap(1/3**0.5, 1, xi, h, size) for xi in range(size**2)])), phi0)

    I = np.identity(size)
    second = (np.diag(-2 * np.ones(size)) + np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)) / h**2
    r = np.diag(np.array([calc_r_l(z, 1, xi, h) for xi in range(size)])) - np.diag(np.array([calc_r_l(1, 2, xi, h) for xi in range(size)]))
    r_overlap = np.diag(np.array([calc_r_overlap(1, 0, xi, h, size) for xi in range(size**2)])) + np.diag(np.array([calc_r_overlap(2/5, 2, xi, h, size) for xi in range(size**2)]))
    single = -second / 2 - r

    hamiltonian = np.kron(single, I) + np.kron(I, single) + r_overlap
    phi1 = np.zeros(size**2).transpose()

    eigenvalues, eigenvectors = eigh(hamiltonian)

    print(min(eigenvalues))

    for i in range(size**2):
        phi1 += eigenvectors[:,i] * eigenvectors[:,i].dot(phi_overlap) / (gs0 - eigenvalues[i])

    gs1 = phi1.dot(np.matmul(hamiltonian, phi1))
    #phi1 = phi1 / np.linalg.norm(phi1)

    
    return (gs0, phi0, gs1, phi1)
    #graph(phi1, size)

def eigensystem_l2(r, size, z, excitation, eigvals_only=False):
    h = r / size
    
    gs0, phi0, gs1, phi1 = eigensystem_l1(r, size, z, excitation)
    phi0_overlap = np.matmul(np.diag(np.array([calc_r_overlap(1/5**0.5, 2, xi, h, size) for xi in range(size**2)])), phi0)
    phi1_overlap = np.matmul(np.diag(np.array([calc_r_overlap(2/15**0.5, 1, xi, h, size) for xi in range(size**2)]))
                             + np.diag(np.array([calc_r_overlap(.33197, 3, xi, h, size) for xi in range(size**2)])), phi1)

    I = np.identity(size)
    second = (np.diag(-2 * np.ones(size)) + np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)) / h**2
    r = np.diag(np.array([calc_r_l(z, 1, xi, h) for xi in range(size)])) - np.diag(np.array([calc_r_l(3, 2, xi, h) for xi in range(size)]))
    r_overlap = (np.diag(np.array([calc_r_overlap(1, 0, xi, h, size) for xi in range(size**2)]))
                 + np.diag(np.array([calc_r_overlap(2/7, 2, xi, h, size) for xi in range(size**2)]))
                 + np.diag(np.array([calc_r_overlap(2/7, 4, xi, h, size) for xi in range(size**2)])))
    single = -second / 2 - r

    hamiltonian = np.kron(single, I) + np.kron(I, single) + r_overlap
    phi2 = np.zeros(size**2).transpose()

    eigenvalues, eigenvectors = eigh(hamiltonian)

    for i in range(size**2):
        phi2 += eigenvectors[:,i] * (eigenvectors[:,i].dot(phi0_overlap) + eigenvectors[:,i].dot(phi1_overlap)) / (gs0 - eigenvalues[i])
    
    print(phi2.dot(np.matmul(hamiltonian, phi2)))
    #graph_contour(phi2, size, 3)

