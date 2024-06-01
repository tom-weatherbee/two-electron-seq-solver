import numpy as np
from scipy.special import eval_legendre
from scipy.integrate import quad
from scipy.sparse.linalg import eigsh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import time

def calc_r(z, power, i, h):
    return z / (i + 1)**power / h**power

def calc_r_overlap(k, i, h, size):
    r1 = i % size + 1
    r2 = i // size + 1
    return min(r1, r2)**k / max(r1, r2)**(k + 1) / h

def legendre_Function(l, l1, k, x):
    return eval_legendre(l, x) * eval_legendre(k, x) * eval_legendre(l1, x)

def create_M(l, l1, h, size):
    overlap = np.zeros((size**2, size**2))
    C = ((2 * l + 1) * (2 * l1 + 1))**0.5 / 2
    for k in np.arange(l - l1, l + l1 + 1):
        I, _ = quad(lambda x : legendre_Function(l, l1, k, x), -1, 1)
        overlap += C * I * np.diag(np.array([calc_r_overlap(k, xi, h, size) for xi in range(size**2)]))
    return overlap

def eigensystem_r(r, size, z, excitation=-1, eigvals_only=False):
    h = r / size
    num_vec = r * 2

    I = np.identity(size)
    second = (np.diag(-2 * np.ones(size)) + np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)) / h**2
    r = np.diag(np.array([calc_r(z, 1, xi, h) for xi in range(size)]))
    r_overlap = create_M(0, 0, h, size)
    single = -second / 2 - r

    hamiltonian = np.kron(single, I) + np.kron(I, single) + r_overlap


    if eigvals_only:
        if excitation == -1:
            return eigsh(hamiltonian, k=num_vec, return_eigenvectors=False)
        return eigsh(hamiltonian, k=num_vec, return_eigenvectors=False)[excitation]
    else:
        if excitation == -1:
            return eigsh(hamiltonian, k=num_vec, which='SA')
        else:
            eigenvalues, eigenvectors = eigsh(hamiltonian, k=num_vec, which='SA')
            return (eigenvalues[0], eigenvectors[:, 0])

def eigensystem_l(r, size, z, excitation, vectors, l):
    h = r / size
    num_vec = r * 2

    overlaps = []
    for i in range(len(vectors)):
        overlaps.append(create_M(l, i, h, size))
    
    I = np.identity(size)
    second = (np.diag(-2 * np.ones(size)) + np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)) / h**2
    r = np.diag(np.array([calc_r(z, 1, xi, h) for xi in range(size)]))
    r2 = np.diag(np.array([calc_r(l * (l + 1) / 2, 2, xi, h) for xi in range(size)]))
    r_overlap = create_M(l, l, h, size)
    single = -second / 2 - r + r2

    hamiltonian = np.kron(single, I) + np.kron(I, single) + r_overlap
    phi = np.zeros(size**2).transpose()

    eigenvalues, eigenvectors = eigsh(hamiltonian, k=num_vec, which='SA')

    for i in range(num_vec):
        ev = eigenvectors[:, i]
        coeff = 0
        for j in range(len(overlaps)):
            coeff += ev.dot(np.matmul(overlaps[j], vectors[j][1]))
        coeff = coeff / (vectors[0][0] - eigenvalues[i])
        phi += ev * coeff


    gs = vectors[len(vectors) - 1][0] + vectors[0][1].dot(np.matmul(create_M(l, 0, h, size), phi))
    
    return (gs, phi)

def calculate_partial(r, size, z, excitation, l):
    vectors = []
    vectors.append(eigensystem_r(r, size, z, excitation))

    for i in range(l):
        vectors.append(eigensystem_l(r, size, z, excitation, vectors, i + 1))
    
    return vectors

def nevilleAlgo(i, j, spacing, startoffset, r, z, excitation, l):
    dp = nevilleAlgoHelper(i, j, spacing, startoffset, r, z, excitation, l, {}, True)
    for k in range(l + 1):
        print("l=" + str(k))
        sl = ""
        for m in range(j):
            s = ""
            for n in range(m, j):
                s += str(dp[(m, n)][k]) + ", "
            s = s.rstrip(" ,")
            sl += s + "\n"
        print(sl)
    return dp

def nevilleAlgoHelper(i, j, spacing, startoffset, r, z, excitation, l, dptable, init):
    if (i, j) in dptable:
        return dptable[(i, j)]
    if i == j:
        if (i, j) in dptable:
            return dptable[(i, j)]
        out = [k[0] for k in calculate_partial(r, (i + startoffset) * spacing, z, excitation, l)]
        dptable[(i, j)] = out
        if init:
            return dptable
        return out
    else:
        hi2 = (r / (i + startoffset) / spacing)**2
        hj2 = (r / (j + startoffset) / spacing)**2
        nv1 = [ hj2 * k for k in nevilleAlgoHelper(i, j - 1, spacing, startoffset, r, z, excitation, l, dptable, False) ]
        nv2 = [ hi2 * k for k in nevilleAlgoHelper(i + 1, j, spacing, startoffset, r, z, excitation, l, dptable, False) ]

        out = [ k / (hj2 - hi2) for k in [ a - b for a, b in zip(nv1, nv2) ] ]
        dptable[(i, j)] = out
        if init:
            return dptable
        return out

nevilleAlgo(0, 7, 5, 4, 9, 2, 0, 4)

def neville(startoffset, increment, num, r, z, excitation, l):
    dptable = {}
    for i in range(num):
        size = (startoffset + i) * increment
        dptable[(i, i)] = [x[0] for x in calculate_partial(r, size, z, excitation, l)]

    for i in range(num):
        for j_indexeer in range(num - i):
            j = num - 1 -  j_indexeer
            hi2 = (r / (i + startoffset) / increment)**2
            hj2 = (r / (j + startoffset) / increment)**2
            nv1 = [ hj2 * k for k in dptable[(i, j - 1)] ]
            nv2 = [ hi2 * k for k in dptable[(i + 1, j)] ]

            out = [ k / (hj2 - hi2) for k in [ a - b for a, b in zip(nv1, nv2) ] ]
            dptable[(i, j)] = out

    for k in range(l + 1):
        print("l=" + str(k))
        sl = ""
        for m in range(num):
            s = ""
            for n in range(m, num):
                s += str(dp[(m, n)][k]) + ", "
            s = s.rstrip(" ,")
            sl += s + "\n"
        print(sl)

