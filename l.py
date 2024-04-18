import numpy as np
from scipy.linalg import eigh
from scipy.special import eval_legendre
from scipy.integrate import quad

import matplotlib.pyplot as plt

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

    I = np.identity(size)
    second = (np.diag(-2 * np.ones(size)) + np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)) / h**2
    r = np.diag(np.array([calc_r(z, 1, xi, h) for xi in range(size)]))
    r_overlap = create_M(0, 0, h, size)
    single = -second / 2 - r

    hamiltonian = np.kron(single, I) + np.kron(I, single) + r_overlap

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

def eigensystem_l(r, size, z, excitation, vectors, l):
    h = r / size

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

    eigenvalues, eigenvectors = eigh(hamiltonian)

    for i in range(size**2):
        ev = eigenvectors[:, i]
        coeff = 0
        for j in range(len(overlaps)):
            coeff += ev.dot(np.matmul(overlaps[j], vectors[j][1]))
        coeff = coeff / (vectors[len(vectors) - 1][0] - eigenvalues[i])
        phi += ev * coeff

    gs = vectors[len(vectors) - 1][0] + vectors[0][1].dot(np.matmul(create_M(l, 0, h, size), phi))
    
    return (gs, phi)

def calculate_partial(r, size, z, excitation, l):
    vectors = []
    vectors.append(eigensystem_r(r, size, z, excitation))

    for i in range(l):
        vectors.append(eigensystem_l(r, size, z, excitation, vectors, i + 1))
    
    return vectors


def graph_contour(eigenvector, size, factor, name):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    new_size = int(size / factor)

    data = np.zeros((new_size, new_size))
    X = np.arange(1, new_size + 1)
    X, Y = np.meshgrid(X, X)

    for i in range(size**2):
        r1 = i // size
        r2 = i % size
        if r1 < new_size and r2 < new_size:
            data[r1, r2] = eigenvector[i]**2

    ax.contour(X, Y, data)
    ax.set_aspect('equal', adjustable='box')
    #plt.show()
    plt.savefig(str(name).zfill(3) + 'tes.png', transparent=True)
    plt.close()

def nevilleAlgo(i, j, spacing, startoffset, r, z, excitation, l):
    dp = nevilleAlgoHelper(i, j, spacing, startoffset, r, z, excitation, l, {}, True)
    for k in range(j):
        sl = ""
        for m in range(j):
            s = ""
            for n in range(i, j):
                s += str(dp[(n, j)][k]) + ", "
            s = s.rstrip(" ,")
            sl += s + "\n"
        print(sl)
    return dp

def nevilleAlgoHelper(i, j, spacing, startoffset, r, z, excitation, l, dptable, init):
    if (i, j) in dptable:
        return dptable[(i, j)]
    if i == j:
        out = [k[0] for k in calculate_partial(r, (i + startoffset) * spacing, z, excitation, l)]
        dptable[(i, j)] = out
        if init:
            return dptable
        return out
    else:
        hi2 = (r / (i + startoffset) / spacing)**2
        hj2 = (r / (j + startoffset) / spacing)**2
        nv1 = [ hj2 * k for k in nevilleAlgoHelper(i, j - 1, spacing, startoffset, r, z, excitation, l, dptable, False) ] #* hj2 - hi2
        nv2 = [ hi2 * k for k in nevilleAlgoHelper(i + 1, j, spacing, startoffset, r, z, excitation, l, dptable, False) ] #) / (hj2 - hi2)

        out = [ k / (hj2 - hi2) for k in [ a - b for a, b in zip(nv1, nv2) ] ]
        dptable[(i, j)] = out
        if init:
            return dptable
        return out
dp = nevilleAlgo(i=0, j=8, spacing=5, startoffset=4, r=9, z=2, excitation=0, l=1)

#print(dp)
