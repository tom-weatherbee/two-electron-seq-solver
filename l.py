import numpy as np
from scipy.linalg import eigh
from scipy.special import eval_legendre
from scipy.integrate import quad

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import time

import pylanczos

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

    print(hamiltonian)

    t0 = time.time()
    ev, _ = eigsh(dia_h)
    print(str(min(ev)) + ":" + str(time.time() - t0))
    t0 = time.time()
    print(dia_h)
    ev, _ = eigh(hamiltonian)
    print(str(min(ev)) + ":" + str(time.time() - t0))


    return None

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

eigensystem_r(9, 20, 2, 0)

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

def graph_contour(eigenvector, size, factor, name):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    new_size = int(size / factor)

    data = np.zeros((new_size, new_size))
    X = np.linspace(0, 9  * new_size / size, num=new_size)
    X, Y = np.meshgrid(X, X)

    for i in range(size**2):
        r1 = i // size
        r2 = i % size
        if r1 < new_size and r2 < new_size:
            data[r1, r2] = eigenvector[i]**2

    ax.contour(X, Y, data)
    ax.set_aspect('equal', adjustable='box')
    #plt.show()
    plt.savefig(str(name).zfill(3) + 'l.png', transparent=True)
    plt.close()

def graph_wireframe(eigenvector, size, factor, excitation):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    new_size = int(size / factor)

    data = np.zeros((new_size, new_size))
    X = np.linspace(0, 9 * new_size / size, num=new_size)
    X, Y = np.meshgrid(X, X)

    for i in range(size**2):
        r1 = i // size
        r2 = i % size

        if r1 < new_size and r2 < new_size:
            data[r1, r2] = eigenvector[i]**2

    ax.plot_wireframe(X, Y, data)
    ax.view_init(25, 45)
    #plt.show()
    plt.savefig(str(excitation).zfill(3) + '_wireframe.png', transparent=True)
    plt.close()

#es = eigensystem_r(9, 60, 2, 0)[1]
#graph_contour(es, 60, 3.8, 0)
#graph_wireframe(es, 60, 2.5, 0)

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




#def nevilleAlgoR(i, j, spacing, startoffset, r, z, excitation):
#    dp = nevilleAlgoHelperR(i, j, spacing, startoffset, r, z, excitation, {}, True)
#    for m in range(j):
#        s = ""
#        for n in range(m, j):
#            s += str(dp[(m, n)]) + ", "
#        print(s.rstrip(" ,"))
#
#def nevilleAlgoHelperR(i, j, spacing, startoffset, r, z, excitation, dptable, init):
#    if (i, j) in dptable:
#        return dptable[(i, j)]
#    if i == j:
#        out = eigensystem_r(r, (i + startoffset) * spacing, z, excitation, True)
#        dptable[(i, j)] = out
#        if init:
#            return dptable
#        return out
#    else:
#        hi2 = (r / (i + startoffset) / spacing)**2
#        hj2 = (r / (j + startoffset) / spacing)**2
#        out = (nevilleAlgoHelperR(i, j - 1, spacing, startoffset, r, z, excitation, dptable, False) * hj2 - hi2
#               * nevilleAlgoHelperR(i + 1, j, spacing, startoffset, r, z, excitation, dptable, False)) / (hj2 - hi2)
#        dptable[(i, j)] = out
#        if init:
#            return dptable
#        return out
#
#nevilleAlgoR(0, 4, 25, 2, 20, 1, 0)
#nevilleAlgoR(0, 5, 25, 2, 20, 2, 1)
