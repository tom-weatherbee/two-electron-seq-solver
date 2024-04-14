import numpy as np
from scipy.linalg import eigh

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

def gs_eigensystem_r(r, size, z, eigvals_only=False):
    h = r / size

    I = np.identity(size)
    second = (np.diag(-2 * np.ones(size)) + np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)) / h**2
    r = np.diag(np.array([calc_r(xi, h) for xi in range(size)]))
    r_super = np.diag(np.array([calc_r_super(xi, h, size) for xi in range(size**2)]))
    single = -second / 2 - z * r

    hamiltonian = np.kron(single, I) + np.kron(I, single) + r_super

    if eigvals_only:
        return min(eigh(hamiltonian, eigvals_only=True))
    else:
        eigenvalues, eigenvectors = eigh(hamiltonian)
        gs = min(eigenvalues)
        gs_index = np.where(eigensystem.eigenvalues == gs)[0][0]
        gs_vec = eigensystem.eigenvectors[:,gs_index]
        return (gs, gs_vec)


def gs_eigensystem_x(r, size, z, eigvals_only=False):
    h = r / size

    I = np.identity(size)
    second = (np.diag(-2 * np.ones(size)) + np.diag(np.ones(size - 1), k=1) + np.diag(np.ones(size - 1), k=-1)) / h**2
    x2 = np.diag(np.array([calc_x(xi, h) for xi in range(size)]))
    x2_super = np.diag(np.array([calc_x_super(xi, h, size) for xi in range(size**2)]))
    single = (-np.matmul(x2, second) / 8) + (3 / 32 * np.matmul(x2, x2)) - (z * x2)

    hamiltonian = np.kron(single, I) + np.kron(I, single) + x2_super

    if eigvals_only:
        return min(eigh(hamiltonian, eigvals_only=True))
    else:
        eigenvalues, eigenvectors = eigh(hamiltonian)
        gs = min(eigenvalues)
        gs_index = np.where(eigensystem.eigenvalues == gs)[0][0]
        gs_vec = eigensystem.eigenvectors[:,gs_index]
        return (gs, gs_vec)

def nevilleAlgo(i,j,z):
    if i == j:
        return gs_eigensystem_r(9, (i + 4) * 5, 2, True)
    else:
        hi2 = (9 / (i + 4) / 5)**2
        hj2 = (9 / (j + 4) / 5)**2
        return (nevilleAlgo(i, j - 1, z) * hj2 - hi2 * nevilleAlgo(i + 1, j, z)) / (hj2 - hi2)

print(nevilleAlgo(0, 4, 2))
