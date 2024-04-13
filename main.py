
import numpy as np
import sys
import argparse
import os
import math
np.set_printoptions(threshold=sys.maxsize, linewidth=os.get_terminal_size().columns)
parser = argparse.ArgumentParser()
parser.add_argument('-s')
parser.add_argument('-r')
parser.add_argument('-z')
args = parser.parse_args()

def calch(numGridPoints):
    return int(args.r)/numGridPoints

def calcGroundState(numGridPoints, z):
    h = calch(numGridPoints)

    I = np.identity(numGridPoints)

    second = np.zeros((numGridPoints, numGridPoints))

    r = np.zeros((numGridPoints, numGridPoints))
    rsmall = np.zeros((numGridPoints * numGridPoints, numGridPoints * numGridPoints))

    for i in range(numGridPoints):
        second[i, i] = -2
        r[i, i] = 1 / (i + 1) / h
        if i < numGridPoints - 1:
            second[i, i + 1] = 1
            second[i + 1, i] = 1

    second = second / h**2

    d_approx = np.kron(second, I) + np.kron(I, second)
    r_tot = z * (np.kron(r, I) + np.kron(I, r))

    for i in range(numGridPoints * numGridPoints):
        r1 = i % numGridPoints + 1
        r2 = i // numGridPoints + 1

        rsmall[i, i] = 1 / max(r1, r2) / h

    hamiltonian = -d_approx / 2 - r_tot + rsmall

    return sorted(np.linalg.eig(hamiltonian).eigenvalues)[0:2]

def calcGroundStateWithX(resolution, z):
    h = calch(resolution**2)**(1/2)

    I = np.identity(resolution)

    second = np.zeros((resolution, resolution))

    r_inv = np.zeros((resolution, resolution))
    rbig = np.zeros((resolution * resolution, resolution * resolution))

    for i in range(resolution):
        second[i, i] = -2
        r_inv[i, i] = 1 / ((i + 1)*h)**(2)
        if i < resolution - 1:
            second[i, i + 1] = 1
            second[i + 1, i] = 1

    second = second / h**2

    for i in range(resolution * resolution):
        x1 = i % resolution + 1
        x2 = i // resolution + 1

        rbig[i, i] = 1 / (max(x1, x2)*h)**(2)

    hamiltonian = -1/2*(np.matmul(1/4*np.kron(r_inv,I),np.kron(second, I)-3/4*np.kron(r_inv,I))+np.matmul(1/4*np.kron(I,r_inv),np.kron(I, second)-3/4*np.kron(I,r_inv))) - z*np.kron(r_inv,I) - z*np.kron(I,r_inv) + rbig

    return min(np.linalg.eig(hamiltonian).eigenvalues)


def nevilleAlgo(x, i, j):
    print(i,j)
    if i == j:
        return calcGroundState((i+4)*5)
    else:
        print(i,j)
        return ((x - calch((i+4)*5))*nevilleAlgo(x, i+1, j) - (x - calch((j+4)*5))*nevilleAlgo(x, i, j-1))/(calch((j+4)*5)-calch((i+4)*5))


resolution = int(args.s)
print(calcGroundState(resolution, int(args.z)))