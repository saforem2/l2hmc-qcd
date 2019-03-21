import numpy as np

GELLMANN_MATRICES = np.array([
    np.matrix([  # lambda_1
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ], dtype=np.complex),
    np.matrix([  # lambda_2
        [0, 1j, 0],
        [1j, 0, 0],
        [0,  0, 0],
    ], dtype=np.complex),
    np.matrix([  # lambda_3
        [+1, 0, 0],
        [0, -1, 0],
        [0,  0, 0],
    ], dtype=np.complex),
    np.matrix([  # lambda_4
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0],
    ], dtype=np.complex),
    np.matrix([  # lambda_5
        [0,   0, -1j],
        [0,   0,   0],
        [+1j, 0,   0],
    ], dtype=np.complex),
    np.matrix([  # lambda_6
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ], dtype=np.complex),
    np.matrix([  # lambda_7
        [0, 0,   0],
        [0, 0, -1j],
        [0, 1j,  0],
    ], dtype=np.complex),
    np.matrix([  # lambda_8
        [+1, 0, 0],
        [0, +1, 0],
        [0, 0, -2],
    ], dtype=np.complex) / np.sqrt(3),
])

PAULI_MATRICES = np.array([
    np.matrix([
        [0, 1],
        [1, 0],
    ], dtype=np.complex),
    np.matrix([
        [0, -1j],
        [+1j, 0],
    ], dtype=np.complex),
    np.matrix([
        [+1, 0],
        [0, -1],
    ], dtype=np.complex),
])






DIRAC_MATRICES = np.array([
    np.matrix([
        [+1, 0, 0, 0],
        [0, +1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
    ], dtype=np.complex),
    np.matrix([
        [0, 0, 0, +1],
        [0, 0, +1, 0],
        [0, -1, 0, 0],
        [-1, 0, 0, 0],
    ], dtype=np.complex),
    np.matrix([
        [0, 0, 0, -1j],
        [0, 0, +1j, 0],
        [0, +1j, 0, 0],
        [-1j, 0, 0, 0],
    ], dtype=np.complex),
    np.matrix([
        [0, 0, +1, 0],
        [0, 0, 0, -1],
        [-1, 0, 0, 0],
        [0, +1, 0, 0],
    ], dtype=np.complex)
])


CHIRAL_DIRAC_MATRICES = np.array([
    np.matrix([
        [0, 0, +1, 0],
        [0, 0, 0, +1],
        [+1, 0, 0, 0],
        [0, +1, 0, 0],
    ], dtype=np.complex),
    np.matrix([
        [0, 0, 0, +1j],
        [0, 0, +1j, 0],
        [0, -1j, 0, 0],
        [-1j, 0, 0, 0],
    ], dtype=np.complex),
    np.matrix([
        [0, 0, 0, -1j],
        [0, 0, +1j, 0],
        [0, +1j, 0, 0],
        [-1j, 0, 0, 0],
    ], dtype=np.complex),
    np.matrix([
        [0, 0, +1j, 0],
        [0, 0, 0, -1j],
        [-1j, 0, 0, 0],
        [0, 1j, 0, 0],
    ], dtype=np.complex),
])

GAMMA_5 = np.matrix([
    [0, 0, +1, 0],
    [0, 0, 0, +1],
    [+1, 0, 0, 0],
    [0, +1, 0, 0],
], dtype=np.complex)


CHIRAL_GAMMA_5 = np.matrix([
    [+1, 0, 0, 0],
    [0, +1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, -1],
], dtype=np.complex)

ETA = np.matrix([
    [-1, 0, 0, 0],
    [0, +1, 0, 0],
    [0, 0, +1, 0],
    [0, 0, 0, +1],
], dtype=np.int8)
