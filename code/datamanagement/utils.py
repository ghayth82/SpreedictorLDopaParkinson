import numpy as np

def __rotX(angle):
    return np.array(
            [[1,        0,                  0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]])
def __rotY(angle):
    return np.array(
            [[np.cos(angle), 0, np.sin(angle)],
            [0,              1,         0],
            [-np.sin(angle), 0, np.cos(angle)]])
def __rotZ(angle):
    return np.array(
            [[np.cos(angle), -np.sin(angle),    0],
            [np.sin(angle), np.cos(angle),      0],
            [0,                 0,              1]])

def __rotate(angles):
    return np.matmul(np.matmul(__rotX(angles[0]), __rotY(angles[1])), __rotZ(angles[2]))

def randomRotation(timeseries):
    """Rotate the timeseries.

    A timeseries matrix ``steps x coord`` will be
    randomly rotated by +/- 10 degree around each coordinate axis.
    """

    angles = np.random.uniform(-np.pi/2.*(1./9), np.pi/2.*(1./9), size = 3)

    return np.matmul(timeseries, __rotate(angles).T)

def batchRandomRotation(timeseries):
    """Apply :func:`randomRotation` to all samples."""

    for t in range(timeseries.shape[0]):
        timeseries[t] = randomRotation(timeseries[t])

    return timeseries


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    (Taken from http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    and validated by grl)

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def batchRandomRotationFull(timeseries):
    """Apply :func:`randomRotation` to all samples."""

    for t in range(timeseries.shape[0]):
        M = rand_rotation_matrix()
        timeseries[t] = np.matmul(timeseries[t], M)

    return timeseries