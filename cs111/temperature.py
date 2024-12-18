import numpy as np
import scipy


#############################################################################
# Make the temperature matrix (discrete Laplacian operator) in 2 dimensions #
#############################################################################

def make_A(k):
    """Create the matrix of the discrete Laplacian operator in two dimensions on a k-by-k grid.
    Parameters: 
      k: number of grid points in each dimension.
    Outputs:
      A: the sparse k**2-by-k**2 matrix representing the finite difference approximation to Laplace's equation.
    """
    # First make a list with one triple (row, column, value) for each nonzero element of A
    triples = []
    for x in range(k):
        for y in range(k):

            # what row of the matrix is grid point (x,y)?
            row = x + k * y

            # the diagonal element in this row
            col = row
            triples.append((row, col, 4.0))
            # connect to grid neighbors in x dimension
            if x > 0:
                col = row - 1
                triples.append((row, col, -1.0))
            if x < k - 1:
                col = row + 1
                triples.append((row, col, -1.0))
            # connect to grid neighbors in y dimension
            if y > 0:
                col = row - k
                triples.append((row, col, -1.0))
            if y < k - 1:
                col = row + k
                triples.append((row, col, -1.0))

    # Finally convert the list of triples to a scipy sparse matrix
    ndim = k * k
    rownum = [t[0] for t in triples]
    colnum = [t[1] for t in triples]
    values = [t[2] for t in triples]
    A = scipy.sparse.csr_matrix((values, (rownum, colnum)), shape=(ndim, ndim))

    return A


# end of make_A

#############################################################################
# Define an exact solution that we will impose on the boundary              #
#############################################################################

def exact_solution(x, y):
    # return np.cos(2*np.pi*x) + np.sin(2*np.pi*y)
    return np.cos(x) + np.sin(y)


# end of exact_solution

#############################################################################
# Define an exact solution that we will impose on the boundary              #
#############################################################################

def S(x, y):
    return np.cos(x) + np.sin(y)


# end of S

#############################################################################
# Make the temperature matrix (discrete Laplacian operator) in 2 dimensions #
# This version imposes the boundary condition exactly at the boundary       #
# and does not assume that dx = dy = 1                                      #
#############################################################################

def make_linear_system(xmin, xmax, ymin, ymax, k):
    """Create the matrix of the discrete Laplacian operator in two dimensions on a k-by-k grid.
    Parameters:
      :param xmin: interval [xmin, xmax]
      :param xmax: interval [xmin, xmax]
      :param ymin: interval [ymin, ymax]
      :param ymax: interval [ymin, ymax]
      :param k   : number of grid points in each dimension.
    Outputs:
      A  : the sparse k**2-by-k**2 matrix representing the finite difference approximation to Laplace's equation.
      rhs: the right-hand side of the linear system

    """
    # Define the grid
    x = np.linspace(xmin, xmax, k)
    y = np.linspace(ymin, ymax, k)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    one_over_dx_square = 1 / dx / dx
    one_over_dy_square = 1 / dy / dy

    # Start with a vector of zeros
    ndim = k * k
    rhs = np.zeros(shape=ndim)

    # First make a list with one triple (row, column, value) for each nonzero element of A
    triples = []

    # Treat the interior points:
    for i in range(1, k - 1):
        for j in range(1, k - 1):
            # what row of the matrix is grid point (i, j)?
            row = i + k * j
            # fill out the row in the matrix:
            triples.append((row, row, 2.0 * one_over_dx_square + 2.0 * one_over_dy_square))
            triples.append((row, row - 1, -1.0 * one_over_dx_square))
            triples.append((row, row + 1, -1.0 * one_over_dx_square))
            triples.append((row, row - k, -1.0 * one_over_dy_square))
            triples.append((row, row + k, -1.0 * one_over_dy_square))
            rhs[row] = S(x[i], y[j])

    # Treat the boundary points:
    for i in range(k):
        j = 0  # left wall
        row = i + k * j
        triples.append((row, row, 1.0))
        rhs[row] = exact_solution(x[i], y[j])
        j = k - 1  # right wall
        row = i + k * j
        triples.append((row, row, 1.0))
        rhs[row] = exact_solution(x[i], y[j])

    for j in range(1, k - 1):  # Need to avoid 0 and k-1 because duplicate entries are summed up with csr_matrix
        i = 0  # bottom wall
        row = i + k * j
        triples.append((row, row, 1.0))
        rhs[row] = exact_solution(x[i], y[j])
        i = k - 1  # top wall
        row = i + k * j
        triples.append((row, row, 1.0))
        rhs[row] = exact_solution(x[i], y[j])

    # Finally convert the list of triples to a scipy sparse matrix
    ndim = k * k
    rownum = [t[0] for t in triples]
    colnum = [t[1] for t in triples]
    values = [t[2] for t in triples]
    A = scipy.sparse.csr_matrix((values, (rownum, colnum)), shape=(ndim, ndim))

    return A, rhs


# end of make__linear_system

#############################################################################
# Make the temperature matrix (discrete Laplacian operator) in 3 dimensions #
#############################################################################
def make_A_3D(k):
    """
    Create the matrix of the discrete Laplacian operator in three dimensions on a k-by-k-by-k grid.
    Parameters:
      k: number of grid points in each dimension.
    Outputs:
      A: the sparse k**3-by-k**3 matrix representing the finite difference approximation to Laplace's equation.
    """
    # First make a list with one triple (row, column, value) for each nonzero element of A.
    triples = []
    for x in range(k):
        for y in range(k):
            for z in range(k):

                # What row of the matrix is grid point (x,y,z)?
                row = x + k * y + (k ** 2) * z

                # The diagonal element in this row.
                col = row
                triples.append((row, col, 6.0))

                # Connect to grid neighbors in x dimension
                if x > 0:
                    col = row - 1
                    triples.append((row, col, -1.0))
                if x < k - 1:
                    col = row + 1
                    triples.append((row, col, -1.0))

                # Connect to grid neighbors in y dimension.
                if y > 0:
                    col = row - k
                    triples.append((row, col, -1.0))
                if y < k - 1:
                    col = row + k
                    triples.append((row, col, -1.0))

                # Connect to grid neighbors in z dimension.
                if z > 0:
                    col = row - k ** 2
                    triples.append((row, col, -1.0))
                if z < k - 1:
                    col = row + k ** 2
                    triples.append((row, col, -1.0))

    # Finally convert the list of triples to a scipy sparse matrix
    ndim = k ** 3
    rownum = [t[0] for t in triples]
    colnum = [t[1] for t in triples]
    values = [t[2] for t in triples]
    A = scipy.sparse.csr_matrix((values, (rownum, colnum)), shape=(ndim, ndim))

    return A


# end of make_A_3D


#############################################################################
# Make a 16-by-16 version of the temperature matrix for demos               #
#############################################################################

def make_A_small():
    """Make a small k=4 version of the temperature matrix, as a dense array"""
    A = make_A(4)
    return A.toarray()


# end of make_A_small


#############################################################################
# Make a right-hand side vector for the 2D Laplacian / temperature matrix   #
#############################################################################

def make_b(k, top=0, bottom=0, left=0, right=0):
    """Create the right-hand side for the temperature problem on a k-by-k grid.
    Parameters: 
      k: number of grid points in each dimension.
      top: list of k values for top boundary (optional, defaults to 0)
      bottom: list of k values for bottom boundary (optional, defaults to 0)
      left: list of k values for top boundary (optional, defaults to 0)
      right: list of k values for top boundary (optional, defaults to 0)
    Outputs:
      b: the k**2 element vector (as a numpy array) for the rhs of the Poisson equation with given boundary conditions
    """
    # Start with a vector of zeros
    ndim = k * k
    rhs = np.zeros(shape=ndim)

    # Fill in the four boundaries as appropriate
    rhs[0: k] += top
    rhs[ndim - k: ndim] += bottom
    rhs[0: ndim: k] += left
    rhs[k - 1: ndim: k] += right

    return rhs


# end of make_rhs


#############################################################################
# Make a size-16 version of the right-hand side for demos                   #
#############################################################################

def make_b_small():
    """Make a small k=4 version of the right-hand side vector"""
    return make_b(4, top=radiator(4))


# End of make_rhs_small


#############################################################################
# Make one wall with a radiator                                             #
#############################################################################

def radiator(k, width=.3, temperature=100., default=0):
    """
	Create one wall with a radiator
	Parameters:
	k: number of grid points in each dimension; length of the wall
	width: width of the radiator as a fraction of length of the wall
	temperature: temperature of the radiator
	default: default temperature for the rest of the elements along the wall
	Outputs:
	wall: the k element vector (as a numpy array) for the boundary conditions at the wall
	"""
    rad_start = int(k * (0.5 - width / 2))
    rad_end = int(k * (0.5 + width / 2))
    wall = np.zeros(k) + default
    wall[rad_start: rad_end] = temperature

    return wall
# End of radiator
