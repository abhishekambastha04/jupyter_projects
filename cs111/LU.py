import numpy as np
import numpy.linalg as npla

#############################################################################
# Factor A = L@U with no pivoting                                           #
#############################################################################

def LUfactorNoPiv(A):
    """Factor a square matrix, A == L @ U (no partial pivoting)
    Parameters: 
      A: the matrix.
    Outputs (in order):
      L: the lower triangular factor, same dimensions as A, with ones on the diagonal
      U: the upper triangular factor, same dimensions as A
    """
    
    # Check the input
    m, n = A.shape
    assert m == n, 'input matrix A must be square'
    
    # Make a copy of the matrix that we will transform into L and U
    LU = A.astype(np.float64).copy()
    
    # Eliminate each column in turn
    for piv_col in range(n):
            
        # Update the rest of the matrix
        pivot = LU[piv_col, piv_col]
        assert pivot != 0., "pivot is zero, can't continue"
        for row in range(piv_col + 1, n):
            multiplier = LU[row, piv_col] / pivot
            LU[row, piv_col] = multiplier
            LU[row, (piv_col+1):] -= multiplier * LU[piv_col, (piv_col+1):]
            
    # Separate L and U in the result
    U = np.triu(LU)
    L = LU - U + np.eye(n)
    
    return (L, U)


#############################################################################
# Solve a unit lower triangular system L@y = b by forward substitution      #
#############################################################################

def Lsolve(L, b, unit_diag=False):
    """Forward solve a unit lower triangular system Ly = b for y
    Parameters: 
      L: the matrix, must be square, lower triangular, with ones on the diagonal
      b: the right-hand side vector
      unit_diag: if True, assume the diagonal is all ones
    Output:
      y: the solution vector to L @ y == b
    """
    
    # Check the input
    m, n = L.shape
    assert m == n, "matrix L must be square"
    assert np.all(np.tril(L) == L), "matrix L must be lower triangular"
    if unit_diag:
        assert np.all(np.diag(L) == 1), "matrix L must have ones on the diagonal"
    
    # Make a copy of the rhs that we will transform into the solution
    assert b.ndim == 1, "right-hand side must be a 1-dimensional vector"
    assert b.shape[0] == n, "right-hand side must be same size as matrix"
    y = b.astype(np.float64).copy()
    
    # Forward solve
    for row in range(n):
        if not unit_diag:
            y[row] /= L[row, row]
        y[row+1:] -= y[row] * L[row+1:, row]
        
    return y


#############################################################################
# Solve a unit lower triangular system L@y = b by forward substitution      #
#############################################################################

def LsolveVanilla(L, b):
    """Forward solve a unit lower triangular system Ly = b for y
    Parameters:
      L: the matrix, must be square, lower triangular, with ones on the diagonal
      b: the right-hand side vector
    Output:
      y: the solution vector to L @ y == b
    """

    # Check the input
    m, n = L.shape
    assert m == n, "matrix L must be square"
    assert np.all(np.tril(L) == L), "matrix L must be lower triangular"

    # Make a copy of the rhs that we will transform into the solution
    assert b.ndim == 1, "right-hand side must be a 1-dimensional vector"
    assert b.shape[0] == n, "right-hand side must be same size as matrix"
    y = b.astype(np.float64).copy()
    # Note that y[row] = b[row] for all rows

    # Forward solve
    for row in range(n):
        for col in range(0, row):
            y[row] -= L[row, col] * y[col]  # Note that y[row] = b[row] for all rows
        y[row] /= L[row, row]
    return y


#############################################################################
# Solve an upper triangular system U@x = y by back substitution             #
#############################################################################

def UsolveVanilla(U, y):
    """Backward solve an upper triangular system Ux = y for x
    Parameters:
      U: the matrix, must be square, upper triangular, with nonzeros on the diagonal
      y: the right-hand side vector
    Output:
      x: the solution vector to U @ x == y
    """
    # Check the input
    m, n = U.shape
    assert m == n, "matrix must be square"
    assert np.all(np.triu(U) == U), "matrix U must be upper triangular"

    # Make a copy of the rhs that we will transform into the solution
    assert y.ndim == 1, "right-hand side must be a 1-dimensional vector"
    assert y.shape[0] == n, "right-hand side must be same size as matrix"
    x = y.astype(np.float64).copy()
    # Note that x[row] = y[row] for all rows

    # Back solve
    for row in reversed(range(n)):
        for col in range(row+1, n):
            x[row] -= U[row, col] * x[col]  # Note that x[row] = y[row] for all rows
        x[row] /= U[row, row]

    return x


#############################################################################
# Solve an upper triangular system U@x = y by back substitution             #
#############################################################################

def Usolve(U, y, unit_diag=False):
    """Backward solve an upper triangular system Ux = y for x
    Parameters: 
      U: the matrix, must be square, upper triangular, with nonzeros on the diagonal
      y: the right-hand side vector
      unit_diag: if True, assume the diagonal is all ones
    Output:
      x: the solution vector to U @ x == y
    """
    # Check the input
    m, n = U.shape
    assert m == n, "matrix must be square"
    assert np.all(np.triu(U) == U), "matrix U must be upper triangular"
    if unit_diag:
        assert np.all(np.diag(U) == 1), "matrix U must have ones on the diagonal"

    # Make a copy of the rhs that we will transform into the solution
    assert y.ndim == 1, "right-hand side must be a 1-dimensional vector"
    assert y.shape[0] == n, "right-hand side must be same size as matrix"
    x = y.astype(np.float64).copy()

    # Back solve
    for row in reversed(range(n)):
        if not unit_diag:
            x[row] /= U[row, row]
        x[:row] -= x[row] * U[:row, row]
        
    return x



#############################################################################
# Factor A = L@U with (optional) partial pivoting                           #
#############################################################################

def LUfactor(A, pivoting=True):
    """Factor a square matrix with partial pivoting, A[p,:] == L @ U
    Parameters: 
      A: the matrix.
      pivoting: if False, don't do partial pivoting
    Outputs (in order):
      L: the lower triangular factor, same dimensions as A, with ones on the diagonal
      U: the upper triangular factor, same dimensions as A
      p: the permutation vector that permutes the rows of A by partial pivoting
    """
    
    # Check the input
    m, n = A.shape
    assert m == n, 'input matrix A must be square'
    
    # Initialize p to be the identity permutation
    p = np.array(range(n))
    
    # Make a copy of the matrix that we will transform into L and U
    LU = A.astype(np.float64).copy()
    
    # Eliminate each column in turn
    for piv_col in range(n):
        
        # Choose the pivot row and swap it into place
        if pivoting:
            piv_row = piv_col + np.argmax(np.abs(LU[piv_col:, piv_col]))
            assert LU[piv_row, piv_col] != 0., "can't find nonzero pivot, matrix is singular"
            LU[[piv_col, piv_row], :] = LU[[piv_row, piv_col], :]   # swap the rows piv_col and piv_row
            p [[piv_col, piv_row]   ] = p [[piv_row, piv_col]   ]   # swap the list elements piv_col and piv_row
            
        # Update the rest of the matrix
        pivot = LU[piv_col, piv_col]
        assert pivot != 0., "pivot is zero, can't continue"
        for row in range(piv_col + 1, n):
            multiplier = LU[row, piv_col] / pivot
            LU[row, piv_col] = multiplier
            LU[row, (piv_col+1):] -= multiplier * LU[piv_col, (piv_col+1):]
            
    # Separate L and U in the result
    U = np.triu(LU)
    L = LU - U + np.eye(n)
    
    return L, U, p


#############################################################################
# Solve A@x = b by LU factorization with partial pivoting                   #
#############################################################################

def LUsolve(A, b, pivoting=True):
    """Solve a linear system Ax = b for x by LU factorization with partial pivoting.
    Parameters: 
      A: the matrix.
      b: the right-hand side
      pivoting: if False, don't do partial pivoting
    Outputs (in order):
      x: the computed solution
      rel_res: relative residual norm,
        norm(b - Ax) / norm(b)
    """
    
    # Check the input
    m, n = A.shape
    assert m == n, 'input matrix A must be square'
    
    # LU factorization
    L, U, p = LUfactor(A, pivoting=pivoting)
    
    # Forward and back substitution
    y = Lsolve(L, b[p], unit_diag=True)
    x = Usolve(U, y)
    
    # Residual norm
    rel_res = npla.norm(b - A@x) / npla.norm(b)
    
    return (x, rel_res)


