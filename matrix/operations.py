"""
operations.py contains, um, a bunch of functions
Vector functions: norm, dot, unit, scalar
Matrix functions: determinant, rref, rank, elementary row operations, fundamental subspaces,
addition, subtraction, multiplication, trace, inverse, adjoint, eigenvalues, eigenvectors, diagonalization, triangularization, svd, orthogonalization
"""

# Module imports
from __future__ import division		# Change integer / to real /
from fractions import Fraction
from api import call_api
from random import randrange
from error import Error


# Latexify matrices

def latexify(M):
    A = [r'\begin{bmatrix} ' + '\n']
    for i in xrange(len(M)):
        for j in xrange(len(M[0])):
            A.append(make_frac(M[i][j]))
            if j != len(M[0]) - 1:
                A.append(r' & ')
        if i != len(M) - 1:
            A.append(r' \\' + '\n')
    A.append('\n' + r' \end{bmatrix}')
    return ''.join(A)


# Helper function to create result

def result(ans, step, *steps):
    res = Matrix(ans) if isinstance(ans, list) else ans
    for s in steps:
        res.steps.extend(s.steps)
    res.steps.append(step % str(res))
    res.steps.append('done')
    return res

# Matrix class defining convenience matrix operations


class Matrix(object):
    def __init__(self, M, steps=None):
        self.M = M
        self.steps = steps or []

    def __getitem__(self, index):
        return self.M[index]

    def __setitem__(self, index, value):
        self.M[index] = value
        return value

    def __delitem__(self, index):
        del self.M[index]

    def __iter__(self):
        return iter(self.M)

    def __add__(self, other):
        A, B = self.M, other.M
        return result([[A[i][j] + B[i][j] for j in xrange(len(A[0]))]
                       for i in xrange(len(A))],
                      str(self) + ' + ' + str(other) + ' = %s', self, other)

    def __sub__(self, other):
        A, B = self.M, other.M
        return result([[A[i][j] - B[i][j] for j in xrange(len(A[0]))]
                       for i in xrange(len(A))],
                      str(self) + ' - ' + str(other) + ' = %s',
                      self, other)

    def __mul__(self, other):
        if isinstance(other, int):
            res = Matrix([[c * other for c in r] for r in self.M])
            res.steps = self.steps[:]
        else:
            A, B = self.M, other.M
            res = Matrix([[sum([A[i][k] * B[k][j]
                                for k in xrange(len(B))])
                           for j in xrange(len(B[0]))]
                          for i in xrange(len(A))])
            res.steps = self.steps + other.steps
        res.steps.append(str(self) + ' * ' + str(other) + ' = ' + str(res))
        res.steps.append('done')
        return res

    # Called when multiplying a non-Matrix object with a Matrix object
    def __rmul__(self, other):
        return result([[c * other for c in r] for r in self.M],
                      str(other) + r' \thinspace ' + str(self) + ' = ' + str(result),
                      self)

    def __truediv__(self, other):
        return result([[val / other for val in r] for r in self.M],
                      str(self) + ' / ' + str(other) + ' = %s',
                      self)

    def __rtruediv__(self, other):
        if len(self.M[0]) == 1 and len(self.M) == 1:
            val = self.M[0][0]
            result([[other / val]],
                   str(other) + ' / ' + str(self) + ' = ' + str(result),
                   self)
        else:
            raise Error("Wrong type of operand passed for division")

    def __pow__(self, x):
        def modexpt(A, e):
            if e == 1:
                return A
            elif e % 2 == 0:
                X = modexpt(A, e // 2)
                return X * X
            else:
                X = modexpt(A, e // 2)
                return A * (X * X)

        if len(self) != len(self[0]):
            raise Error("Not a Square Matrix")
        elif x < 0:
            return result(inv(modexpt(self, abs(x))),
                          str(self) + ('^{%s}' % x) + ' = %s',
                          self)
        elif x == 0:
            return result(identity(len(self)),
                          str(self) + '^0 = %s',
                          self)
        else:
            return result(modexpt(self, x),
                          str(self) + ('^{%s}' % x) + ' = %s',
                          self)

    def __str__(self):
        return latexify(self.M)

    def __repr__(self):
        return str(self.M)

    def __len__(self):
        return len(self.M)


# Additional Functions

def identity(n, val=1):
    return Matrix([[val if i == j else 0.0 for j in xrange(n)] for i in xrange(n)])


# HTML Fraction maker

def make_frac(num):
    n = Fraction(num).limit_denominator().numerator
    d = Fraction(num).limit_denominator().denominator
    if d == 1:
        return "%d" % n
    elif Fraction(num).limit_denominator(10000).denominator != d:
        return r"%.3f" % num
    else:
        return r"\frac{%d}{%d}" % (n, d)


# Dimnension checking

def chk(x, y):
    return not (x and y and x.isdigit() and y.isdigit()
                and min(int(x), int(y)) > 0 and max(int(x), int(y)) <= 10)


# Vector Functions

def norm(v):
    return result([[sum([v[i][0] ** 2 for i in xrange(len(v))]) ** 1 / 2]],
                  str(r'\left|\left|' + str(v) + r'\right|\right| = %s'), v)


def dot(v, w):
    return tr(trans(w) * v)


def unit(v):
    val = norm(v)
    return result([[v[i][0] / val] for i in xrange(len(v))],
                  r'\hat{' + str(v) + '} = %s', v)

def rows(M):
    return len(M)

def cols(M):
    return len(M[0])

# Matrix Functions

# Transpose

def trans(A):
    return result([[A[i][j] for i in xrange(len(A))] for j in xrange(len(A[0]))],
                  str(A) + '^T = %s',
                  A)


# Trace

def tr(A):
    return result([[sum([A[i][i] for i in xrange(len(A))])]],
                  'tr(' + str(A) + ') = %s',
                  A)


# Determinant

def deter(X):
    l = len(X)
    if l == 1:
        return X[0][0]
    return sum([(-1) ** i * X[i][0] * deter(minor(X, i + 1, 1)) for i in xrange(l)])


def minor(X, i, j):
    y = X[:]
    del y[i - 1]
    y = zip(*y)
    del y[j - 1]
    return zip(*y)


def det(M):
    return result([[deter(M)]],
                  'det \\left( ' + str(M) + ' \\right) = %s',
                  M)


# Check if a matrix is symmetric

def isSymmetric(A):
    m, n = len(A), len(A[0])
    return all(all(A[i][j] == A[m - i - 1][n - j - 1] if i != j else True for j in xrange(n))
               for i in xrange((m + 1) // 2))


# Inverse

def inv(A):
    if deter(A.M) == 0:
        raise Error("Matrix is NOT INVERTIBLE")
    res = solve(A, identity(len(A)))
    res.steps[-2] = "%s^{-1} = %s" % (A, res)
    return res


# Adjoint Matrix

def adj(A):
    return result([[((-1) ** (i + j)) * deter([row[:j] + row[j + 1:] for row in A[:i] + A[i + 1:]])
                    for i in xrange(len(A))]
                   for j in xrange(len(A[0]))],
                  r'adj\left( ' + str(A) + r' \right) = %s', A)


# Cramer's Rule

def cramer(A, b):
    d = deter(A.M)
    if d == 0:
        raise Error("Cramer rule can't be used since the system doesn't have a unique solution")
    return result([[deter([row[:i] + b[i] + row[i + 1:] for row in A]) / d] for i in xrange(len(A))], str(A) + r'^{-1} = %s', A)

# Elementary Row Operations


def row_swap(i, j):
    return r"R_{%d} \leftrightarrow R_{%d}" % (i, j)


def row_mult(i, k):
    return r"R_{%d} \mapsto %s R_{%d}" % (i, make_frac(k), i)


def row_sub(i, k, j):
    return r"R_{%d} \mapsto R_{%d} - %s R_{%d}" % (i, i, make_frac(k), j)

# Rank - Takes in an RREF matrix


def rank(A, res=False):
    i = j = 0
    cols = []
    while i < len(A) and j < len(A[0]):
        while j < len(A[0]) and A[i][j] == 0:
            j += 1
        if j == len(A[0]):
            break
        cols.append(j)
        i += 1
    return cols if res else i

# Spanning Set

# Find solution of RREF matrices NOTE: Buggy!!


def span(A, d):
    b = d.M
    if all(k == [0] for k in b):
        b = [[] for i in xrange(len(d))]

    cols = rank(A, True)
    free = [i for i in xrange(len(A[0])) if i not in cols]

    soln = len(b[0])
    freeVar = [0 for i in xrange(len(free))]
    totalZeroes = [0 for i in xrange(soln + len(free))]

    M = [b[i] + freeVar if i in cols else totalZeroes[:] for i in xrange(len(A[0]))]

    for i, j in enumerate(free):
        M[j][i + soln] = 1
        for k in xrange(len(cols)):
            if cols[k] > j:
                break
            M[cols[k]][i + soln] = -A[k][j]

    return Matrix(d.M, A.steps + d.steps) if len(M[0]) == 0 else Matrix(M, A.steps + d.steps)


# RREF
# @param A: Matrix, Y: A Similar matrix
# @return Matrix(M: rref, soln: elementary matrix)


def rref(A, Y=Matrix([])):
    M = A.M[:]				# Makes a copy of matrix so it doesn't change the original values
    req = True
    if Y.M == []:
        req = False
        Y.M = [[0.0] for i in xrange(len(M))]
    X = Matrix(Y.M[:], Y.steps)
    ans = [M[:]]
    steps = []
    soln = [X[:]]
    lead = 0
    rowCount = len(M)
    columnCount = len(M[0])

    def calc():
        steps.append('')
        res = Matrix(ans[-1])
        res.soln = Matrix(soln[-1])
        res.steps = A.steps + X.steps + \
                    [('detail', step, latexify(P) + r' \quad \quad ' + (latexify(sol) if req else ''))
                     for P, step, sol in zip(ans, steps, soln)]
        return res

    for r in xrange(rowCount):
        if lead >= columnCount:
            return calc()
        i = r

        while M[i][lead] == 0:		# To go to the next non-zero pivot
            i += 1
            if i == rowCount:
                i = r
                lead += 1
                if columnCount == lead:   	# Return the matrix if all values left are zeroes
                    return calc()
        M[i], M[r] = M[r], M[i]			# Swap the row with non-zero pivot and current row

        if Y:
            X[i], X[r] = X[r], X[i]

        if i != r:
            ans.append(M[:])
            soln.append(X[:])
            steps.append(row_swap(r + 1, i + 1))

        pivot = M[r][lead]
        if pivot != 1:
            M[r] = [mrx / pivot for mrx in M[r]]

        if Y and (pivot != 1):
            X[r] = [val / pivot for val in X[r]]

        if pivot != 1:
            ans.append(M[:])
            soln.append(X[:])
            steps.append(row_mult(r + 1, 1 / pivot))

        for i in xrange(rowCount):
            if (i != r) and (M[i][lead] != 0):
                lv = M[i][lead]		# R_i -> R_i - R_ij * R_r
                M[i] = [iv - lv * rv for rv, iv in zip(M[r], M[i])]

                if Y:
                    X[i] = [iv - lv * rv for rv, iv in zip(X[r], X[i])]
                ans.append(M[:])
                soln.append(X[:])
                steps.append(row_sub(i + 1, lv, r + 1))
        lead += 1			# Each iteration gives one leading one
    return calc()

# Solve system of Linear Equations
# @param A: Linear Equations, x: Corresponding solutions

def solve(A, x):
    res = rref(A, Y=x)
    i = 1
    # No non-zero elements in the (n - i + 1)th row of RREF
    while all(x == 0 for x in res[-i]):
        if any(x != 0 for x in res.soln[-i]):		# If corresponding b_i is non-zero
            raise Error("No Solution Exists")
        i += 1
    return result(span(res, res.soln),
                  r"\text{Solution of System }" + str(A) + r' \quad ' + str(x) + r' is \, %s')


# Fundamental Subspaces
# @param A: Matrix

def subspaces(A):
    R = rref(A)

    colSpace = col(A)
    rowSpace = row(A, R)
    nullSpace = null(A, R)
    lnullSpace = lnull(A)

    return [colSpace, rowSpace, nullSpace, lnullSpace]


# Column Space

def col(A):
    ans = [[] for _ in xrange(len(A))]
    cols = rank(A, True)
    for c in cols:
        for k in xrange(len(A)):  # Column space
            ans[k].append(A[k][c])
    return result(ans, r'Col \left( ' + str(A) + r' \right) = %s', A)


# Row Space

def row(A, R=None):
    ans = [[] for i in xrange(len(A[0]))]
    if R is None:
        R = rref(A)
    for i in xrange(rank(A)):
        for k in xrange(len(A[0])):
            ans[k].append(R[i][k])
    return result(ans, r'Row \left( ' + str(A) + r' \right) = %s', A)


# Null Space

def null(A, R=None):
    if R is None:
        R = rref(A)
    return span(R, Matrix([[0]] * len(R)))

def lnull(A):
    R = rref(trans(A))
    return span(R, Matrix([[0]] * len(R)))

# Eigenvalues

def eigenval(A):
    ans = call_api('eigenvalues of %s' % str(A))[0]
    return result([[float(k) for k in ans]], r'Eigenvalues of ' + str(A) + ' are = %s', A)


def remove_dup(a):
    res = []
    seen = set()
    for i in a:
        if i not in seen:
            res.append(i)
            seen.add(i)
    return res

# Eigenvectors


def eigenvec(A):
    ans = call_api('eigenvectors of %s' % repr(A), "Corresponding eigenvalues")
    res = trans(Matrix([list(eval(val)) for val in ans[0]], A.steps))
    res.eigenval = [float(val) for val in ans[1]]
    res.steps.append('Eigenvectors of ' + str(A) + ' = ' + str(res))
    return res

""" Old shitty method
def eigenvec(A, res=False):
    ans = [[] for i in xrange(len(A))]
    eigen = eigenval(A)[0]
    print "lambda: ", eigen
    lam = [(k, eigen.count(k)) for k in remove_dup(eigen)]
    l = []
    for k, a in lam:
        ans = map(list.__add__, ans, solve(sub(A, make_id(len(A), val=k)),
                                           [[0] for i in xrange(len(A))]))
        l += [k for i in xrange(a)]
    return ans, l if res else ans
"""


# Diagonalize a matrix
# @param A: Matrix
# @return Matrix(M: Eigenvectors, eigenval: Eigenvalues, D: Diagonalized matrix)

def diagonalize(A):
    V = eigenvec(A)
    l = V.eigenval 	# Corresponding Eigenvalues
    if len(l) != len(V[0]):
        raise Error("Given matrix can't be diagonalized")

    V.D = [[l[i] if i == j else 0 for j in xrange(len(l))] for i in xrange(len(l))]
    return V


# Obtain an orthogonal matrix from a basis set

def orthogonalset(A, normal=False):
    T = trans(A)
    k = 0

    while all(val == 0 for val in T[k]):
        k += 1

    def mag(v, w):
        return sum([v[i] * w[i] for i in xrange(len(v))])

    norms = [mag(T[k], T[k])]

    P = [T[k]]
    x = 1

    for i in xrange(k + 1, len(T)):
        w = T[i]
        cons = [mag(w, P[l]) / norms[l] for l in xrange(x)]
        v = [w[j] - sum([cons[y] * P[y][j] for y in xrange(x)]) for j in xrange(len(A))]

        if all(val == 0 for val in v):
            continue

        P.append(v)
        norms.append(mag(v, v))
        x += 1

    if normal:
        for r in xrange(len(P)):
            v = P[r]
            n = mag(v, v) ** 0.5
            P[r] = [v[y] / n for y in xrange(len(A))]

    return trans(result(P, r'orthogonal set is %s', A))

# Orthogonalize Matrix


def orthogonalize(M):
    return orthogonalset(M, True)

"""
    def mag(v):
        return sum([v[i] ** 2 for i in xrange(len(v))])
    P = []
    print "M: ", M
    for v in trans(orthogonalset(M)):
        n = mag(v) ** 0.5
        P.append([v[i] / n for i in xrange(len(v))])

    return trans(P)
"""


# Method of Least Squares

def leastSquares(X, y):
    P = trans(X)
    return inv(P * X) * (P * y)


# Generate Random Vector

def generateVector(n):
    return [[randrange(0, 10)] for _ in xrange(n)]


# Extend Basis for Rn

def extendBasis(A):
    P = A.M
    steps = []
    n = len(P[0])
    if n == 1:
        def makeE(i):
            v = [[0] for _ in xrange(len(P))]
            v[i] = [1]
            return v

        for i in xrange(len(P)):
            if P[i][0] == 0:
                v = makeE(i)
                P = map(list.__add__, P, v)
                steps.append('Extending Basis set %s using %s' %
                             (latexify(P), latexify(v)))

    for _ in xrange(len(P[0]), len(P)):
        v = generateVector(len(P))
        while True:
            try:
                if len(solve(P, v)[0]) != 1:
                    v = generateVector(len(P))
            except:
                P = map(list.__add__, P, v)
                steps.append('Extending Basis set %s using %s' %
                             (latexify(P), latexify(v)))
                break

    A.steps.extend(steps)
    return result(P, 'Extended basis set of ' + str(A) + ' = %s', A)


# Triangularize Matrix

def triangularize(A):
    if len(A) == 1:
        res = Matrix([[1]], A.steps)
        res.T = A
        return res

    P = eigenvec(A)
    l = P.eigenval

    B = extendBasis(Matrix([[v[0]] for v in P]))
    P = orthogonalize(B)

    T = trans(P) * (A * P)
    T = Matrix([v[1:] for v in T], T.steps)
    b = T[0]

    Q = triangularize(T[1:])
    T1 = Q.T

    NP = P * Matrix([[1] + [0 for _ in xrange(len(Q))]] + [[0] + v for v in Q])
    NP.T = [[l[0]] + ([b] * Q)[0]] + [[0] + v for v in T1]
    NP.steps.append(r'P^T \, A \, P = {0}^T \, {1} \, {0} = {2}'.format(NP, A, NP.T))
    NP.steps.append('done')

    return NP

# Orthogonally Diagonalize Matrix


def orthodiagonalize(M):
    if not isSymmetric(M):
        raise Exception("Not a symmetric matrix")
    P = orthogonalize(eigenvec(M))
    res = P * (trans(P) * (M * P))
    res.steps.append('Orthodiagonalizing {0} = {1}'.format(str(M), str(res)))

# Singular Value Decomposition


def svd(M):
    A = trans(M) if len(M) < len(M[0]) else M
    n = len(A[0])
    B = trans(A) * A
    V = eigenvec(B)
    eigen = V.eigenval
    sing = [eigen[i] ** 0.5 for i in xrange(n)]
    V = orthogonalize(V)

    U = trans(Matrix([trans(A * Matrix([[v[i]] for i in xrange(len(v))]) / s)[0]
                      for v, s in zip(trans(V), sing) if s]))

    nullA = solve(A, Matrix([[0] for _ in xrange(len(A))]))

    if any(val != [0] for val in nullA):
        orthoA = orthogonalize(nullA)
        V = result(map(list.__add__, V, orthoA), '%s', V, orthoA)

    lnullA = solve(trans(A), Matrix([[0] for i in xrange(len(A[0]))]))

    if any(val != [0] for val in lnullA):
        orthoA =  orthogonalize(lnullA)
        U = result(map(list.__add__, U, orthoA), '%s', U, orthoA)
    S = Matrix([[sing[i] if i == j else 0 for j in xrange(len(M[0]))] for i in xrange(len(M))])

    return (V, S, U) if len(A) < len(A[0]) else (U, S, V)



"""
Exporting functions that are available for the user
"""

# Map function names to the functions themselves

functions = {
    '__builtins__': __builtins__,
    'Matrix': Matrix,
    '__package__': __package__,
    'adj': adj,
    'col': col,
    'cols': cols,
    'cramer': cramer,
    'det': det,
    'diagonalize': diagonalize,
    'dot': dot,
    'eigenval': eigenval,
    'id': identity,
    'inv': inv,
    'lnull': lnull,
    'norm': norm,
    'null': null,
    'orthodiagonalize': orthodiagonalize,
    'orthogonalize': orthogonalize,
    'rank': rank,
    'row': row,
    'rows': rows,
    'rref': rref,
    'solve': solve,
    'subspaces': subspaces,
    'svd': svd,
    'tr': tr,
    'trans': trans,
    'triangularize': triangularize,
    'unit': unit
}

# Map function names to function and number of arguments

funcs = {
    '+': lambda A, B: A + B,
    '-': lambda A, B: A - B,
    '*': lambda A, B: A * B,
    '/': lambda A, k: A / k,
    '^': lambda A, x: A ^ x,
    'rref': (rref, 1),
    'det': (det, 1),
    'inv': (inv, 1),
    'id': (identity, 1),
    'solve': (solve, 2),
    'dot': (dot, 2),
    'norm': (norm, 1),
    'unit': (unit, 1),
    'trans': (trans, 1),
    'tr': (tr, 1),
    'adj': (adj, 1),
    'cramer': (cramer, 1),
    'rank': (rank, 1),
    'col': (col, 1),
    'row': (row, 1),
    'null': (null, 1),
    'lnull': (lnull, 1),
    '': (id, 1)
}
