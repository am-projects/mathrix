#########
# func.py contains, um, a bunch of functions
# Vector functions: norm, dot, unit, scalar
# Matrix functions: determinant, rref, rank, elementary row operations, fundamental subspaces,
# addition, subtraction, multiplication, trace, inverse, adjoint, eigenvalues, eigenvectors, diagonalization,
# triangularization, svd, orthogonalization
#########


# Module imports
from __future__ import division		# Change integer / to real /
from fractions import Fraction
from api import call_api
from random import randrange

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
    return not (x and y and x.isdigit() and y.isdigit() and
                min(int(x), int(y)) > 0 and max(int(x), int(y)) <= 10)

# Vector Functions


def norm(v):
    return sum([v[i][0] ** 2 for i in xrange(len(v))]) ** 1 / 2


def dot(v, w):
    return tr(mult(trans(w), v))


def unit(v):
    val = norm(v)
    return [[v[i][0] / val] for i in xrange(len(v))]


def scalar(v, x):
    return [[x * val[0]] for val in v]

# Matrix Functions

# Create Identity Matrix


def make_id(m, val=1.0):
    return [[val if i == j else 0.0 for j in xrange(m)] for i in xrange(m)]

# Determinant


def deter(X):
    l = len(X)
    if l == 1:
        return X[0][0]
    return sum([(-1) ** i * X[i][0] * deter(minor(X, i + 1, 1)) for i in xrange(l)])


def minor(X, i, j):
    y = X[:]
    del(y[i - 1])
    y = zip(*y)
    del(y[j - 1])
    return zip(*y)

"""
def deter(M, i = 0, j = 0, k = {}, c = 0):
    if j == len(M[0]):
        return 0
    elif k.get(j):
        return deter(M, i, j + 1, k, c)
    elif i == (len(M) - 1):
        return M[i][j]
    else:
        u = {key:val for key,val in k.items()}
        u[j] = len(k)
        return (((-1) ** c) * M[i][j] * deter(M, i + 1, 0, u)) + deter(M, i, j + 1, k, c + 1)
"""


def det(M):
    return [[deter(M)]]

# Cramer's Rule


def cramer(A, b):
    d = deter(A)
    if not d:
        raise Exception("Cramer rule can't be used since the system doesn't have a unique solution")
    return [[deter([row[:i] + b[i] + row[i + 1:] for row in A]) / d] for i in xrange(len(A))]

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
        while j < len(A[0] and A[i][j] == 0):
            j += 1
        if j == len(A[0]):
            break
        cols.append(j)
        i += 1
    return cols if res else i

# Spanning Set

# Find solution of RREF matrices NOTE: Buggy!!


def span(A, d):
    b = d
    if all(k == [0] for k in d):
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

    if len(M[0]) == 0:
        return d
    else:
        return M

# RREF


def rref(A, result=False, Y=[]):
    M = A[:]				# Makes a copy of matrix so it doesn't change the original values
    if Y == []:
        Y = [[0.0] for i in xrange(len(M))]
    X = Y[:]
    if result:
        ans = [M[:]]
        steps = ['']
        soln = [X[:]]
    lead = 0
    rowCount = len(M)
    columnCount = len(M[0])

    for r in xrange(rowCount):
        if lead >= columnCount:
            return M if not result else (ans, steps, soln)
        i = r

        while M[i][lead] == 0:		# To go to the next non-zero pivot
            i += 1
            if i == rowCount:
                i = r
                lead += 1
                if columnCount == lead:   	# Return the matrix if all values left are zeroes
                    return M if not result else (ans, steps, soln)
        M[i], M[r] = M[r], M[i]			# Swap the row with non-zero pivot and current row

        if Y:
            X[i], X[r] = X[r], X[i]

        if result and (i != r):
            ans.append(M[:])
            soln.append(X[:])
            steps.append(row_swap(r + 1, i + 1))

        pivot = M[r][lead]
        if pivot != 1:
            M[r] = [mrx / pivot for mrx in M[r]]

        if Y and (pivot != 1):
            X[r] = [val / pivot for val in X[r]]

        if result and (pivot != 1):
            ans.append(M[:])
            soln.append(X[:])
            steps.append(row_mult(r + 1, 1 / pivot))

        for i in xrange(rowCount):
            if (i != r) and (M[i][lead] != 0):
                lv = M[i][lead]		# R_i -> R_i - R_ij * R_r
                M[i] = [iv - lv * rv for rv, iv in zip(M[r], M[i])]

                if Y:
                    X[i] = [iv - lv * rv for rv, iv in zip(X[r], X[i])]
                if result:
                    ans.append(M[:])
                    soln.append(X[:])
                    steps.append(row_sub(i + 1, lv, r + 1))
        lead += 1			# Each iteration gives one leading one
    return M if not result else (ans, steps, soln)

# Solve system of Linear Equations


def solve(A, x, result=False):
    ans, steps, soln = rref(A, result=True, Y=x)
    i = 1
    # No non-zero elements in the (n - i + 1)th row of RREF
    while all(x == 0 for x in ans[-1][-i]):
        if any(x != 0 for x in soln[-1][-i]):		# If corresponding b_i is non-zero
            raise Exception("No Solution Exists")
        i += 1
    return (ans, steps, soln) if result else span(ans[-1], soln[-1])

# Fundamental Subspaces


def subspaces(A):
    R = rref(A)

    # Subspaces returned in this sequence, [Col(A), Row(A), Null(A), Null(A^T)]

    ans = [[[] for i in xrange(len(A))], [[] for i in xrange(len(A[0]))]]

    cols = rank(R, True)

    for i, c in enumerate(cols):
        for k in xrange(len(A)):  # Column space
            ans[0][k].append(A[k][c])
        for k in xrange(len(A[0])):  # Row space
            ans[1][k].append(R[i][k])

    ans.append(span(R, [[0] for i in xrange(len(R))]))  # Null space

    ans.append(span(rref(trans(A)), [[0] for i in xrange(len(R[0]))]))  # Left Null space

    return ans


# Addition


def add(A, B):
    return [[A[i][j] + B[i][j] for j in xrange(len(A[0]))] for i in xrange(len(A))]

# Subtraction


def sub(A, B):
    return [[A[i][j] - B[i][j] for j in xrange(len(A[0]))] for i in xrange(len(A))]

# Multiplication


def mult(A, B):
    return [[sum([A[i][k] * B[k][j] for k in xrange(len(B))]) for j in xrange(len(B[0]))]
            for i in xrange(len(A))]

# Transpose


def trans(A):
    return [[A[i][j] for i in xrange(len(A))] for j in xrange(len(A[0]))]

# Trace


def tr(A):
    return [[sum([A[i][i] for i in xrange(len(A))])]]

# Check if a matrix is symmetric


def isSymmetric(A):
    m, n = len(A), len(A[0])
    return all(all(A[i][j] == A[m - i - 1][n - j - 1] if i != j else True for j in xrange(n))
               for i in xrange((m + 1) // 2))

# Inverse


def inv(A, result=False):
    if deter(A) == 0:
        raise Exception("Matrix is NOT INVERTIBLE")
    return solve(A, make_id(len(A)), result)

# Adjoint Matrix


def adj(A):
    return [[((-1) ** (i + j)) * deter([row[:j] + row[j + 1:] for row in (A[:i] + A[i + 1:])])
             for i in xrange(len(A))] for j in xrange(len(A[0]))]

# Eigenvalues


def eigenval(A):
    ans = call_api('eigenvalues of %s' % str(A))[0]
    return [[float(k) for k in ans]]


def remove_dup(a):
    res = []
    seen = set()
    for i in a:
        if i not in seen:
            res.append(i)
            seen.add(i)
    return res

# Eigenvectors


def eigenvec(A, res=False):
    ans = call_api('eigenvectors of %s' % str(A), "Corresponding eigenvalues" if res else "")
    ans[0] = trans([list(eval(val)) for val in ans[0]])

    if res:
        ans[1] = [float(val) for val in ans[1]]

    return ans if res else ans[0]

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


def diagonalize(A):
    V, l = eigenvec(A, True)

    if len(l) != len(V[0]):
        raise Exception("Given matrix can't be diagonalized")

    return V, [[l[i] if i == j else 0 for j in xrange(len(l))] for i in xrange(len(l))]

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
            norm = mag(v, v) ** 0.5
            P[r] = [v[y] / norm for y in xrange(len(A))]

    return trans(P)

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
    return mult(inv(mult(P, X)), mult(P, y))

# Generate Random Vector


def generateVector(n):
    return [[randrange(0, 10)] for i in xrange(n)]

# Extend Basis for Rn


def extendBasis(P):
    n = len(P[0])
    if n == 1:
        def makeE(i):
            v = [[0] for j in xrange(len(P))]
            v[i] = [1]
            return v

        for i in xrange(len(P)):
            if P[i][0] == 0:
                v = makeE(i)
                P = map(list.__add__, P, v)

    for k in xrange(len(P[0]), len(P)):
        v = generateVector(len(P))
        while True:
            try:
                len(solve(P, v)[0]) != 1
                v = generateVector(len(P))
            except Exception:
                P = map(list.__add__, P, v)
                break

    return P

# Triangularize Matrix


def triangularize(A):
    if (len(A) == 1):
        return [[1]], A

    P, l = eigenvec(A, True)

    B = extendBasis([[v[0]] for v in P])
    P = orthogonalize(B)

    T = mult(trans(P), mult(A, P))
    T = [v[1:] for v in T]
    b = T[0]

    Q, T1 = triangularize(T[1:])

    NP = mult(P, [[1] + [0 for i in xrange(len(Q))]] + [[0] + v for v in Q])

    return NP, [[l[0]] + mult([b], Q)[0]] + [[0] + v for v in T1]

# Orthogonally Diagonalize Matrix


def orthodiagonalize(M):
    if not isSymmetric(M):
        raise Exception("Not a symmetric matrix")
    P = orthogonalize(eigenvec(M))
    return P, mult(trans(P), mult(M, P))

# Singular Value Decomposition


def svd(M):
    A = trans(M) if len(M) < len(M[0]) else M
    n = len(A[0])
    B = mult(trans(A), A)
    V, eigen = eigenvec(B, True)
    eigen = [eigen[i] ** 0.5 for i in xrange(n)]
    V = orthogonalize(V)

    U = trans([trans(scalar(mult(A, [[v[i]]
                                     for i in xrange(len(v))]), 1 / sing))[0]
               for v, sing in zip(trans(V), eigen) if sing])

    nullA = solve(A, [[0] for i in xrange(len(A))])

    if any(val != [0] for val in nullA):
        V = map(list.__add__, V, orthogonalize(nullA))

    lnullA = solve(trans(A), [[0] for i in xrange(len(A[0]))])

    if any(val != [0] for val in lnullA):
        U = map(list.__add__, U, orthogonalize(lnullA))

    S = [[eigen[i] if i == j else 0 for j in xrange(len(M[0]))] for i in xrange(len(M))]

    return (V, S, U) if len(A) < len(A[0]) else (U, S, V)
