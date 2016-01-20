# All operations provided

OPERATIONS = [('Add', 'add'),
              ('Subtract', 'sub'),
              ('Multiply', 'mult'),
              ('Transpose', 'trans'),
              ('RREF', 'rref'),
              ('Det', 'det'),
              ('Inverse', 'inv'),
              ('Trace', 'tr'),
              ('Adjoint', 'adj'),
              ('Solve', 'solve'),
              ('Eigenval', 'eigenval'),
              ('Eigenvec', 'eigenvec')]

# Functions requiring 2 or more matrices
BINARY_OP = ['add', 'sub', 'mult', 'solve', 'cramer']

MULTIPLE_OP = ['add', 'sub', 'mult']

primary = ['add', 'sub']

UNARY_OP = [
    'det',
    'tr',
    'trans',
    'rref',
    'inv',
    'adj',
    'eigenval',
    'eigenvec']      # Functions requiring just 1 matrix

ALL_OPS = [k[1] for k in OPERATIONS] + ['evaluate']	     # All function names

SQUARE_OP = [
    'det',
    'tr',
    'adj',
    'inv',
    'eigenval',
    'eigenvec'
]               # Functions requiring Square matrix

detailed = ['inv', 'rref', 'solve']	     # Functions providing a detailed solution

systems = ['cramer', 'solve']

solvable = ['solve']

SPECIAL = ['mult', 'evaluate']
