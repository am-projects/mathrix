import func
import re


class Matrix(object):

    def __init__(self, M):
        self.M = M

    def __rmul__(self, other):
        if isinstance(other, int):
            return Matrix([[c * other for c in r] for r in self.M])
        else:
            return Matrix(func.mult(self.M, other.M))

    def __add__(self, other):
        return Matrix(func.add(self.M, other.M))

    def __sub__(self, other):
        return Matrix(func.sub(self.M, other.M))

    def __mul__(self, other):
        if isinstance(other, int):
            return Matrix([[c * other for c in r] for r in self.M])
        else:
            return Matrix(func.mult(self.M, other.M))

    def __pow__(self, x):
        def modexpt(A, e):
            if e == 1:
                return A
            elif e % 2 == 0:
                X = modexpt(A, e / 2)
                return func.mult(X, X)
            else:
                X = modexpt(A, e // 2)
                return func.mult(A, func.mult(X, X))

        if len(self.M) != len(self.M[0]):
            raise Exception("Not a Square Matrix")
        elif x < 0:
            return Matrix(func.inv(modexpt(self.M, abs(x))))
        elif x == 0:
            return Matrix(func.make_id(len(self.M)))
        else:
            return Matrix(modexpt(self.M, x))

    def __str__(self):
        return str([[func.make_frac(w) for w in v] for v in self.M])

    def __len__(self):
        return len(self.M)

    def __repr__(self):
        return repr(self.M)

    def clen(self):
        return len(self.M[0])


def dot(A, B):
    return func.dot(A.M, B.M)[0][0]


def tr(self):
    return func.tr(self.M)[0][0]


def id(n):
    return Matrix(func.make_id(n))


def rows(A):
    return len(A)


def cols(A):
    return A.clen()


def inv(self):
    return Matrix(func.inv(self.M))


def adj(self):
    return Matrix(func.adj(self.M))


def det(self):
    return func.deter(self.M)


def trans(self):
    return Matrix(func.trans(self.M))


def rref(self):
    return Matrix(func.rref(self.M))


def egval(self):
    return Matrix(func.eigenval(self.M))


def egvec(self):
    return Matrix(func.eigenvec(self.M))


def col(self):
    return Matrix(func.subspaces(self.M)[0])


def row(self):
    return Matrix(func.subspaces(self.M)[1])


def null(self):
    return Matrix(func.subspaces(self.M)[2])


def lnull(self):
    return Matrix(func.subspaces(self.M)[3])

# For the lack of structures in Python :/


def parse(exp):
    if re.match(r"exit", exp):
        raise Exception("Invalid expression")
    exp = '**'.join(exp.split('^'))
    exp = re.sub(r"([A-Z])[*]{2}[T]", lambda x: "trans(%s)" % x.group(1), exp)
    exp = re.sub(r"([A-Z])[']", lambda x: "trans(%s)" % x.group(1), exp)
    return exp


class Expression(object):
    def __init__(self, q='', exp='', n=1):
        if not exp:
            var = [chr(x) for x in xrange(65, 65 + n)]
            if q in ['add', 'sub', 'mult']:
                self.exp = ('%s' % {'add':'+', 'sub':'-', 'mult':'*'}[q]).join(var)
            else:
                self.exp = q + '(' + ','.join(var) + ')'
            self.pretty_exp = repr(self)
            self.unknowns = var
            self.n = n
            self.q = q
        else:
            self.setExp(exp)
        self.dim = []
        self.matrices = []
        self.error = ""

    def __repr__(self):
        #if self.pretty_exp:
         #   return self.pretty_exp

        exp = r''.join(self.exp.split())
        exp = re.sub(r"[*]{2}", r"^", exp)
        exp = r''.join(exp.split('*'))
        exp = re.sub(r"\^([-]?\d+)", lambda x: "^{%s}" % x.group(1), exp)
        return exp

    # Sets an Expression

    def setExp(self, exp):
        print "Expression is " + exp
        self.exp = parse(exp)
        self.q = "evaluate"
        pretty_exp = repr(self)

        unknowns = re.findall(r'([A-Z])', self.exp)
        unknowns = sorted(set(unknowns))

        self.pretty_exp = pretty_exp
        self.unknowns = unknowns
        self.n = len(unknowns)

    def getExp(self):
        return self.pretty_exp

    def getVars(self):
        return self.unknowns

    def chkDim(self):
        return self.dim is not None

    def chkMat(self):
        return self.matrices is not None

    def setVars(self, matrices):
        self.matrices = [Matrix(x) for x in matrices]



def evaluate(Exp, var):
    for k in var:
        var[k] = Matrix(var[k])
    try:
        ans = eval(Exp.exp, globals(), var)
    except:
        raise Exception("This expression can't be evaluated. Please check the specs.")
    return ans.M if isinstance(ans, Matrix) else [[ans]]
