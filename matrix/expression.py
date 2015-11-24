from operations import *
import re

# For the lack of structures in Python :/


def parse(exp):
    if re.match(r"exit", exp):
        raise Exception("Invalid expression")
    exp = '**'.join(exp.split('^'))
    exp = re.sub(r"([A-Z])[*]{2}[T]", lambda x: "trans(%s)" % x.group(1), exp)
    exp = re.sub(r"([A-Z])[']", lambda x: "trans(%s)" % x.group(1), exp)
    return exp


def latexify2(exp):
    exp = r''.join(exp.split())
    exp = re.sub(r"[*]{2}", r"^", exp)
    # exp = r''.join(exp.split('*'))
    exp = re.sub(r"\^([-]?\d+)", lambda x: "^{%s}" % x.group(1), exp)
    return exp


class Expression(object):
    def __init__(self, q='', exp='', n=1):
        if not exp:
            var = [chr(x) for x in xrange(65, 65 + n)]
            if q in ['add', 'sub', 'mult']:
                self.exp = ('%s' % {'add': '+', 'sub': '-', 'mult': '*'}[q]).join(var)
            else:
                self.exp = q + '(' + ','.join(var) + ')'
            self.unknowns = var
            self.n = n
            self.q = q
        else:
            self.setExp(exp)

        self.pretty_exp = latexify2(self.exp)
        self.dim = []
        self.matrices = []
        self.error = ""

    def __repr__(self):
        return self.pretty_exp

    # Sets an Expression

    def setExp(self, exp):
        print "Expression is " + exp
        self.exp = parse(exp)
        self.q = "evaluate"

        unknowns = re.findall(r'([A-Z])', self.exp)
        unknowns = sorted(set(unknowns))

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
        raise Error("This expression can't be evaluated. Please check the specs.")
    return ans if isinstance(ans, Matrix) else [[ans]]
