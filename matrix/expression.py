import operations
import re

from error import Error
from operations import Matrix

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
        self.functions = operations.functions

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

    def evaluate(self, var):
        Exp = self
        print Exp.exp, " Vars: ", var
        for k in var:
            var[k] = Matrix(var[k])

        try:
            ans = eval(Exp.exp, self.functions, var)
        except:
            raise Error("This expression can't be evaluated. Please check the specs.")

        return ans if isinstance(ans, Matrix) else [[ans]]





"""
Alternate way to evaluate expressions
"""

# Match closing and opening parens

parens = {
    ')': '(',
    ']': '[',
    '}': '{'
}

# Check user expression

def check(exp):
    last = []
    openp = ['(', '[', '{']
    closep = [')', ']', '}']
    count = 0

    for i in exp:
        if i in openp:
            if i == '[':
                if count < 2:
                    count += 1
                else:
                    raise Error('A Matrix can only be constructed like [[1,2,...],[3,...],...]')
            last.append(i)
            print last, i
        elif i in closep:
            if i == ']':
                if count > 1:
                    count -= 1
                else:
                    raise Error('A Matrix can only be constructed like [[1,2,...],[3,...],...]')
            if last == [] or last.pop() != parens[i]:
                print last, i
                raise Error('Incorrect expression')

    if '[' in exp or ']' in exp:
        if not (re.match(r'[,[][[]', exp) and re.match(r'[]][ ,]]', exp)):
            raise Error('Incorrect Expression')


fl = re.compile(r'^[-]?\d+([.]\d*)?$')

# Expression evaluator

def getParam(var, par):
    print "Getting parameter: ", par
    if par.isdigit():
        par = int(par)
    elif fl.match(par):
        par = float(par)
    elif par[0] == '[':
        par = Matrix([solveExp(exp) for exp in par[1:-1].split(',')])
    elif len(par) == 1 and par.isupper():
        par = var[par]
    else:
        Exp = Expression()
        Exp.exp = par
        Exp.unknowns = var
        par = solveExp(Exp)
    return par

def solveExp(Exp):
    exp = '(' + Exp.exp + ')'
    var = Exp.unknowns
    frames = []
    params = ['']
    x = i = 0
    print "Solving ", exp

    def applyFunc(func):
        return funcs[func][0](*[getParam(params.pop(), var) for _ in xrange(funcs[func][1])])

    while i < len(exp):
        c = exp[i]
        print c
        if c == '(':
            frames.append(exp[x:i])
            x = i + 1
            if exp[i-1:i].isalpha():	# Check if previous char is a letter (means it's part of function name)
                params[-1] = ''
        elif c == ')':
            func = frames.pop()
            print params
            params.append(applyFunc(func))
        elif c == '[' and exp[i+1] == '[':
            x = i + 2
            end = exp[i:].find(']]')+2
            print "List end: ", end
            params.append(getParam(var, exp[i:end]))
            i = end - 1
        elif c == ',':
            params.append('')
            x = i + 1
        else:
            params[-1] = params[-1] + c
        i += 1

    return params

def evaluate(Exp, unknowns):
    print Exp.exp
    check(Exp.exp)
    Exp.unknowns = {var: val for var, val in zip(Exp.unknowns, unknowns)}
    return solveExp(Exp)

def readExp(exp, i):
    ans = ''
    isFunc = True
    while i < len(exp):
        c = exp[i]
        if c == '(':
            isFunc = True
            break
        elif c == ')':
            pass
