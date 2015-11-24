import webapp2
import os
import jinja2
import sys
import logging

import expression
from operations import func
from expression import re, Expression
from store import Session
from log import Log

# from google.appengine.ext import db

template_dir = os.path.join(os.path.dirname(__file__), '../templates')
jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir), autoescape=True)

# Debugging

DEBUG = os.environ['SERVER_SOFTWARE'].startswith('Development')  # Debug environment


def console(s):
    sys.stderr.write('%s\n' % repr(s))

# Floating point number check

fl = re.compile(r'^[-]?\d+([.]\d*)?$')

# All operations provided

formulas = [('Add', 'add'),
            ('Subtract', 'sub'),
            ('Multiply', 'mult'),
            ('Transpose', 'trans'),
            ('RREF', 'rref'),
            ('Determinant', 'det'),
            ('Inverse', 'inv'),
            ('Trace', 'tr'),
            ('Adjoint', 'adj'),
            ('Solve', 'solve'),
            ('Eigenvalues', 'eigenval'),
            ('Eigenvectors', 'eigenvec')]

double = ['add', 'sub', 'mult', 'solve', 'cramer']      # Functions requiring 2 or more matrices

multiple = ['add', 'sub', 'mult']

primary = ['add', 'sub']

single = [
    'det',
    'tr',
    'trans',
    'rref',
    'inv',
    'adj',
    'eigenval',
    'eigenvec']      # Functions requiring just 1 matrix

total = [k[1] for k in formulas] + ['evaluate']	     # All function names

square = [
    'det',
    'tr',
    'adj',
    'inv',
    'eigenval',
    'eigenvec']               # Functions requiring Square matrix

detailed = ['inv', 'rref', 'solve']	     # Functions providing a detailed solution

systems = ['cramer', 'solve']

solvable = ['solve']

special = ['mult', 'evaluate']


# User Sessions

users = Session()


def beautify(matrices):		# To display fractions
    if not matrices:
        return
    M = [[[i for i in row] for row in A] for A in matrices]
    for A in M:
        for i in xrange(len(A)):
            for j in xrange(len(A[0])):
                if not isinstance(A[i][j], str):
                    A[i][j] = func.make_frac(A[i][j])
    return M


class Handler(webapp2.RequestHandler):

    def home(self):               # Easy redirection to homepage
        self.redirect('/mathrix')

    def write(self, *a, **kw):
        self.response.write(*a, **kw)

    def render_str(self, template, **params):
        template = 'mathrix/' + template
        params['f'] = formulas
        params['square'] = square
        params['detailed'] = detailed
        params['systems'] = systems
        params['double'] = double
        t = jinja_env.get_template(template)
        return t.render(params)

    def render(self, template, **kw):
        self.write(self.render_str(template, **kw))

    def log(self):
        Exp = self.Exp
        logging.info("Expression to be evaluated: %s", Exp.exp)
        if Exp.dim:
            logging.info("Dimensions of matrices are: %s", Exp.dim)
        if Exp.matrices:
            logging.info("Inputed matrices are: %s", Exp.matrices)


class MainHandler(Handler):
    def get(self):
        logging.info("MainHandler: GET")
        self.render("front.html")

# Handler: /mathrix/num


class MultipleMatrices(Handler):
    def get(self):
        logging.info("MultipleMatrices: GET")
        q = self.request.get('q')

        if q not in multiple:	 # Goto home: Wrong operation
            self.home()

        self.render("get_num.html", q=q)

# Handler: /mathrix


class InputHandler(Handler):

    def match(self, q):
        # q = self.request.get('q')
        q = 'evaluate' if q == 'eval' else q
        num = self.request.get('num')
        if not q:
            return self.home()

        # Get number of matrices (redirect to /mathrix/num) for particular functions
        # if num is '' and q in multiple:
        #    return self.redirect('/mathrix/num?q=%s' % q)

        logging.info('Dimn %s with num %s', q, num)
        if q not in total and num is not '':
            return self.home()

        logging.info("Operation called: %s", q)
        num = 1 if q in single else (num or 2)

        user_id = self.request.get('user_id') or users.addUser(Expression(q=q, n=int(num)))

        try:
            Exp = self.Exp = users.getUser(int(user_id))
        except KeyError:
            logging.error("Wrong Key: %s", user_id)
            return self.home()

        logging.info("Created user - %s", int(user_id))

        self.render("get_dimn.html", num=Exp.n, q=q, user_id=user_id,
                    double=double, exp=Exp.getExp(), unknowns=Exp.getVars())

    def get(self, q):
        logging.info("InputHandler: GET")
        logging.info("Number of active users = %s" % users.users)
        self.match(q)

    # For the obtaining the dimensions and rendering the matrix input

    def test(self, m, n, q='det', num='1'):
        logging.error("Checking dimensions m:%s n:%s" % (m, n))
        if func.chk(m, n) or q not in total or not (num and num.isdigit()):
            # logging.error(self.request.get_all('q'))
            self.home()
            return
        return True

    def get_dim(self):
        q = self.Exp.q
        num = self.Exp.n
        dim = self.Exp.dim
        if dim and dim[0][0]:
            return
        for i in xrange(num if q not in primary else 1):
            m = self.request.get('r' + str(i))
            # If operation works on only square matrices
            n = self.request.get('c' + str(i)) if q not in square else m
            if not self.test(m, n):
                # error = 'All fields are required'
                # self.Exp.dim.append(['', ''])
                raise Exception("All dimensions are required")
            else:
                dim.append([int(m), int(n)])
        logging.info("Matrix dimensions are: %s", self.Exp.dim)

        # For add and sub, only one dimension set is taken
        if q == 'add' or q == 'sub':
            self.Exp.dim = [dim[0] for i in xrange(num)]
        # logging.error('finding dimn ' + error)
        # if not error:
        #    error = self.chk_dim(dim, q)
        # return dim

    def chk_dim(self, dim, q):
        logging.error('Checking dimensions for ' + q)
        if q in square:
            return 'Can be calculated only for square matrices' if dim[0][0] != dim[0][1] else ''
        elif q == 'mult':
            r = dim[0][0]
            for m, n in dim:
                if m != r:
                    return 'Matrix multiplication requires no. of columns and rows of 2 consecutive matrices to be the same'
                r = n
        elif q == 'solve' and dim[0][0] != dim[1][0]:
            return "Number of linear equations and solution vector doesn't match"
        return ''

    def generateMatrices(self):
        logging.info("MatrixHandler: Generating Matrices")
        return [[['' for j in xrange(n)] for i in xrange(m)] for (m, n) in self.Exp.dim]

    def post(self, q):
        logging.info("InputHandler: POST")
        user_id = self.request.get('user_id')

        if user_id is '' or not user_id.isdigit():
            self.home()
            return

        try:
            self.Exp = users.getUser(int(user_id))
        except KeyError:
            logging.error("Wrong Key: %s", user_id)
            return self.home()
        self.log()
        try:
            self.get_dim()
            self.Exp.matrices = self.generateMatrices()
        except Exception as inst:
            self.Exp.error = inst.args[0]

        self.redirect('/mathrix/input?user_id=%s' % user_id)


# Handler: /mathrix/input


class MatrixHandler(Handler):

    def initialize(self, *a, **kw):
        webapp2.RequestHandler.initialize(self, *a, **kw)
        self.user_id = user_id = self.request.get('user_id')
        if user_id is '' or not user_id.isdigit():
            self.home()
            return
        logging.info("User: %s", user_id)
        try:
            self.Exp = users.getUser(int(user_id))
        except KeyError:
            logging.error("Wrong Key: %s", user_id)
            return self.home()
        logging.info("Initialized MatrixHandler")
        self.log()

    def getMatrices(self):
        dim = self.Exp.dim
        matrices = self.Exp.matrices
        if matrices and matrices[0][0][0]:
            return ''

        error = ''
        for p, (r, c) in enumerate(dim):
            A = matrices[p]
            for i in xrange(r):
                for j in xrange(c):
                    A[i][j] = self.request.get('A[%d][%d][%d]' % (p, i, j))
                    if not (A[i][j] and fl.match(A[i][j])):
                        error = "All cells need to be filled"
                    else:
                        A[i][j] = float(A[i][j])

        return error

    def get(self):
        logging.info("MatrixHandler: GET")
        self.render("input.html", user_id=self.user_id, error=self.Exp.error, matrices=self.Exp.matrices,
                    q=self.Exp.q, exp=self.Exp.exp, unknowns=self.Exp.getVars())

    def post(self):
        logging.info("MatrixHandler: POST")

        error = self.getMatrices()
        if error:
            logging.error("MatrixHandler: Matrix Input Error")
            self.get()

        return self.redirect('/mathrix/result?user_id=%s' % self.user_id)

# Expression Evaluator


class EvalHandler(Handler):

    def get(self):
        logging.info("EvalHandler: GET")
        self.render("evaluate.html")

    def post(self):
        logging.info("EvalHandler: POST")
        exp = self.request.get('eval')
        try:
            Exp = Expression(exp=exp)
            user_id = users.addUser(Exp)
            logging.info('Evaluate called for exp:%s' % exp)
            return self.redirect('/mathrix/eval?user_id=%s' % user_id)
        except Exception as inst:
            self.render("evaluate.html", exp=exp, error=inst.args[0])
            return


class OutputHandler(Handler):
    def initialize(self, *a, **kw):
        webapp2.RequestHandler.initialize(self, *a, **kw)
        self.user_id = user_id = self.request.get('user_id')
        if user_id is '' or not user_id.isdigit():
            logging.error("OutputHandler: User session Not Found")
            self.home()
            return
        logging.info("User: %s", user_id)
        try:
            self.Exp = users.getUser(int(user_id))
        except KeyError:
            Log.wrongKey(user_id)
            return self.home()
        self.log()

    # Performs the actual calculation
    def calc(self, op, matrices):
        logging.info("Input matrices: %s", matrices)
        f = getattr(func, op)
        if op in detailed:
            return f(*matrices)
        result = f(*matrices[:2])
        for A in matrices[2:]:
            result = f(result, A)
        return result

    def post(self):
        logging.info("OutputHandler: POST")
        self.get()

    def get(self):
        logging.info("OutputHandler: GET")
        Exp = self.Exp
        matrices = Exp.matrices

        if matrices == []:
            return self.home()

        op = Exp.q
        logging.info('Operation %s to be executed' % op)
        soln, steps, error, span = None, None, None, None
        logging.info("Loading...")
        self.render("loading.html")
        logging.info("In detailed? (%s)", op)
        logging.info("Detailed option: %s", self.request.get('detail'))
        try:
            if op == 'evaluate':
                logging.info('Evaluating %s', Exp.exp)

                ans = [expression.evaluate(Exp, {k: v for k, v in zip(Exp.getVars(), matrices)})]
            elif op in detailed:
                logging.info("Detailed solution for %s", op)
                # matrices.append(True)
                ans, steps, soln = self.calc(op, matrices + [True])
                if op in solvable:
                    span = func.span(ans[-1], soln[-1])
                    span = beautify([span])[0]
                soln = beautify(soln)
            else:
                logging.info("Calculating result")
                ans = [self.calc(op, matrices)]
                if op in solvable:
                    span = ans[0]
                    span = beautify([span])[0]
                    ans, soln = [matrices[0]], [matrices[1]]
                    soln = beautify(soln)
            ans = beautify(ans)
        except Exception as inst:
            error = True
            logging.error("Calculation Error " + repr(inst))
            ans = inst.args[0]
        except:
            self.home()
        logging.info("Calculated result")
        # logging.error(steps)
        msg = "The result is"
        self.response.clear()	 # To clear the 'loading' response
        self.render("output.html", msg=msg, ans=ans, steps=steps,
                    soln=soln, error=error, span=span, q=op, exp=Exp.getExp())


app = webapp2.WSGIApplication([
    (r'/mathrix/result.*', OutputHandler),
    (r'/mathrix/num.*', MultipleMatrices),
    (r'/mathrix/input.*', MatrixHandler),
    (r'/mathrix/evaluate.*', EvalHandler),
    (r'/mathrix/([a-z]+)', InputHandler),
    (r'/mathrix', MainHandler)
], debug=DEBUG)
