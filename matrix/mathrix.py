import jinja2
import os
import webapp2

import expression
import operations

from error import Error
from expression import re, Expression
from log import Log
from store import Session

# from google.appengine.ext import db

template_dir = os.path.join(os.path.dirname(__file__), '../templates')
jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir), autoescape=True)

# Turns on debugging if code is not being run in production mode

DEBUG = os.environ['SERVER_SOFTWARE'].startswith('Development')  # Debug environment


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


##############################

# Handlers for Mathrix

##############################

"""Base Handler defining convenience template render functions"""


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
        Log.i("Expression to be evaluated: %s", Exp.exp)
        if Exp.dim:
            Log.i("Dimensions of matrices are: %s", Exp.dim)
        if Exp.matrices:
            Log.i("Inputed matrices are: %s", Exp.matrices)

"""
Handler: /mathrix
Renders the the front page
"""


class MainHandler(Handler):
    def get(self):
        Log.get(self)
        self.render("front.html")

"""
Handler: /mathrix/num
Gets the number of matrices for add/sub/mult
"""


class MultipleMatrices(Handler):
    def get(self):
        Log.get(self)
        q = self.request.get('q')

        if q not in multiple:	 # Goto home: Wrong operation
            self.home()

        self.render("get_num.html", q=q)

"""
Handler: /mathrix?q=
Gets the dimensions of input matrices
"""


class InputHandler(Handler):

    def initialize(self, *a, **kw):
        webapp2.RequestHandler.initialize(self, *a, **kw)
        self.Exp = None

    def match(self, q):
        # q = self.request.get('q')
        q = 'evaluate' if q == 'eval' else q
        num = self.request.get('num')
        if not q:
            return self.home()

        # Get number of matrices (redirect to /mathrix/num) for particular functions
        # if num is '' and q in multiple:
        #    return self.redirect('/mathrix/num?q=%s' % q)

        Log.i('Dimn %s with num %s', q, num)
        if q not in total and num is not '':
            return self.home()

        Log.i("Operation called: %s", q)
        num = 1 if q in single else (num or 2)

        user_id = self.request.get('user_id') or users.addUser(Expression(q=q, n=int(num)))

        try:
            Exp = self.Exp = users.getUser(user_id)
        except KeyError:
            Log.e("Wrong Key: %s", user_id)
            return self.home()

        Log.i("Created user - %s", user_id)

        self.render("get_dimn.html", num=Exp.n, q=q, user_id=user_id,
                    double=double, exp=Exp.getExp(), unknowns=Exp.getVars())

    def get(self, q):
        Log.get(self)
        Log.i("Number of active users = %s" % users.users)
        self.match(q)

    # For the obtaining the dimensions and rendering the matrix input

    def test(self, m, n, q='det', num='1'):
        Log.e("Checking dimensions m:%s n:%s" % (m, n))
        if operations.chk(m, n) or q not in total or not (num and num.isdigit()):
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
                raise Error("All dimensions are required")
            else:
                dim.append([int(m), int(n)])
        Log.i("Matrix dimensions are: %s", self.Exp.dim)

        # For add and sub, only one dimension set is taken
        if q == 'add' or q == 'sub':
            self.Exp.dim = [dim[0] for i in xrange(num)]
        # Log.e('finding dimn ' + error)
        # if not error:
        #    error = self.chk_dim(dim, q)
        # return dim

    def chk_dim(self, dim, q):
        Log.e('Checking dimensions for ' + q)
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
        Log.i("MatrixHandler: Generating Matrices")
        return [[[''] * n for _ in xrange(m)] for (m, n) in self.Exp.dim]

    def post(self, q):
        Log.post(self)
        user_id = self.request.get('user_id')

        if user_id is '':
            self.home()
            return

        try:
            self.Exp = users.getUser(user_id)
        except KeyError:
            Log.wrongKey(user_id)
            return self.home()
        self.log()
        try:
            self.get_dim()
            self.Exp.matrices = self.generateMatrices()
        except Exception as inst:
            self.Exp.error = inst.args[0]

        self.redirect('/mathrix/input?user_id=%s' % user_id)


"""
Handler: /mathrix/input
Gets the input matrices
"""


class MatrixHandler(Handler):

    def initialize(self, *a, **kw):
        webapp2.RequestHandler.initialize(self, *a, **kw)
        self.user_id = user_id = self.request.get('user_id')
        if user_id is '':
            self.home()
            return
        Log.i("User: %s", user_id)
        try:
            self.Exp = users.getUser(user_id)
        except KeyError:
            Log.wrongKey(user_id)
            return self.home()
        Log.i("Initialized MatrixHandler")
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
        Log.get(self)
        self.render("input.html", user_id=self.user_id,
                    error=self.Exp.error, matrices=self.Exp.matrices,
                    q=self.Exp.q, exp=self.Exp.exp, unknowns=self.Exp.getVars())

    def post(self):
        Log.post(self)

        error = self.getMatrices()
        if error:
            Log.e("MatrixHandler: Matrix Input Error")
            self.get()

        return self.redirect('/mathrix/result?user_id=%s' % self.user_id)

# Expression Evaluator


class EvalHandler(Handler):

    def get(self):
        Log.get(self)
        self.render("evaluate.html")

    def post(self):
        Log.post(self)
        exp = self.request.get('eval')
        try:
            Exp = Expression(exp=exp)
            user_id = users.addUser(Exp)
            Log.i('Evaluate called for exp:%s' % exp)
            return self.redirect('/mathrix/eval?user_id=%s' % user_id)
        except Exception as inst:
            self.render("evaluate.html", exp=exp, error=inst.args[0])
            return


class OutputHandler(Handler):
    def initialize(self, *a, **kw):
        webapp2.RequestHandler.initialize(self, *a, **kw)
        self.user_id = user_id = self.request.get('user_id')
        if user_id is '':
            Log.e("OutputHandler: User session Not Found")
            self.home()
            return
        Log.i("User: %s", user_id)
        try:
            self.Exp = users.getUser(user_id)
        except KeyError:
            Log.wrongKey(user_id)
            return self.home()
        self.log()

    # TODO: Remove this, not used for computation anymore
    # Performs the actual calculation
    def calc(self, op, matrices):
        Log.i("Input matrices: %s", matrices)
        f = getattr(operations, op)
        if op in detailed:
            return f(*matrices)
        result = f(*matrices[:2])
        for A in matrices[2:]:
            result = f(result, A)
        return result

    def post(self):
        Log.post(self)
        self.get()

    def get(self):
        Log.get(self)
        Exp = self.Exp
        matrices = Exp.matrices

        if matrices == []:
            return self.home()

        op = Exp.q
        Log.i('Operation %s to be executed' % op)
        ans = operations.Matrix([])
        error = ''
        Log.i("Loading...")
        self.render("loading.html")

        retry = 2

        while retry:
            try:
                retry -= 1
                ans = expression.evaluate(Exp, {k: v for k, v in zip(Exp.getVars(), matrices)})
                Log.i("Calculated result %s", ans)
                Log.i("Steps %s", ans.steps)
                break
            except Error as inst:
                Log.e("Calculation Error " + repr(inst))
                error = inst.args[0]
                break
            except:
                if retry == 0:
                    return self.home()

        self.response.clear()	 # To clear the 'loading' response

        self.render("output.html", ans=str(ans), steps=ans.steps,
                    error=error, q=op, exp=Exp.getExp())


app = webapp2.WSGIApplication([
    (r'/mathrix/result.*', OutputHandler),
    (r'/mathrix/num.*', MultipleMatrices),
    (r'/mathrix/input.*', MatrixHandler),
    (r'/mathrix/evaluate.*', EvalHandler),
    (r'/mathrix/([a-z]+)', InputHandler),
    (r'/mathrix', MainHandler)
], debug=DEBUG)
