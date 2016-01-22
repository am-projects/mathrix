"""
mathrix.py is responsible for handling the HTTP requests made by users
"""


import webapp2

import feedback
import operations

from error import Error
from expression import re, Expression
from handler import Handler, DEBUG
from log import Log
from operators import *
from store import Session



# Floating point number check

fl = re.compile(r'^[-]?\d+([.]\d*)?$')


# User Sessions

users = Session()


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

        if q not in MULTIPLE_OP:	 # Goto home: Wrong operation
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
        Log.i("match() called")
        q = 'evaluate' if q == 'eval' else q
        num = self.request.get('num')
        if not q:
            return self.home()

        Log.i('Dimn %s with num %s', q, num)
        if q not in ALL_OPS and num is not '':
            return self.home()

        Log.i("Operation called: %s", q)
        num = 1 if q in UNARY_OP else (num or 2)

        user_id = self.request.get('user_id') or users.addUser(Expression(q=q, n=int(num)))

        try:
            Exp = self.Exp = users.getUser(user_id)
        except KeyError:
            Log.e("Wrong Key: %s", user_id)
            return self.home()

        Log.i("Created user - %s", user_id)

        self.render("get_dimn.html", num=Exp.n, q=q, user_id=user_id,
                    double=BINARY_OP, exp=Exp.getExp(), unknowns=Exp.getVars())

    def get(self, q):
        Log.get(self)
        Log.i("Operation: %s called", q)
        if q not in ALL_OPS:
            return self.home()
        Log.i("Number of active users = %s" % users.users)
        self.match(q)

    # For the obtaining the dimensions and rendering the matrix input

    def test(self, m, n, q='det', num='1'):
        Log.e("Checking dimensions m:%s n:%s" % (m, n))
        if operations.chk(m, n) or q not in ALL_OPS or not (num and num.isdigit()):
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
            # If operation works on only SQUARE_OP matrices
            n = self.request.get('c' + str(i)) if q not in SQUARE_OP else m
            if not self.test(m, n):
                raise Error("All dimensions are required")
            else:
                dim.append([int(m), int(n)])
        Log.i("Matrix dimensions are: %s", self.Exp.dim)

        # For add and sub, only one dimension set is taken
        if q == 'add' or q == 'sub':
            self.Exp.dim = [dim[0] for i in xrange(num)]

    def chk_dim(self, dim, q):
        Log.e('Checking dimensions for ' + q)
        if q in SQUARE_OP:
            return 'Can be calculated only for SQUARE_OP matrices' if dim[0][0] != dim[0][1] else ''
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

"""
Handler: /mathrix/evaluate
Renders the evaluate expression page
"""


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


"""
Handler: /mathrix/evaluate/help
Renders the help page for evaluating expression
"""


class HelpHandler(Handler):
    def get(self):
        Log.get(self)
        self.render("help.html")


"""
Handler: /mathrix/result
Calculates and displays the result
"""


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

        retry = 3

        while retry:
            try:
                retry -= 1
                ans = Exp.evaluate({k: v for k, v in zip(Exp.getVars(), matrices)})
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
    (r'/mathrix/result', OutputHandler),
    (r'/mathrix/num', MultipleMatrices),
    (r'/mathrix/input', MatrixHandler),
    (r'/mathrix/evaluate/help', HelpHandler),
    (r'/mathrix/evaluate', EvalHandler),
    (r'/mathrix/feedback', feedback.FeedbackHandler),
    (r'/mathrix/([a-z]+)', InputHandler),
    (r'/mathrix', MainHandler)
], debug=DEBUG)
