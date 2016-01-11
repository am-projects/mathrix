import webapp2
import jinja2
import os

from log import Log
from operators import *


# Turns on debugging if code is not being run in production mode

DEBUG = os.environ['SERVER_SOFTWARE'].startswith('Development')  # Debug environment

##############################

# Handlers for Mathrix

##############################


template_dir = os.path.join(os.path.dirname(__file__), '../templates')
jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir), autoescape=True)


def render_str(template, **params):
    template = 'mathrix/' + template
    params['f'] = OPERATIONS
    params['SQUARE_OP'] = SQUARE_OP
    params['detailed'] = detailed
    params['systems'] = systems
    params['BINARY_OP'] = BINARY_OP
    t = jinja_env.get_template(template)
    return t.render(params)


"""
Base Handler defining convenience template render functions
"""
class Handler(webapp2.RequestHandler):

    def home(self):               # Easy redirection to homepage
        self.redirect('/mathrix')

    def write(self, *a, **kw):
        self.response.write(*a, **kw)

    def render(self, template, **kw):
        self.write(render_str(template, **kw))

    def log(self):
        Exp = self.Exp
        Log.i("Expression to be evaluated: %s", Exp.exp)
        if Exp.dim:
            Log.i("Dimensions of matrices are: %s", Exp.dim)
            if Exp.matrices:
                Log.i("Inputed matrices are: %s", Exp.matrices)
