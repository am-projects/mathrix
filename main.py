import webapp2
import os
import jinja2

template_dir = os.path.join(os.path.dirname(__file__), '.')
jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir), autoescape=True)


class Handler(webapp2.RequestHandler):

    def write(self, *a, **kw):
        self.response.write(*a, **kw)

    def render_str(self, template, **params):
        t = jinja_env.get_template(template)
        return t.render(params)

    def render(self, template, **kw):
        self.write(self.render_str(template, **kw))


class MainHandler(Handler):

    def get(self):
        # self.render("tryside.html")
        self.redirect('/mathrix')
        # self.response.write("Hello World!!")

app = webapp2.WSGIApplication([
    ('/', MainHandler)
], debug=True)
