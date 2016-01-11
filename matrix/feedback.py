import webapp2

from google.appengine.api import mail
from handler import Handler, DEBUG

from log import Log

message = mail.EmailMessage(subject="Mathrix Feedback",
                            to = "Admin <ourproject.am@gmail.com>")

SENDER_TEMPLATE = "Feedback <{0}>"

default = "ourproject.am@gmail.com"

body = """
New Feedback:

{0}
"""


class FeedbackHandler(Handler):

    def get(self):
        Log.get(self)
        self.render("feedback.html")

    def post(self):
        Log.post(self)
        feedback = self.request.get('comment')
        message.body = body.format(feedback)
        sender = self.request.get('email')
        message.sender = SENDER_TEMPLATE.format(sender or default)
        message.send()
        self.render("feedback.html")
