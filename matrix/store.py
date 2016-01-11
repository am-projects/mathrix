import random
import string

from error import Error


sessionKeyChoice = string.ascii_letters

try:
    random = random.SystemRandom()
except NotImplementedError:
    print "System PRNG is not available. Using the more predictable default one"


# Storage class to create and track user sessions

class Session(object):
    MAX_SESSIONS = 10000	# Maximum number of active connections

    def __init__(self):
        self.store = {}
        self.keymap = {}
        self.users = 1

    # Generate a session key to prevent change between user sessions

    def generateKey(self, val):
        key = ''.join(random.choice(sessionKeyChoice) for _ in xrange(8))
        self.keymap[val] = key
        return key

    def addUser(self, Exp):
        self.users = (self.users % self.MAX_SESSIONS) + 1
        self.removeUser(self.users)	# Remove any previously stored user sessions
        user = self.generateKey(self.users)
        self.store[user] = Exp
        return user

    def getUser(self, user):
        if self.store.get(user) is None:
            raise KeyError("User session cannot be validated. <a href='/mathrix'>Return</a> to the main page")
        else:
            return self.store[user]

    def removeUser(self, user):
        if self.keymap.get(user):
            self.store.pop(self.keymap[user])

