import hmac
import hashlib

from error import Error

SECRET = 'QWERTY' # open('/matrix/secret.txt').read()


# Encrypt user id preventing change between user sessions

def encrypt(val):
    return hmac.new(SECRET, str(val), hashlib.sha256).hexdigest()


# Storage class to create and track user sessions

class Session(object):
    MAX_SIZE = 10000	# Maximum number of active connections

    def __init__(self):
        self.store = {}
        self.users = 1

    def addUser(self, Exp):
        self.users = (self.users % self.MAX_SIZE) + 1
        user = encrypt(self.users)
        self.store[user] = Exp
        return user

    def getUser(self, user):
        if self.store.get(user) is None:
            raise KeyError("User session cannot be validated. <a href='/mathrix'>Return</a> to the main page")
        else:
            return self.store[user]

    def removeUser(self, user):
        self.store.pop(user)
