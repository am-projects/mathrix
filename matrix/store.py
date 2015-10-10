# Storage class to create and track user sessions

class Session(object):
    MAX_SIZE = 10000

    def __init__(self):
        self.store = {}
        self.users = 0

    def addUser(self, Exp):
        self.users += 1
        user_id = self.users
        if self.users > self.MAX_SIZE:
            user_id = self.users = 1

        self.store[user_id] = Exp
        return user_id

    def getUser(self, user_id):
        return self.store[user_id]

    def removeUser(self, user_id):
        self.users -= 1
        self.store.pop(user_id)
