import logging


class Log(object):
    def wrongOp(self, q):
        logging.error("Wrong operation: %s", q)

    def exp(self, exp):
        logging.info("Evaluating expression: %s", exp)

    def wrongKey(self, key):
        logging.error("Wrong key: %s", key)
