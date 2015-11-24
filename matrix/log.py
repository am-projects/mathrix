import logging

# Logger Class for swift logging operations

Logger = logging.getLogger('Mathrix')

class Log(object):
    @staticmethod
    def wrongOp(q):
        Logger.error("Wrong operation: %s", q)

    @staticmethod
    def exp(exp):
        Logger.info("Evaluating expression: %s", exp)

    @staticmethod
    def wrongKey(key):
        Logger.error("Wrong key: %s", key)

    @staticmethod
    def get(clss):
        Logger.info("%s: GET" % clss.__class__.__name__)

    @staticmethod
    def post(clss):
        Logger.info("%s: POST" % clss.__class__.__name__)

    @staticmethod
    def wtf(log, *a, **kw):
        Logger.critical(log, *a, **kw)

    @staticmethod
    def e(log, *a, **kw):
        Logger.error(log, *a, **kw)

    @staticmethod
    def w(log, *a, **kw):
        Logger.warn(log, *a, **kw)

    @staticmethod
    def i(log, *a, **kw):
        Logger.info(log, *a, **kw)

    @staticmethod
    def d(log, *a, **kw):
        Logger.debug(log, *a, **kw)
