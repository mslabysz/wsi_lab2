import numpy as np


def quadratic(x):
    return np.sum(x**2)


_cec2017 = None


def _cec2017_evaluator():
    global _cec2017
    if _cec2017 is None:
        from cecpy.benchmark import CECEdition, CECEvaluator

        _cec2017 = CECEvaluator(CECEdition.CEC2017, [10, 30])
    return _cec2017


def _cec2017_value(x, problem):
    column = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    return float(_cec2017_evaluator()(problem, column)[0])


def f3(x):
    return _cec2017_value(x, 3)


def f18(x):
    return _cec2017_value(x, 18)
