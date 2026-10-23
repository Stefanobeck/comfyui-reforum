"""Safe evaluation of the maths used in Deforum schedules and prompts.

Deforum (A1111) evaluated these strings with numexpr. numexpr is not part of ComfyUI, and a plain eval() is not safe,
so this module walks the Python syntax tree itself and only allows numbers, the names given by the caller and the
numexpr functions. Same results as numexpr for every expression Deforum accepted.
"""
import ast
import math
import operator

import numpy as np


def _where(cond, a, b):
    return a if cond else b


# The functions numexpr offered (plus a few harmless extras Deforum users write: pi, floor, ceil, min, max, round).
FUNCTIONS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "arcsin": math.asin, "arccos": math.acos, "arctan": math.atan, "arctan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "arcsinh": math.asinh, "arccosh": math.acosh, "arctanh": math.atanh,
    "log": math.log, "log10": math.log10, "log1p": math.log1p, "exp": math.exp, "expm1": math.expm1,
    "sqrt": math.sqrt, "abs": abs, "where": _where, "floor": math.floor, "ceil": math.ceil,
    "min": min, "max": max, "round": round,
}
CONSTANTS = {"pi": math.pi, "e": math.e}

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow,
        ast.BitAnd: lambda a, b: bool(a) and bool(b), ast.BitOr: lambda a, b: bool(a) or bool(b),
        ast.BitXor: lambda a, b: bool(a) != bool(b)}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: lambda a: not bool(a), ast.Not: lambda a: not a}
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.Eq: operator.eq, ast.NotEq: operator.ne}

MAX_EXPONENT = 100


class ExpressionError(ValueError):
    pass


def evaluate(expression, **names):
    """Value of a maths expression. names: the variables (t, max_f, s ...)."""
    text = str(expression).strip()
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"cannot read the expression '{text}': {e.msg}") from None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise ExpressionError(f"exponent too large in '{text}'")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Compare) and all(type(o) in _CMP for o in node.ops):
            left = ev(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = ev(comp)
                if not _CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            if node.id in CONSTANTS:
                return CONSTANTS[node.id]
            raise ExpressionError(f"unknown name '{node.id}' in '{text}'")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS
                and not node.keywords):
            return FUNCTIONS[node.func.id](*[ev(a) for a in node.args])
        raise ExpressionError(f"not allowed in a Deforum expression: '{ast.unparse(node)}' (in '{text}')")

    value = ev(tree)
    if isinstance(value, (bool, np.bool_)):
        value = float(value)
    return value
