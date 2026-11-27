"""Check the Boolean grammar before handing model text to SymPy."""

import ast
import re


_NOR_CALL = re.compile(r"\bNOR\s*\(([^)]*)\)", re.IGNORECASE)
_SYMPY_CALLS = {"And", "Or", "Not", "Xor", "Implies", "Equivalent"}
_ALLOWED_NODES = (
    ast.Expression, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.Invert, ast.Call,
)


def translate_formula(formula: str) -> str:
    # Match BooleanExpression._parse_formula in boolean_dataset.py:117-139.
    translated = _NOR_CALL.sub(lambda match: "~(" + " | ".join(match[1].split(",")) + ")", formula)
    return translated.replace("AND", "&").replace("OR", "|").replace("NOT", "~").replace("XOR", "^").strip()


def is_safe_formula(formula: str, variables: list[str]) -> bool:
    try:
        tree = ast.parse(translate_formula(formula), mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return False
    allowed_names = set(variables)
    calls = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False
        if isinstance(node, ast.Name):
            legal = _SYMPY_CALLS if id(node) in calls else allowed_names
            if node.id not in legal:
                return False
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            return False
    return True
