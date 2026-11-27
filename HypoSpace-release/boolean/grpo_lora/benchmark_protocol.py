"""Official Boolean prompt and response parsing without the API client dependency."""

import re
import sys
from typing import List, Optional

from config import BOOLEAN_DIR

if str(BOOLEAN_DIR) not in sys.path:
    sys.path.insert(0, str(BOOLEAN_DIR))

from boolean_dataset import BooleanObservation


class BooleanPrompt:
    def __init__(self, task):
        self.variables = task["variables"]
        self.operators = task["operators"]
        self.max_depth = task["max_depth"]
        self.complete_dataset = {"metadata": {"mechanistic_opts": task["mechanistic_opts"]}}

    # Ported from boolean_benchmark.py:79-238 at 75cf9f5.
    def create_prompt(self, observations: List[BooleanObservation],
                      prior_hypotheses: List[str]) -> str:
        obs_lines = []
        for obs in observations:
            obs_lines.append(obs.to_string())
        obs_block = "\n".join(obs_lines)

        # Format prior hypotheses
        if prior_hypotheses:
            prior_block = "\n".join([f"Expression: {h}" for h in prior_hypotheses])
        else:
            prior_block = "None"

        # Build operator description
        op_descriptions = []
        if 'AND' in self.operators: op_descriptions.append("AND (conjunction)")
        if 'OR'  in self.operators: op_descriptions.append("OR (disjunction)")
        if 'NOT' in self.operators: op_descriptions.append("NOT (negation)")
        if 'XOR' in self.operators: op_descriptions.append("XOR (exclusive or)")
        if 'NOR' in self.operators: op_descriptions.append("NOR (NOT (x OR y))")

        operators_str = ", ".join(op_descriptions)

        # Pull mechanistic opts from the loaded dataset
        mech = self.complete_dataset.get('metadata', {}).get('mechanistic_opts', {})
        comm = mech.get('apply_commutativity', True)
        idem = mech.get('apply_idempotence_and_or', True)
        flat = mech.get('flatten_associativity', True)

        eq_rules = []
        if comm:
            eq_rules.append(
                "- Commutativity: reorderings are the SAME (e.g., x AND y = y AND x)."
            )
        if idem:
            eq_rules.append(
                "- Idempotence: duplicates of the same input under AND/OR collapse (e.g., x AND x = x; x OR x = x)."
            )
        if flat:
            eq_rules.append(
                "- Associativity flattening: cascades of the SAME operator are the SAME regardless of parentheses "
                "(e.g., (x AND y) AND x = x AND (y AND x) = x AND x AND y)."
            )
        else:
            eq_rules.append(
                "- No associativity flattening: different parenthesizations of the SAME operator are DIFFERENT "
                "(e.g., (x AND y) AND x ≠ x AND (y AND x))."
            )

        eq_rules.append(
            "- No distributivity, absorption, or De Morgan normalization: mixing operators keeps expressions DIFFERENT "
            "(e.g., (x AND y) OR z ≠ x AND (y OR z))."
        )

        rules_text = "\n        ".join(eq_rules)

        prompt = f"""You are given partial observations of a Boolean function with variables: {', '.join(self.variables)}

            Allowed operators: {operators_str}

            Observations (input -> output):
            {obs_block}

            Prior expressions generated (avoid repeating any expression\x20
            that is equivalent under the rules below):
            {prior_block}

            Task: Generate a single Boolean expression that is consistent with ALL observations.

            Requirements:
            1. Use ONLY the variables: {', '.join(self.variables)}
            2. Use ONLY these operators: {', '.join(self.operators)}
            3. The expression must match all given observations
            4. Structural uniqueness is judged by these rules:
            {rules_text}
            5. Expression depth should be at most {self.max_depth} levels of nesting
            6. Do not use boolean constants True or False anywhere in the expression.

            Output format:\x20
            - Return ONLY the Boolean expression on a single line
            - Use plain text format (no LaTeX, no markdown, no special formatting)
            - Use uppercase for operators.\x20
            - Use lowercase for variables: x, y
            - Use parentheses for grouping when needed
            - Start your response with "Expression: " followed by the expression
           \x20
            Examples of correct format:
            Expression: x AND y
            Expression: (x OR y) AND NOT x
            Expression: NOT (x AND y)
            Expression: NOR(x, y)
           \x20
            DO NOT use formats like:
            - \\(x \\land y\\)  (LaTeX)
            - `x AND y`  (markdown)
            - x ∧ y  (mathematical symbols)
            """
        return prompt

    def parse_llm_response(self, response: str) -> Optional[str]:
        """Parse LLM response to extract Boolean expression."""
        if not isinstance(response, str):
            return None

        response = response.strip()

        # Clean common formatting issues
        def clean_expression(expr: str) -> str:
            # Remove LaTeX delimiters
            expr = re.sub(r'\\[()\[\]]', '', expr)
            expr = re.sub(r'\$+', '', expr)

            # Remove markdown backticks
            expr = expr.strip('`')

            # Convert LaTeX/math symbols to text
            expr = expr.replace('\\land', 'AND')
            expr = expr.replace('\\lor', 'OR')
            expr = expr.replace('\\lnot', 'NOT')
            expr = expr.replace('\\neg', 'NOT')
            expr = expr.replace('\\wedge', 'AND')
            expr = expr.replace('\\vee', 'OR')
            expr = expr.replace('∧', 'AND')
            expr = expr.replace('∨', 'OR')
            expr = expr.replace('¬', 'NOT')
            expr = expr.replace('⊕', 'XOR')

            # Remove \text{} wrappers
            expr = re.sub(r'\\text\{([^}]+)\}', r'\1', expr)

            # Clean up extra spaces and symbols
            expr = re.sub(r'\*+$', '', expr)  # Remove trailing asterisks
            expr = expr.strip('"\'').rstrip('.,;')

            return expr.strip()

        # Look for "Expression:" line
        lines = response.split('\n')
        for line in lines:
            if 'Expression:' in line or 'expression:' in line:
                parts = re.split(r'[Ee]xpression:', line)
                if len(parts) >= 2:
                    expr = clean_expression(parts[1])
                    if expr:
                        return expr

        # Fallback: try first non-empty line with operators
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                cleaned = clean_expression(line)
                if any(op in cleaned.upper() for op in ['AND','OR','NOT','XOR','NOR']):
                    return cleaned
                if all(c.isalnum() or c in '() ' for c in cleaned):
                    return cleaned

        return None
