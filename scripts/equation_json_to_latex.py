"""Convert exported EQN JSON graphs to LaTeX using only the Python standard library.

Examples:
  python scripts/equation_json_to_latex.py path/to/equations
  python scripts/equation_json_to_latex.py equation.json --document --precision 6
  python scripts/equation_json_to_latex.py equation.json --expanded --select reward

By default, writes a .tex file beside each JSON, retaining shared intermediate
terms and full coefficient precision. --expanded substitutes those terms; large
expressions are bounded by --max-chars. --document adds an amsmath preamble.
"""

import argparse
import json
import math
import re
from pathlib import Path

_ARITY = {
    "sin": 1,
    "cos": 1,
    "sigmoid": 1,
    "log": 1,
    "square": 1,
    "add": 2,
    "subtract": 2,
    "divide": 2,
    "maximum": 2,
}


def _finite(value):
    """Return whether a value is a finite built-in integer or float.

    Args:
        value (object): Candidate numeric graph attribute; only finite built-in int and float values
            are accepted.

    Returns:
        bool: Whether value is a finite built-in integer or float; booleans and nonnumeric values
            are rejected.
    """
    return type(value) in (int, float) and math.isfinite(value)


def validate_graph(graph):
    """Require the version-1, topologically ordered graph produced by EQN export.

    Args:
        graph (dict): Exported equation graph containing inputs, nodes, named outputs, and metadata.

    Returns:
        None.

    Raises:
        ValueError: Expected a version-1 equation graph. inputs must be a list of names. Expected
            nonempty nodes and outputs. Output references an invalid node.
    """
    if not isinstance(graph, dict) or graph.get("version") != 1:
        raise ValueError("Expected a version-1 equation graph")
    inputs, nodes, outputs = (graph.get(key) for key in ("inputs", "nodes", "outputs"))
    if not isinstance(inputs, list) or not all(isinstance(name, str) for name in inputs):
        raise ValueError("inputs must be a list of names")
    if not isinstance(nodes, list) or not nodes or not isinstance(outputs, dict) or not outputs:
        raise ValueError("Expected nonempty nodes and outputs")
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise TypeError(f"Node {index} must be an object")
        op, args = node.get("op"), node.get("args", [])
        if not isinstance(args, list) or any(
            type(i) is not int or not 0 <= i < index for i in args
        ):
            raise ValueError(f"Node {index}: arguments must refer to earlier nodes")
        if op == "input":
            valid = not args and type(node.get("index")) is int and 0 <= node["index"] < len(inputs)
        elif op == "constant":
            valid = not args and _finite(node.get("value"))
        elif op == "affine":
            weights = node.get("weights")
            valid = (
                bool(args)
                and isinstance(weights, list)
                and len(weights) == len(args)
                and all(_finite(w) for w in weights)
                and _finite(node.get("bias"))
            )
        else:
            valid = isinstance(op, str) and op in _ARITY and len(args) == _ARITY[op]
        if not valid:
            raise ValueError(f"Node {index}: invalid or unsupported operator {op!r}")
    if any(type(i) is not int or not 0 <= i < len(nodes) for i in outputs.values()):
        raise ValueError("Output references an invalid node")


def _escape(text):
    """Escape LaTeX special characters and replace line breaks with spaces.

    Args:
        text (str): Text to escape for safe LaTeX rendering.

    Returns:
        str: LaTeX-escaped text with newline characters replaced by spaces.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    return "".join(replacements.get(c, c) for c in text.replace("\n", " ").replace("\r", " "))


def _input_name(name):
    """Map an input name to a familiar mathematical symbol or escaped LaTeX text.

    Args:
        name (str): Exported equation input label, formatted as a mathematical coordinate or escaped
            text.

    Returns:
        str: Mathematical coordinate notation for recognized input names, otherwise safely escaped
            text.
    """
    pendulum = {
        "cos_theta": r"\cos(\theta)",
        "sin_theta": r"\sin(\theta)",
        "theta_dot": r"\dot{\theta}",
        "absorbing_indicator": r"z_{\mathrm{abs}}",
    }
    if name in pendulum:
        return pendulum[name]
    match = re.fullmatch(r"(obs|action|x)_(\d+)", name)
    if match:
        symbol = {"obs": "s", "action": "a", "x": "x"}[match[1]]
        return f"{symbol}_{{{match[2]}}}"
    return r"\text{\texttt{" + _escape(name) + "}}"


def _number(value, precision):
    """Format a coefficient as LaTeX, optionally rounding its significant digits.

    Args:
        value (float or int): Finite coefficient to express in decimal or scientific notation.
        precision (int or None): Significant-digit limit; None preserves the unrounded numeric
            string.

    Returns:
        str: Finite numeric coefficient formatted to the selected precision, using LaTeX scientific
            notation when necessary.
    """
    text = repr(value) if precision is None else format(value, f".{precision}g")
    if "e" in text.lower():
        mantissa, exponent = text.lower().split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    return text


def _affine_terms(node, args, number):
    """Format nonzero affine coefficients and bias as signed LaTeX terms.

    Args:
        node (dict): Equation graph node being formatted or inspected.
        args (sequence[str]): Already formatted LaTeX expressions for the node operands.
        number (callable): Formatter converting numeric coefficients into LaTeX text.

    Returns:
        list[str]: Signed affine-term fragments in display order, omitting zero coefficients.
    """
    terms = [
        (w < 0, rf"{number(abs(w))}\,\left({arg}\right)")
        for w, arg in zip(node["weights"], args)
        if w != 0
    ]
    if node["bias"] != 0 or not terms:
        terms.append((node["bias"] < 0, number(abs(node["bias"]))))
    return [
        (" - " if negative else (" + " if i else "")) + term
        for i, (negative, term) in enumerate(terms)
    ]


def _expression(node, args, number):
    """Render one supported equation graph operation as LaTeX.

    Args:
        node (dict): Equation graph node being formatted or inspected.
        args (sequence[str]): Already formatted LaTeX expressions for the node operands.
        number (callable): Formatter converting numeric coefficients into LaTeX text.

    Returns:
        str: LaTeX expression for the supplied node and already-formatted operands.
    """
    op = node["op"]
    if op == "affine":
        return "".join(_affine_terms(node, args, number))
    if op in ("sin", "cos", "log"):
        function = "ln" if op == "log" else op
        return rf"\{function}\left({args[0]}\right)"
    if op == "sigmoid":
        return rf"\frac{{1}}{{1+\exp\left(-\left({args[0]}\right)\right)}}"
    if op == "square":
        return rf"\left({args[0]}\right)^{{2}}"
    if op in ("add", "subtract"):
        symbol = "+" if op == "add" else "-"
        return rf"\left({args[0]}\right) {symbol} \left({args[1]}\right)"
    if op == "divide":
        return rf"\frac{{{args[0]}}}{{{args[1]}}}"
    if op == "maximum":
        return rf"\max\left\{{{args[0]},\,{args[1]}\right\}}"
    raise ValueError(f"Unsupported operator: {op}")


def graph_to_latex(
    graph, *, precision=None, expanded=False, select=None, document=False, max_chars=200000
):
    """Render equations without changing operators, thresholds or graph structure.

    Args:
        graph (dict): Exported equation graph containing inputs, nodes, named outputs, and metadata.
        precision (int or None): Optional significant-digit limit for coefficients; None avoids
            coefficient rounding.
        expanded (bool): Expand shared graph terms instead of using intermediate definitions.
        select (str or None): Named equation output to render.
        document (bool): Wrap the generated equations in a complete LaTeX document.
        max_chars (int): Maximum length allowed for a rendered expanded expression.

    Returns:
        str: Selected equation rendered as LaTeX, optionally including shared intermediate
            definitions or a complete document.

    Raises:
        ValueError: precision must be between 1 and 17 significant digits. max_chars must be
            positive. Expanded expression exceeds --max-chars; use shared terms (default). LaTeX
            exceeds --max-chars; select one output or use shared terms.
    """
    validate_graph(graph)
    if precision is not None and not 1 <= precision <= 17:
        raise ValueError("precision must be between 1 and 17 significant digits")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    outputs = graph["outputs"]
    if select is not None:
        if select not in outputs:
            raise ValueError(f"Unknown output {select!r}; choose from {list(outputs)}")
        outputs = {select: outputs[select]}
    needed = set(outputs.values())
    for index in reversed(range(len(graph["nodes"]))):
        if index in needed:
            needed.update(graph["nodes"][index].get("args", []))
    number = lambda value: _number(value, precision)
    expressions, rows = {}, []
    for index, node in enumerate(graph["nodes"]):
        if index not in needed:
            continue
        op = node["op"]
        if op == "input":
            expressions[index] = _input_name(graph["inputs"][node["index"]])
        elif op == "constant":
            expressions[index] = number(node["value"])
        else:
            args = [expressions[i] for i in node["args"]]
            expression = _expression(node, args, number)
            if len(expression) > max_chars:
                raise ValueError(
                    "Expanded expression exceeds --max-chars; use shared terms (default)"
                )
            if expanded:
                expressions[index] = expression
            else:
                # Three affine terms per row keep the large learned networks readable.
                parts = _affine_terms(node, args, number) if op == "affine" else [expression]
                chunks = ["".join(parts[i : i + 3]) for i in range(0, len(parts), 3)]
                rows.append(rf"t_{{{index}}} &= {chunks[0]}")
                rows.extend(r"&\quad " + chunk for chunk in chunks[1:])
                expressions[index] = rf"t_{{{index}}}"
    for name, index in outputs.items():
        label = r"\text{\texttt{" + _escape(name) + "}}"
        relation = r"\approx" if precision is not None else "="
        rows.append(rf"{label} &{relation} {expressions[index]}")
    note = (
        "% Full stored coefficient precision; no numerical simplification.\n"
        if precision is None
        else f"% DISPLAY APPROXIMATION: coefficients rounded to {precision} significant digits.\n"
    )
    body = note + "\\begin{align*}\n" + " \\\\\n".join(rows) + "\n\\end{align*}\n"
    if len(body) > max_chars:
        raise ValueError("LaTeX exceeds --max-chars; select one output or use shared terms")
    if document:
        body = (
            "\\documentclass{article}\n\\usepackage[margin=1in]{geometry}\n"
            "\\usepackage{amsmath}\n\\allowdisplaybreaks\n\\begin{document}\n"
            "\\small\n" + body + "\\end{document}\n"
        )
    return body


def main():
    """Convert equation JSON files to LaTeX using the selected formatting options.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Export JSON files or equation directories"
    )
    parser.add_argument("-o", "--output", type=Path, help="Output .tex path (one JSON input only)")
    parser.add_argument(
        "--precision",
        type=int,
        help="Display rounding, 1–17 significant digits; default: full precision",
    )
    parser.add_argument(
        "--expanded", action="store_true", help="Substitute all intermediate expressions"
    )
    parser.add_argument(
        "--select", help="Export one output, e.g. reward, probability or raw_output"
    )
    parser.add_argument(
        "--document", action="store_true", help="Include a compilable LaTeX document preamble"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=200000,
        help="Maximum formula size (default: 200000 characters)",
    )
    args = parser.parse_args()
    files = []
    for path in args.paths:
        found = sorted(path.glob("*.json")) if path.is_dir() else [path]
        if not found:
            parser.error(f"No JSON files in {path}")
        files.extend(found)
    files = list(dict.fromkeys(path.resolve() for path in files))
    if args.output and len(files) != 1:
        parser.error("--output requires exactly one JSON input")
    root = Path(__file__).resolve().parents[1]
    prepared = []
    for source in files:
        target = (args.output or source.with_suffix(".tex")).resolve()
        if target == source:
            parser.error("Output must not overwrite the input graph")
        if not target.is_relative_to(root) or target.suffix != ".tex":
            parser.error("Output must be a .tex file inside this repository")
        try:
            graph = json.loads(source.read_text(encoding="utf-8"))
            latex = graph_to_latex(
                graph,
                precision=args.precision,
                expanded=args.expanded,
                select=args.select,
                document=args.document,
                max_chars=args.max_chars,
            )
        except (OSError, TypeError, ValueError) as error:
            parser.error(f"{source}: {error}")
        prepared.append((target, latex))
    for target, latex in prepared:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(latex, encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()
