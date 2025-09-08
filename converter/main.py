import ast, sys, os, json
from typing import cast

def is_comment(n):
    return type(n) is ast.Expr and type(n.value) is ast.Constant

class Parsed:
    imports = []
    functions = []

    def __init__(self, imports, functions):
        self.imports = imports
        self.functions = functions

    def to_notebook(self):
        def _make_cell(type, source):
            return {
                "metadata": {},
                "cell_type": type,
                "source": [f"{line}\n" for line in source.split("\n")]
            }

        cells = []

        cells.append(_make_cell("code", "#Load libraries\n" + "\n".join([ast.unparse(x) for x in self.imports])))

        for docs, func in self.functions:
            cells.append(_make_cell(
                "markdown",
                cast(ast.Constant, cast(ast.Expr, docs).value).value))
            cells.append(_make_cell(
                "code",
                "\n".join([ast.unparse(x) for x in cast(ast.FunctionDef, func).body])))

        return {
            "metadata": {
                "kernelspec": {
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 0,
            "cells": cells 
        }



def parse(path):

    try:
        with open(path) as f:
            code = f.read()
        tree = ast.parse(code)
    except:
        print("Failed to parse code")
        exit(1)

    imports = [x for x in tree.body if type(x) is ast.Import or type(x) is ast.ImportFrom]
    functions = [
        (tree.body[i - 1], x)
        for i, x in enumerate(tree.body)
        if type(x) is ast.FunctionDef \
        and i > 0 \
        and is_comment(tree.body[i - 1])
    ]

    return Parsed(imports, functions)

def main():
    if len(sys.argv) < 3:
        print("Usage: converter <code> <notebook>")
        exit(1)

    path = sys.argv[1]
    if not os.path.exists(path) or not os.path.isfile(path):
        print(f"File {path} not found")
        exit(1)

    parsed = parse(path)

    try:
        with open(sys.argv[2], "w") as f:
            f.write(json.dumps(parsed.to_notebook()))
    except:
        print("Failed to create notebook")
        exit(1)

if __name__ == "__main__":
    main()
