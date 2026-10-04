# Lab 01 — Python Testing and GitHub Actions

**Course:** IE7374 MLOps  
**Author:** Zoya Arief

## Overview

This lab covers creating a Python virtual environment, organizing code
in GitHub, writing calculator functions, testing with pytest and unittest,
and automating tests with GitHub Actions.

## Calculator functions

| Function | Operation |
| --- | --- |
| `fun1(x, y)` | Addition |
| `fun2(x, y)` | Subtraction |
| `fun3(x, y)` | Multiplication |
| `fun4(x, y, z)` | Sum of three numbers |
| `fun5(x, y, z)` | Subtract y and z from x |
| `fun6(x, y)` | Division with input and zero-divisor validation |

`fun5` and `fun6` are my two additional functions.

## Local setup

From the repository root:

```bash
cd lab_01
python3 -m venv .venv
source .venv/bin/activate
./.venv/bin/python -m pip install -r requirements.txt
```

## Run tests

From inside `lab_01`:

```bash
./.venv/bin/python -m pytest test/test_pytest.py -v
./.venv/bin/python -m unittest test.test_unittest -v
```

Both test suites contain eight tests covering arithmetic results,
invalid inputs, and division by zero.

## GitHub Actions

The workflows are stored at the repository root:

- `.github/workflows/lab_01_pytest_action.yml`
- `.github/workflows/lab_01_unittest_action.yml`

The pytest workflow also uploads an XML test report as the
`test-results` artifact.

Both workflows passed after the initial implementation was pushed.

## Reference

Based on the instructor's Lab 1:
https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab1