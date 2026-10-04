import sys
import os
import unittest

# Make src importable when running this file directly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import calculator


class TestCalculator(unittest.TestCase):

    def test_fun1(self):
        self.assertEqual(calculator.fun1(2, 3), 5)
        self.assertEqual(calculator.fun1(5, 0), 5)
        self.assertEqual(calculator.fun1(-1, 1), 0)
        self.assertEqual(calculator.fun1(-1, -1), -2)

    def test_fun2(self):
        self.assertEqual(calculator.fun2(2, 3), -1)
        self.assertEqual(calculator.fun2(5, 0), 5)
        self.assertEqual(calculator.fun2(-1, 1), -2)
        self.assertEqual(calculator.fun2(-1, -1), 0)

    def test_fun3(self):
        self.assertEqual(calculator.fun3(2, 3), 6)
        self.assertEqual(calculator.fun3(5, 0), 0)
        self.assertEqual(calculator.fun3(-1, 1), -1)
        self.assertEqual(calculator.fun3(-1, -1), 1)

    def test_fun4(self):
        self.assertEqual(calculator.fun4(2, 3, 5), 10)
        self.assertEqual(calculator.fun4(5, 0, -1), 4)
        self.assertEqual(calculator.fun4(-1, -1, -1), -3)
        self.assertEqual(calculator.fun4(-1, -1, 100), 98)

    def test_fun5(self):
        self.assertEqual(calculator.fun5(10, 3, 2), 5)
        self.assertEqual(calculator.fun5(5, 0, 0), 5)
        self.assertEqual(calculator.fun5(-1, -1, -1), 1)
        self.assertAlmostEqual(calculator.fun5(2.5, 1, 0.5), 1)

    def test_fun6(self):
        self.assertEqual(calculator.fun6(6, 3), 2)
        self.assertAlmostEqual(calculator.fun6(5, 2), 2.5)
        self.assertEqual(calculator.fun6(-6, 3), -2)
        self.assertEqual(calculator.fun6(0, 5), 0)

    def test_fun6_division_by_zero(self):
        with self.assertRaisesRegex(ValueError, "Cannot divide by zero"):
            calculator.fun6(5, 0)

    def test_invalid_inputs(self):
        for function in (
            calculator.fun1,
            calculator.fun2,
            calculator.fun3,
            calculator.fun6,
        ):
            for x, y in [("hello", 2), (2, None)]:
                with self.subTest(function=function.__name__, x=x, y=y):
                    with self.assertRaisesRegex(
                        ValueError, "Both inputs must be numbers"
                    ):
                        function(x, y)


if __name__ == "__main__":
    unittest.main()