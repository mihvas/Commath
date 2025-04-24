import numpy as np
from numpy.typing import NDArray


class Differentiation:
    """
    Class Differentiation used for differentiation.
    Attributes:
        a: Grid start
        b: Grid end
        n: Grid size
        xn: Grid nodes
        yn: The value of the function in the grid nodes

    Methods:
        interpolation_lagrange: Calculates Lagrange interpolation
        differentiation_polynom: Calculates differentiation polynomial
        differentiation_lagrange: Calculates Lagrange differentiation
    """

    def __init__(self, a: float, b: float, n: int, xn: NDArray[float], yn: NDArray[float]):
        self.a = a
        self.b = b
        self.n = n + 1
        self.h = (b - a) / n
        self.xn = xn
        self.yn = yn

    def differentiation_polynom(self, result: float, xm: NDArray[float], x: float, k: int)->float:
        """
        Calculates differentiation polynomial((x-x0)(x-x1)...)
        :param result: The result of the function
        :param x: The point at which the differentiation is calculated
        :param xm: The point of the polynomial
        :param k: Degree of the derivative
        :return: The k-derivative polynomial to the point x or none if arguments are incorrect
        """
        if k == 0:
            if len(xm) == 0:

                return 1

            return float(np.prod(x - xm))
        elif k > 0 and len(xm) == 0:
            return 0

        summ = 0
        for i in range(len(xm)):
            summ += self.differentiation_polynom(result, np.delete(xm, i), x, k - 1)

        return result+summ

    def differentiation_lagrange(self, x: float, m: int, k: int) -> float | None:
        """
        Calculates Lagrange differentiation
        :param x: The point at which the differentiation is calculated
        :param m: Degree of the interpolation
        :param k: Degree of the derivative
        :return: Lagrange differentiation to the point x or none if arguments are incorrect
        """
        if x < self.a or x > self.b+0.001 or m + 1 > len(self.xn):
            return None
        n0 = int(np.floor((x - self.a) / self.h))

        while n0 + m >= len(self.xn):
            n0 -= 1

        xm = self.xn[n0: n0 + m+1]
        ym = self.yn[n0: n0 + m+1]
        result = 0

        for i in range(m+1):
            cd = self.differentiation_polynom(0, np.delete(xm, i), x, k)

            ck = np.prod((xm[i] - np.delete(xm, i)))
            result += ym[i] * cd/ ck

        return result


def f(x):
    return x ** 2 - np.log10(0.5 * x)


def f_1(x):
    return 2 * x - 1 / (x * np.log(10))


def f_2(x):
    return 2 + 1/ (x ** 2 * np.log(10))


a = 0.5
b = 1
n = 10
h = (b - a) / n

xn = np.arange(a, b + h, h, dtype=float)
yn = np.array([f(x) for x in xn], dtype=float)
diff_f = Differentiation(a, b, n, xn, yn)

for i in xn:
    print(f"Точное значение первой производной в {i}: {f_1(i)},"
          f"\nПолученная первая производна в {i}: {diff_f.differentiation_lagrange(i,10,1)}\n")

    print(
        f"Точное значение вторая производная в {i}: {f_2(i)},"
        f"\nПолученная вторая производная в {i}: {diff_f.differentiation_lagrange(i, 10, 2)}\n")