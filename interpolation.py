import numpy as np
from numpy.typing import NDArray


class Interpolation:
    """
    Class Interpolation used for interpolation.
    Attributes:
        a: Grid start
        b: Grid end
        n: Grid size
        xn: Grid nodes
        yn: The value of the function in the grid nodes

    Methods:
        interpolation_lagrange: Calculates Lagrange interpolation
        separated_difference: Calculates all the separated differences from start to end
        interpolation_newton: Calculates Newton interpolation
        finite_difference: Calculates all the finite differences from start to end
        interpolation_newton_finite_forward: Calculates Newton forward interpolation for finite differences
        interpolation_newton_finite_backward: Calculates Newton backward interpolation for finite differences
        interpolation_gauss_finite_forward: Calculates Gauss forward interpolation for finite differences
        interpolation_gauss_finite_backward: Calculates Gauss backward interpolation for finite differences
        interpolation_stirling: Calculates Stirling interpolation for finite differences
        interpolation_bessel: Calculates Bessel interpolation for finite differences
        choose_interpolation_method: Selects a method for interpolation
        residual_lagrange_term: calculates the minimum and maximal Lagrange residual term
    """

    def __init__(self, a: float, b: float, n: int, xn: NDArray[float], yn: NDArray[float]):
        self.a = a
        self.b = b
        self.n = n + 1
        self.h = (b - a) / n
        self.xn = xn
        self.yn = yn

    def interpolation_lagrange(self, x: float, m: int) -> float | None:
        """
        Calculates Lagrange interpolation
        :param x: The point at which the polynomial is calculated
        :param m: Degree of the interpolation
        :return: Lagrange polynomial to the point x or none if arguments are incorrect
        """
        if x < self.a or x > self.b or m + 1 > len(self.xn):
            return None
        n0 = int(np.floor((x - self.a) / self.h))

        while n0 + m >= len(self.xn):
            n0 -= 1

        xm = self.xn[n0: n0 + m + 1]
        ym = self.yn[n0: n0 + m + 1]

        result = 0
        for i in range(m + 1):
            ck = np.prod((x - np.delete(xm, i)) / (xm[i] - np.delete(xm, i)))
            result += ym[i] * ck

        return result

    def separated_difference(self, xm: NDArray[float], ym: NDArray[float], start: int, end: int, p: int) \
            -> NDArray[float]:
        """
        Calculates all the separated differences from start to end
        :param xm: Grid notes
        :param ym: The value of the function in these nodes
        :param start: The start node
        :param end: The end node
        :param p: The degree of past separated differences
        :return: A list of all the separated differences in order of increasing degree
        """
        xm_len = len(xm)

        for i in range(start, end):
            f_difference = (ym[i + 1] - ym[i]) / (xm[i + p - start] - xm[i - start])
            ym = np.append(ym, f_difference)

        if end - start > 1:
            ym = self.separated_difference(xm, ym, end + 1, 2 * end - start, p + 1)

        return ym

    def interpolation_newton_separated(self, x: float, m: int) -> float | None:
        """
        Calculates Newton interpolation for separated differences
        :param x: The point at which the polynomial is calculated
        :param m: Degree of the interpolation
        :return: Newton polynomial to the point x or none if arguments are incorrect
        """
        if x < self.a or x > self.b or m + 1 > len(self.xn):
            return None

        n0 = int(np.floor((x - self.a) / self.h))

        while n0 + m >= len(self.xn):
            n0 -= 1

        xm = self.xn[n0: n0 + m + 1]
        ym = self.yn[n0: n0 + m + 1]

        ym = self.separated_difference(xm, ym, 0, m, 1)

        wm = 1
        k = 0
        result = 0

        for i in range(m + 1):
            result += ym[k] * wm
            wm *= (x - xm[i])
            k += m + 1 - i

        return result

    def finite_difference(self, ym: NDArray[float]) \
            -> list[NDArray[float]]:
        """
        Calculates all the finite differences from start to end
        :param ym: The value of the function in these nodes
        :return: A list of all the finite differences in order of increasing degree
        """
        result = [ym]
        for i in range(len(ym) - 1):
            values = result[-1]
            difference = np.array([values[j + 1] - values[j] for j in range(len(values) - 1)])
            result.append(difference)

        return result

    def interpolation_newton_finite_forward(self, x: float, m: int) -> float | None:
        """
        Calculates Newton forward interpolation for finite differences
        :param x: The point at which the polynomial is calculated
        :param m: The index of the closest point to x
        :return: Newton forward polynomial to the point x or none if arguments are incorrect
        """

        if x < self.a or x > self.b:
            return None
        t = (x - self.xn[m]) / self.h
        result = 0
        ck = 1
        ym = self.finite_difference(self.yn[m:])
        n = self.n - 1 - m

        for i in range(n + 1):
            result += ck * ym[i][0]
            ck *= (t - i) / (i + 1)

        return result

    def interpolation_newton_finite_backward(self, x: float, m: int) -> float | None:
        """
        Calculates Newton backward interpolation for finite differences
        :param x: The point at which the polynomial is calculated
        :param m: The index of the closest point to x
        :return: Newton backward to the point x or none if arguments are incorrect
        """
        if x < self.a or x > self.b:
            return None

        t = (x - self.xn[-1]) / self.h
        result = 0
        ck = 1
        n = m
        ym = self.finite_difference(self.yn[:m + 1])

        for i in range(n + 1):
            result += ck * ym[i][m - i]
            ck *= (t + i) / (i + 1)

        return result

    def interpolation_gauss_finite_forward(self, x: float, m: int) -> float | None:
        """
        Calculates Gauss forward interpolation for finite differences
        :param x: The point at which the polynomial is calculated
        :param m: The index of the closest point to x
        :return: Gauss forward polynomial to the point x or none if arguments are incorrect
        """
        if x < self.a or x > self.b:
            return None

        l = min(m, self.n - m - 1)
        n = 2 * l
        t = (x - self.xn[n // 2]) / self.h

        result = 0
        ck = 1

        ym = self.finite_difference(self.yn[m - l:m + l + 1])

        for i in range(n + 1):
            w = (i + 1) // 2
            result += ck * ym[i][n // 2 - w]
            ck *= (t - (-1) ** i * w) / (i + 1)

        return result

    def interpolation_gauss_finite_backward(self, x: float, m: int) -> float | None:
        """
        Calculates Gauss backward interpolation for finite differences
        :param x: The point at which the polynomial is calculated
        :param m: The index of the closest point to x
        :return: Gauss backward polynomial to the point x or none if arguments are incorrect
        """
        if x < self.a or x > self.b:
            return None

        l = min(m, self.n - m - 1)
        n = 2 * l
        t = (x - self.xn[n // 2]) / self.h

        result = 0
        ck = 1

        ym = self.finite_difference(self.yn[m - l:m + l + 1])

        for i in range(n + 1):
            w = (i + 1) // 2
            result += ck * ym[i][n // 2 - i // 2]

            ck *= (t + (-1) ** i * w) / (i + 1)
        return result

    def interpolation_stirling(self, x: float, m: int) -> float | None:
        """
        Calculates Stirling interpolation for finite differences
        :param x: The point at which the polynomial is calculated
        :param m: The index of the closest point to x
        :return: Stirling polynomial to the point x or none if arguments are incorrect
        """
        if x < self.a or x > self.b:
            return None

        l = min(m, self.n - m - 1)
        n = 2 * l
        t = (x - self.xn[n // 2]) / self.h

        result = 0
        ck = 1
        ym = self.finite_difference(self.yn[m - l:m + l + 1])
        yi = ym[0][n // 2]

        for i in range(n + 1):
            w = (i + 1) // 2
            result += ck * yi

            if i == self.n - 1:
                break
            yi = ym[i + 1][n // 2 - w]

            if i % 2 != 0:
                ck *= t
            else:
                yi = (yi + ym[i + 1][n // 2 - w - 1]) / 2
                if i != 0:
                    ck *= (t ** 2 - (i + 1) ** 2) / t
                else:
                    ck *= t

            ck /= (i + 1)
        return result

    def interpolation_bessel(self, x: float, m: int) -> float | None:
        """
        Calculates Bessel interpolation for finite differences
        :param x: The point at which the polynomial is calculated
        :param m: The index of the closest point to x
        :return: Bessel polynomial to the point x or none if arguments are incorrect
        """
        if x < self.a or x > self.b:
            return None

        l = min(m, self.n - m - 1)
        n = 2 * l
        t = (x - self.xn[n // 2]) / self.h

        result = 0
        ck = 1

        ym = self.finite_difference(self.yn[m - l:m + l + 1])

        yi = (ym[0][n // 2] + ym[0][n // 2 + 1]) / 2
        # print(t, 1, n // 2)
        for i in range(n + 1):
            w = (i + 1) // 2
            result += ck * yi

            if i == self.n - 1:
                break
            yi = ym[i + 1][n // 2 - w]

            if i % 2 != 0:
                # print(ym[i], i)
                yi = (yi + ym[i + 1][n // 2 - w + 1]) / 2
                ck *= (t + i // 2) * (t - (i + 1) // 2) / (t - 1 / 2)
            else:
                ck *= t - 1 / 2

            ck /= (i + 1)
            # print(ck, yi)
        return result

    def choose_interpolation_method(self, x) -> float | None:
        """
        Selects a method for interpolation
        :param x: The point at which the polynomial is calculated
        :return: Interpolation to the point x or none if arguments are incorrect
        """
        if x < self.a or x > self.b:
            return None

        n0 = int(np.floor((x - self.a) / self.h))
        result = None
        xm = self.xn[n0]
        t = (x - xm) / self.h
        len_right = self.n - n0 - 1
        len_center = 2 * (min(n0, len_right))

        if len_center > max(n0, len_right):
            if abs(t) <= 0.25 and (len_center + 1) % 2 == 0:
                print("Use Stirling")
                result = self.interpolation_stirling(x, n0)
            elif abs(t) <= 0.5 and (len_center + 1) % 2 == 0:
                print("Use Bessel")
                result = self.interpolation_stirling(x, n0)
            else:
                if h / 2 - (x - xm) < 0:
                    print('Use Gauss 2')
                    result = self.interpolation_gauss_finite_backward(x, n0 + 1)
                else:
                    print('Use Gauss 1')
                    result = self.interpolation_gauss_finite_backward(x, n0)

        elif n0 < len_right:
            print("Use Newton 1")
            result = self.interpolation_newton_finite_forward(x, n0)
        elif n0 > len_right:
            print("Use Newton 2")
            result = self.interpolation_newton_finite_backward(x, n0 + 1)

        return result

    def residual_lagrange_term(self, x: float, m: int, ym_derivative: NDArray[float]) -> tuple[float, float] | None:
        """
        Calculates the minimum and maximal Lagrange residual term
        :param x: The point at which the polynomial is calculated
        :param m: Degree of the interpolation
        :param ym_derivative: The value of the m derivative in nodes
        :return: Tuple of min and max residual term or none if arguments are incorrect
        """
        if x < self.a or x > self.b or m + 1 > len(self.xn):
            return None

        n0 = int(np.floor((x - self.a) / self.h))

        while n0 + m >= len(self.xn):
            n0 -= 1

        xm = self.xn[n0: n0 + m + 1]
        ym = ym_derivative[n0:: n0 + m + 1]

        ck = 1

        for i in range(m + 1):
            ck *= (x - xm[i]) / (i + 1)

        rn = [ck * min(ym), ck * max(ym)]

        return min(rn), max(rn)


def f(x):
    return x ** 2 - np.log10(0.5 * x)


a = 0.5
b = 1
n = 10
h = (b - a) / n

xn = np.arange(a, b + h, h, dtype=float)
yn = np.array([f(x) for x in xn], dtype=float)
print(xn)
# yn_derivative_2 = np.array([f_derivative_2(x) for x in xn], dtype=float)
# yn_derivative_3 = np.array([f_derivative_3(x) for x in xn], dtype=float)
# x_star = 0.98
# y_star = f(x_star)
x_stars = np.array([0.53, 0.98, 0.77])
y_stars = np.array([f(x) for x in x_stars])

inter_f = Interpolation(a, b, n, xn, yn)

for i in range(len(x_stars)):
    print(f"Точное значение f({x_stars[i]}) = {y_stars[i]:}")
    interp = inter_f.choose_interpolation_method(x_stars[i])
    print(f"Интерполяция: {interp} Ошибка: {y_stars[i] - interp:}")
# print(f(0.78), inter_f.interpolation_stirling(0.78, 6))
# print(f(0.78), inter_f.interpolation_bessel(0.78, 6))
# y_linear_lagrange = inter_f.interpolation_lagrange(x_star, 1)
# y_linear_newton = inter_f.interpolation_newton_separated(x_star, 1)
# y_quad_lagrange = inter_f.interpolation_lagrange(x_star, 2)
# y_quad_newton = inter_f.interpolation_newton_separated(x_star, 2)
#
# r1_x_star = y_star - y_linear_lagrange
# r2_x_star = y_star - y_quad_lagrange
# r1 = inter_f.residual_lagrange_term(x_star, 1)
# r2 = inter_f.residual_lagrange_term(x_star, 2)
#
# print(f"Точное значение f({x_star}) = {y_star:.6f}")
# print("\nЛинейная интерполяция:")
# print(f"Лагранж: y = {y_linear_lagrange:.6f}")
# print(f"Ньютон: y = {y_linear_newton:.6f}")
# print(f"min R1 = {r1[0]:.6e}, max R1 = {r1[1]:.6e}, R1(x*) = {r1_x_star:.6e}, \n"
#       f"R1(x*)<=10^-4: {abs(r1_x_star) <= 10 ** (-4)}")
#
# print("\nКвадратичная интерполяция:")
# print(f"Лагранж: y = {y_quad_lagrange:.6f}")
# print(f"Ньютон:   y = {y_quad_newton:.6f}")
# print(f"min R2 = {r2[0]:.6e}, max R2 = {r2[1]:.6e}, R2(x*) = {r2_x_star:.6e},\n"
#       f"R2(x*)<=10^-5: {abs(r2_x_star) <= 10 ** (-5)}")
