import math


# -------------------------------------------------
# Физические формулы
# -------------------------------------------------

def velocity(Q, d):
    return 4.0 * Q / (math.pi * d**2)


def reynolds_number(d, v, rho, eta):
    return d * v * rho / eta


def friction_coefficient(k_e, d, Re):
    if Re <= 0:
        return None
    if Re < 2300:
        return 64.0 / Re
    return 0.11 * ((k_e / d) + (68.0 / Re)) ** 0.25


def resistance_coefficient(lam, d, rho0):
    return 626.1 * rho0 * lam / d**5


def leak_coordinate(P_start, P_end, Q_start, Q_end, L, k):
    denom = Q_start**2 - Q_end**2
    if denom <= 0:
        return None
    return ((P_start - P_end) / k - L * Q_end**2) / denom


def pressure_end_from_true_x(P_start, x_true, Q_start, Q_end, L, k):
    return P_start - k * (x_true * (Q_start**2 - Q_end**2) + L * Q_end**2)