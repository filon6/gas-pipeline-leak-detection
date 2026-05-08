import random
import numpy as np
import pandas as pd


from physics import*
# -------------------------------------------------
# Генерация физически согласованного датасета
# -------------------------------------------------

def generate_dataset(n_samples=12000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    rows = []

    while len(rows) < n_samples:
        # -----------------------------
        # Параметры трубы
        # -----------------------------
        d = random.uniform(0.3, 0.8)               # м
        L = random.uniform(200.0, 1000.0)          # м
        rho0 = random.uniform(0.65, 0.85)          # кг/м^3
        eta = random.uniform(1.2e-5, 1.8e-5)       # Па*с
        k_e = random.uniform(1e-5, 3e-5)
        # -----------------------------
        # Режим работы
        # -----------------------------
        Q_start_true = random.uniform(1.0, 5.0)    # м^3/с

        # Размер утечки
        leak_fraction = random.uniform(0.02, 0.20)
        Q_end_true = Q_start_true * (1.0 - leak_fraction)

        # Истинная координата утечки
        x_true = random.uniform(0.0, L)

        # Давление на входе
        P_start_true = random.uniform(3.0e6, 7.0e6)   # Па

        # -----------------------------
        # Физическая модель потока
        # -----------------------------
        v_start = velocity(Q_start_true, d)
        Re_start = reynolds_number(d, v_start, rho0, eta)
        lam = friction_coefficient(k_e, d, Re_start)

        if lam is None or lam <= 0:
            continue

        k = resistance_coefficient(lam, d, rho0)

        # Давление на выходе из физической модели
        P_end_true = pressure_end_from_true_x(
            P_start_true, x_true, Q_start_true, Q_end_true, L, k
        )

        # -----------------------------
        # Ограничения существования решения
        # -----------------------------
        if Q_end_true >= Q_start_true:
            continue

        if Q_start_true**2 - Q_end_true**2 <= 0:
            continue

        if P_end_true <= 0:
            continue

        if (P_start_true - P_end_true) <= k * L * Q_end_true**2:
            continue

        if not (0.0 <= x_true <= L):
            continue

        # -----------------------------
        # Повторяемость экспериментов и шум датчиков
        # -----------------------------
        noise_level = random.uniform(0.0, 0.01)   # 0%...1%

        P_start_meas = P_start_true * (1.0 + random.uniform(-noise_level, noise_level))
        P_end_meas   = P_end_true   * (1.0 + random.uniform(-noise_level, noise_level))
        Q_start_meas = Q_start_true * (1.0 + random.uniform(-noise_level, noise_level))
        Q_end_meas   = Q_end_true   * (1.0 + random.uniform(-noise_level, noise_level))

        # Проверки после шума
        if Q_start_meas <= Q_end_meas:
            continue
        if P_start_meas <= P_end_meas:
            continue
        if Q_start_meas**2 - Q_end_meas**2 <= 0:
            continue
        if (P_start_meas - P_end_meas) <= k * L * Q_end_meas**2:
            continue

        # Аналитическая оценка координаты по шумным измерениям
        x_formula = leak_coordinate(
            P_start_meas, P_end_meas, Q_start_meas, Q_end_meas, L, k
        )

        if x_formula is None:
            continue

        if not (0.0 <= x_formula <= L):
            continue

        # Дополнительные информативные признаки
        deltaP = P_start_meas - P_end_meas
        deltaQ = Q_start_meas - Q_end_meas
        deltaQ2 = Q_start_meas**2 - Q_end_meas**2
        rel_leak = deltaQ / Q_start_meas

        rows.append({
            "d": d,
            "L": L,
            "rho0": rho0,
            "eta": eta,
            "k_e": k_e,

            "Q_start": Q_start_meas,
            "Q_end": Q_end_meas,
            "P_start": P_start_meas,
            "P_end": P_end_meas,

            "v_start": v_start,
            "Re_start": Re_start,
            "lambda": lam,
            "k": k,

            "deltaP": deltaP,
            "deltaQ": deltaQ,
            "deltaQ2": deltaQ2,
            "rel_leak": rel_leak,
            "noise_level": noise_level,
            "leak_fraction": leak_fraction,

            "x_formula": x_formula,
            "x_true": x_true,
            "x_norm": x_true / L
        })

    return pd.DataFrame(rows)
