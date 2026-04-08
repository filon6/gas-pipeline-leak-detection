import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim


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
        k_e = random.uniform(3e-5, 2e-4)           # м

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


# -------------------------------------------------
# Нейросеть
# -------------------------------------------------

class LeakNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()   # т.к. предсказываем x_norm в [0, 1]
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------
# Обучение модели
# -------------------------------------------------

def train_model(X_train, y_train, X_val, y_val, epochs=200, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeakNet(X_train.shape[1]).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred_train = model(X_train_t)
        loss_train = criterion(pred_train, y_train_t)

        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t)
            loss_val = criterion(pred_val, y_val_t)

        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"train_loss={loss_train.item():.6f} | "
                f"val_loss={loss_val.item():.6f}"
            )

    return model, train_losses, val_losses


# -------------------------------------------------
# Основной блок
# -------------------------------------------------

if __name__ == "__main__":
    df = generate_dataset(n_samples=12000, seed=42)

    print("Размер датасета:", df.shape)
    print(df.head())

    df.to_csv("gas_leak_physics_ml_dataset.csv", index=False)

    # -----------------------------
    # Baseline: аналитическая формула
    # -----------------------------
    mae_formula = mean_absolute_error(df["x_true"], df["x_formula"])
    rmse_formula = math.sqrt(mean_squared_error(df["x_true"], df["x_formula"]))
    r2_formula = r2_score(df["x_true"], df["x_formula"])

    print("\nАналитическая формула по шумным измерениям:")
    print(f"MAE  = {mae_formula:.3f} м")
    print(f"RMSE = {rmse_formula:.3f} м")
    print(f"R2   = {r2_formula:.6f}")

    # -----------------------------
    # Признаки для нейросети
    # -----------------------------
    feature_cols = [
        "P_start",
        "P_end",
        "Q_start",
        "Q_end",
        "d",
        "L",
        "rho0",
        "eta",
        "k_e",
        "k",
        "deltaP",
        "deltaQ",
        "deltaQ2",
        "rel_leak",
        "noise_level",
        "leak_fraction"
    ]

    # Цель: нормированная координата
    target_col = "x_norm"

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model, train_losses, val_losses = train_model(
        X_train_scaled, y_train, X_val_scaled, y_val,
        epochs=200, lr=1e-3
    )

    # -----------------------------
    # Предсказание нейросети
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred_norm = model(X_test_t).cpu().numpy().flatten()

    # Метрики в нормированных координатах
    mae_norm = mean_absolute_error(y_test, y_pred_norm)
    rmse_norm = math.sqrt(mean_squared_error(y_test, y_pred_norm))
    r2_norm = r2_score(y_test, y_pred_norm)

    print("\nНейросеть: метрики для x/L")
    print(f"MAE  = {mae_norm:.6f}")
    print(f"RMSE = {rmse_norm:.6f}")
    print(f"R2   = {r2_norm:.6f}")

    # Перевод обратно в метры
    # L находится в X_test, индекс признака "L"
    L_idx = feature_cols.index("L")
    L_test = X_test[:, L_idx]

    x_true_m = y_test * L_test
    x_pred_m = y_pred_norm * L_test

    mae_m = mean_absolute_error(x_true_m, x_pred_m)
    rmse_m = math.sqrt(mean_squared_error(x_true_m, x_pred_m))
    r2_m = r2_score(x_true_m, x_pred_m)

    print("\nНейросеть: метрики для координаты в метрах")
    print(f"MAE  = {mae_m:.3f} м")
    print(f"RMSE = {rmse_m:.3f} м")
    print(f"R2   = {r2_m:.6f}")

    # Для сравнения аналитической формулы на том же test
    df_train, df_temp = train_test_split(df, test_size=0.30, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=42)

    mae_formula_test = mean_absolute_error(df_test["x_true"], df_test["x_formula"])
    rmse_formula_test = math.sqrt(mean_squared_error(df_test["x_true"], df_test["x_formula"]))
    r2_formula_test = r2_score(df_test["x_true"], df_test["x_formula"])

    print("\nАналитическая формула на test:")
    print(f"MAE  = {mae_formula_test:.3f} м")
    print(f"RMSE = {rmse_formula_test:.3f} м")
    print(f"R2   = {r2_formula_test:.6f}")

    # -----------------------------
    # Графики
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("График обучения нейросети")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(x_true_m, x_pred_m, alpha=0.5)
    mn = min(x_true_m.min(), x_pred_m.min())
    mx = max(x_true_m.max(), x_pred_m.max())
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Истинное x, м")
    plt.ylabel("Предсказанное x, м")
    plt.title("Нейросеть: истинные и предсказанные координаты")
    plt.grid(True)
    plt.show()

    errors_nn = x_pred_m - x_true_m
    plt.figure(figsize=(8, 5))
    plt.hist(errors_nn, bins=40)
    plt.xlabel("Ошибка нейросети, м")
    plt.ylabel("Частота")
    plt.title("Гистограмма ошибок нейросети")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(df_test["x_true"], df_test["x_formula"], alpha=0.5)
    mn = min(df_test["x_true"].min(), df_test["x_formula"].min())
    mx = max(df_test["x_true"].max(), df_test["x_formula"].max())
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Истинное x, м")
    plt.ylabel("Формула x, м")
    plt.title("Аналитическая формула: истинные и рассчитанные координаты")
    plt.grid(True)
    plt.show()