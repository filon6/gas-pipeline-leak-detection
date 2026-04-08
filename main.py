import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch

from dataset import generate_dataset
from train import train_model


# -------------------------------------------------
# Создание директорий
# -------------------------------------------------

os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("result", exist_ok=True)


# -------------------------------------------------
# Основной блок
# -------------------------------------------------

if __name__ == "__main__":

    df = generate_dataset(n_samples=12000, seed=42)

    print("Размер датасета:", df.shape)
    print(df.head())

    # сохранение датасета
    df.to_csv("data/gas_leak_physics_ml_dataset.csv", index=False)

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
        "P_start","P_end","Q_start","Q_end",
        "d","L","rho0","eta","k_e","k",
        "deltaP","deltaQ","deltaQ2",
        "rel_leak","noise_level","leak_fraction"
    ]

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
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=200, lr=1e-3
    )

    # сохранение модели
    torch.save(model.state_dict(), "result/leak_model.pth")

    # -----------------------------
    # Предсказание
    # -----------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred_norm = model(X_test_t).cpu().numpy().flatten()

    # -----------------------------
    # Метрики
    # -----------------------------

    mae_norm = mean_absolute_error(y_test, y_pred_norm)
    rmse_norm = math.sqrt(mean_squared_error(y_test, y_pred_norm))
    r2_norm = r2_score(y_test, y_pred_norm)

    print("\nНейросеть: метрики для x/L")
    print(f"MAE  = {mae_norm:.6f}")
    print(f"RMSE = {rmse_norm:.6f}")
    print(f"R2   = {r2_norm:.6f}")

    # -----------------------------
    # Перевод координаты в метры
    # -----------------------------

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

    # -----------------------------
    # Аналитическая формула на test
    # -----------------------------

    df_train, df_temp = train_test_split(df, test_size=0.30, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=42)

    mae_formula_test = mean_absolute_error(df_test["x_true"], df_test["x_formula"])
    rmse_formula_test = math.sqrt(mean_squared_error(df_test["x_true"], df_test["x_formula"]))
    r2_formula_test = r2_score(df_test["x_true"], df_test["x_formula"])

    print("\nАналитическая формула на test:")
    print(f"MAE  = {mae_formula_test:.3f} м")
    print(f"RMSE = {rmse_formula_test:.3f} м")
    print(f"R2   = {r2_formula_test:.6f}")

    # =================================================
    # ГРАФИКИ
    # =================================================

    # график обучения
    plt.figure(figsize=(8,5))

    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")

    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("График обучения нейросети")

    plt.legend()
    plt.grid(True)

    plt.savefig("figures/training_loss.png", dpi=300)

    plt.show()

    # scatter нейросети

    plt.figure(figsize=(6,6))

    plt.scatter(x_true_m, x_pred_m, alpha=0.5)

    mn = min(x_true_m.min(), x_pred_m.min())
    mx = max(x_true_m.max(), x_pred_m.max())

    plt.plot([mn, mx], [mn, mx])

    plt.xlabel("Истинное x, м")
    plt.ylabel("Предсказанное x, м")
    plt.title("Нейросеть: истинные и предсказанные координаты")

    plt.grid(True)

    plt.savefig("figures/nn_prediction.png", dpi=300)

    plt.show()

    # гистограмма ошибок

    errors_nn = x_pred_m - x_true_m

    plt.figure(figsize=(8,5))

    plt.hist(errors_nn, bins=40)

    plt.xlabel("Ошибка нейросети, м")
    plt.ylabel("Частота")
    plt.title("Гистограмма ошибок нейросети")

    plt.grid(True)

    plt.savefig("figures/error_histogram.png", dpi=300)

    plt.show()

    # scatter формулы

    plt.figure(figsize=(6,6))

    plt.scatter(df_test["x_true"], df_test["x_formula"], alpha=0.5)

    mn = min(df_test["x_true"].min(), df_test["x_formula"].min())
    mx = max(df_test["x_true"].max(), df_test["x_formula"].max())

    plt.plot([mn, mx], [mn, mx])

    plt.xlabel("Истинное x, м")
    plt.ylabel("Формула x, м")
    plt.title("Аналитическая формула: истинные и рассчитанные координаты")

    plt.grid(True)

    plt.savefig("figures/formula_prediction.png", dpi=300)

    plt.show()