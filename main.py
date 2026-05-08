import os
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch

from dataset import generate_dataset
from train import train_model


# директории
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("result", exist_ok=True)


if __name__ == "__main__":

    df = generate_dataset(n_samples=12000, seed=42)

    print("Размер датасета:", df.shape)
    print(df.head())

    df.to_csv("data/dataset.csv", index=False)

    # =============================
    # baseline формула
    # =============================

    mae_formula = mean_absolute_error(df["x_true"], df["x_formula"])
    rmse_formula = math.sqrt(mean_squared_error(df["x_true"], df["x_formula"]))
    r2_formula = r2_score(df["x_true"], df["x_formula"])

    print("\nФормула:")
    print(f"MAE  = {mae_formula:.3f} м")
    print(f"RMSE = {rmse_formula:.3f} м")
    print(f"R2   = {r2_formula:.6f}")

    # =============================
    # НОВОЕ: относительная модель
    # =============================

    df["correction_rel"] = (df["x_true"] - df["x_formula"]) / df["L"]
    df["x_formula_norm"] = df["x_formula"] / df["L"]

    feature_cols = [
        "P_start","P_end","Q_start","Q_end",
        "d","L","rho0","eta","k_e","k",
        "deltaP","deltaQ","deltaQ2",
        "rel_leak","noise_level","leak_fraction",
        "x_formula_norm"
    ]

    X = df[feature_cols].values
    y = df["correction_rel"].values

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
        epochs=200
    )

    torch.save(model.state_dict(), "result/model.pth")

    # =============================
    # Предсказание
    # =============================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        correction_rel_pred = model(X_test_t).cpu().numpy().flatten()

    # восстановление координаты
    L_idx = feature_cols.index("L")
    L_test = X_test[:, L_idx]

    # важно: берём те же индексы
    df_train, df_temp = train_test_split(df, test_size=0.30, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=42)

    x_formula_test = df_test["x_formula"].values
    x_true_m = df_test["x_true"].values

    correction_pred = correction_rel_pred * L_test
    x_pred_m = x_formula_test + correction_pred

    # =============================
    # метрики
    # =============================

    mae = mean_absolute_error(x_true_m, x_pred_m)
    rmse = math.sqrt(mean_squared_error(x_true_m, x_pred_m))
    r2 = r2_score(x_true_m, x_pred_m)

    print("\nГибридная модель:")
    print(f"MAE  = {mae:.3f} м")
    print(f"RMSE = {rmse:.3f} м")
    print(f"R2   = {r2:.6f}")

    # =============================
    # ГРАФИКИ
    # =============================

    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.grid()
    plt.title("Обучение")
    plt.savefig("figures/train.png", dpi=300)
    plt.show()

    plt.figure(figsize=(6,6))
    plt.scatter(x_true_m, x_pred_m, alpha=0.5)
    mn = min(x_true_m.min(), x_pred_m.min())
    mx = max(x_true_m.max(), x_pred_m.max())
    plt.plot([mn, mx], [mn, mx])
    plt.grid()
    plt.title("Гибридная модель")
    plt.savefig("figures/hybrid.png", dpi=300)
    plt.show()

    errors = x_pred_m - x_true_m

    plt.figure(figsize=(8,5))
    plt.hist(errors, bins=40)
    plt.grid()
    plt.title("Ошибки")
    plt.savefig("figures/errors.png", dpi=300)
    plt.show()

    plt.figure(figsize=(6, 6))

    plt.scatter(df_test["x_true"], df_test["x_formula"], alpha=0.5)

    mn = min(df_test["x_true"].min(), df_test["x_formula"].min())
    mx = max(df_test["x_true"].max(), df_test["x_formula"].max())

    plt.plot([mn, mx], [mn, mx])

    plt.xlabel("Истинное x, м")
    plt.ylabel("Формула x, м")
    plt.title("Аналитическая формула")

    plt.grid(True)

    plt.savefig("figures/formula.png", dpi=300)
    plt.show()