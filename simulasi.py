import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# 1. LOAD DATASET

df = pd.read_csv("population_by_year_2014_2024.csv")

# Deteksi otomatis kolom negara
country_col = df.columns[0]
country = df[country_col].iloc[0]

print("Negara:", country)

# Tahun 2014–2024
years = list(range(2014, 2025))

# Ambil baris negara pertama
row = df.iloc[0]

# Bersihkan angka populasi → float
P = [float(str(row[str(y)]).replace(",", "").strip()) for y in years]

# Ubah menjadi array
t_obs = np.array(years) - years[0]   # t mulai dari 0
P_obs = np.array(P)


# 2. MODEL LOGISTIK: dP/dt = rP(1 - P/K)
def f_logistic(t, P, r, K):
    return r * P * (1 - P / K)


# 3. RUNGE–KUTTA ORDE 4

def rk4_step(func, t, y, h, r, K):
    k1 = func(t, y, r, K)
    k2 = func(t + 0.5*h, y + 0.5*h*k1, r, K)
    k3 = func(t + 0.5*h, y + 0.5*h*k2, r, K)
    k4 = func(t + h, y + h*k3, r, K)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def rk4_solver(func, t_span, y0, h, r, K):
    t0, t_end = t_span
    ts = np.arange(t0, t_end + h, h)
    ys = np.zeros(len(ts))
    ys[0] = y0

    for i in range(len(ts) - 1):
        ys[i+1] = rk4_step(func, ts[i], ys[i], h, r, K)

    return ts, ys


# 4. ESTIMASI PARAMETER r DAN K (Curve Fitting)

def logistic_analytic(t, r, K):
    P0 = P_obs[0]
    A = (K - P0) / P0
    return K / (1 + A * np.exp(-r * t))

popt, _ = curve_fit(logistic_analytic, t_obs, P_obs, p0=[0.02, max(P_obs)*1.2], maxfev=5000)
r_fit, K_fit = popt

print("Estimasi r :", r_fit)
print("Estimasi K :", K_fit)


# 5. SIMULASI RK4

h = 0.1  # step RK4
ts, ys = rk4_solver(f_logistic, (0, t_obs[-1]), P_obs[0], h, r_fit, K_fit)

# Interpolasi untuk tahun yang sama agar bisa dibandingkan
ys_at_obs = np.interp(t_obs, ts, ys)


# 6. EVALUASI ERROR

rmse = np.sqrt(np.mean((ys_at_obs - P_obs)**2))
print("RMSE :", rmse)


# 7. VISUALISASI

plt.figure(figsize=(10, 6))
plt.scatter(years, P_obs, color='black', label='Data Observasi')
plt.plot(years[0] + ts, ys, color='red', label='Simulasi RK4')
plt.title(f"Model Logistik + RK4 untuk Negara: {country}")
plt.xlabel("Tahun")
plt.ylabel("Populasi")
plt.grid(True)
plt.legend()
plt.show()


# 8. PLOT ERROR

plt.figure(figsize=(10, 4))
plt.plot(years, np.abs(ys_at_obs - P_obs), marker='o', color='purple')
plt.title("Error Absolut per Tahun")
plt.xlabel("Tahun")
plt.ylabel("Error")
plt.grid(True)
plt.show()