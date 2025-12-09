from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import math

app = Flask(__name__, static_folder="static", template_folder="templates")

# -------------------------
# Load dataset & detect columns
# -------------------------
DF_PATH = "population_by_year_2014_2024.csv"
df = pd.read_csv(DF_PATH)

# detect country column (case-insensitive) or fallback to first column
country_col = None
for c in df.columns:
    if "country" in c.lower() or "name" in c.lower():
        country_col = c
        break
if country_col is None:
    country_col = df.columns[0]

# detect year columns that are purely digits (e.g. '2014','2015',...)
year_cols = [c for c in df.columns if c.isdigit()]
year_cols_sorted = sorted([int(y) for y in year_cols])

# -------------------------
# Helpers
# -------------------------
def clean_number(x):
    """Clean string/number input and return float."""
    if pd.isna(x):
        raise ValueError("Missing numeric value")
    s = str(x)
    for ch in ["$", ",", " ", "—", "–"]:
        s = s.replace(ch, "")
    # parentheses negative
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    if s.strip() == "":
        raise ValueError(f"Empty after cleaning original={x}")
    return float(s)

# logistic ODE derivative
def f_logistic(t, P, r, K):
    return r * P * (1 - P / K)

# RK4 step and solver
def rk4_step(func, t, y, h, r, K):
    k1 = func(t, y, r, K)
    k2 = func(t + 0.5*h, y + 0.5*h*k1, r, K)
    k3 = func(t + 0.5*h, y + 0.5*h*k2, r, K)
    k4 = func(t + h, y + h*k3, r, K)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def rk4_solver(func, t_span, y0, h, r, K):
    t0, t_end = t_span
    ts = np.arange(t0, t_end + h, h)
    ys = np.zeros(len(ts))
    ys[0] = y0
    for i in range(len(ts)-1):
        ys[i+1] = rk4_step(func, ts[i], ys[i], h, r, K)
    return ts, ys

# analytic logistic (for fitting)
def logistic_analytic(t, r, K, P0):
    A = (K - P0) / P0
    return K / (1 + A * np.exp(-r * t))

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    # send country list and years to template
    countries = df[country_col].astype(str).tolist()
    years = year_cols_sorted
    return render_template("index.html", countries=countries, years=years)

@app.route("/simulate", methods=["POST"])
def simulate():
    """
    Expects JSON:
    {
      "country": "Australia",
      "start_year": 2018,
      "end_year": 2020
    }
    """
    payload = request.get_json(force=True)
    country = payload.get("country")
    start = int(payload.get("start_year"))
    end = int(payload.get("end_year"))

    # validate years
    if end <= start:
        return jsonify({"error":"end_year must be greater than start_year"}), 400
    years = list(range(start, end+1))
    #if len(years) < 2 or len(years) > 3:
    #    return jsonify({"error":"Choose range of 2 to 3 years only."}), 400

    # find row
    matched = df[df[country_col].astype(str).str.lower() == str(country).lower()]
    if matched.empty:
        return jsonify({"error": f"Country {country} not found."}), 404
    row = matched.iloc[0]

    # extract population values for selected years
    try:
        P = [clean_number(row[str(y)]) for y in years]
    except Exception as e:
        return jsonify({"error": f"Failed to parse population values: {e}"}), 400

    # prepare arrays
    t_obs = np.array([y - years[0] for y in years])  # relative time starting at 0
    P_obs = np.array(P)
    P0 = P_obs[0]

    # initial guesses
    r_guess = np.mean(np.log(P_obs[1:] / P_obs[:-1]))
    K_guess = max(P_obs) * 1.2

    # fit analytic logistic using curve_fit
    try:
        popt, pcov = curve_fit(lambda t, r, K: logistic_analytic(t, r, K, P0),
                               t_obs, P_obs, p0=[r_guess, K_guess], maxfev=10000)
        r_fit, K_fit = float(popt[0]), float(popt[1])
    except Exception as e:
        # fallback to simple estimates
        r_fit, K_fit = float(r_guess), float(K_guess)

    # run RK4 with fine h for smooth curve
    h = 0.1
    ts, ys = rk4_solver(f_logistic, (0, t_obs[-1]), P0, h, r_fit, K_fit)

    # compute analytic solution (for plot)
    G_exact = logistic_analytic(ts, r_fit, K_fit, P0)

    # sample at observation times (for comparison)
    ys_at_obs = np.interp(t_obs, ts, ys)

    # compute RMSE
    rmse = float(np.sqrt(np.mean((ys_at_obs - P_obs)**2)))

    # prepare JSON serializable output
    response = {
        "country": country,
        "years": years,
        "r_fit": r_fit,
        "K_fit": K_fit,
        "rmse": rmse,
        "t_smooth": (years[0] + ts).tolist(),   # absolute years for plotting
        "ys_smooth": ys.tolist(),
        "ys_exact": G_exact.tolist(),
        "obs_years": years,
        "obs_values": P_obs.tolist(),
        "ys_at_obs": ys_at_obs.tolist(),
        "errors_at_obs": (np.abs(ys_at_obs - P_obs)).tolist()
    }
    return jsonify(response)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
