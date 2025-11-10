import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import BSpline, interp1d
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import calmap


def entrenar_y_ajustar_modelo(df_train):
    df_train = df_train.sort_values('Fecha').reset_index(drop=True)
    t0_train = df_train['Fecha'].min()
    times_train = (df_train['Fecha'] - t0_train).dt.total_seconds() / (3600 * 24)
    T_train = times_train.max()
    times_train_np = times_train.values

    K = 8
    degree = 3
    knots = np.linspace(0, T_train, K)
    knots_extended = np.concatenate((np.repeat(knots[0], degree), knots, np.repeat(knots[-1], degree)))
    n_coef = len(knots) + degree - 1

    def spline_basis(t):
        t = np.atleast_1d(t)
        basis_matrix = np.zeros((len(t), n_coef))
        for i in range(n_coef):
            c = np.zeros(n_coef)
            c[i] = 1
            spline = BSpline(knots_extended, c, degree)
            basis_matrix[:, i] = spline(t)
        return basis_matrix

    spline_mat = spline_basis(times_train_np)

    def mu_vals_from_coef(c_mu):
        return spline_mat @ c_mu

    def alpha_vals_from_coef(c_alpha):
        return spline_mat @ c_alpha

    time_grid = np.linspace(0, T_train, 1000)
    spline_grid = spline_basis(time_grid)
    mu_integral_coeffs = np.trapz(spline_grid, time_grid, axis=0)

    lambda_reg = 0.01

    @njit
    def compute_log_likelihood_numba(times, mu_vals, alpha_vals, decay):
        n = len(times)
        ll = 0.0
        for i in range(n):
            excitation = 0.0
            for j in range(i):
                dt = times[i] - times[j]
                excitation += alpha_vals[j] * decay * np.exp(-decay * dt)
            intensity = mu_vals[i] + excitation
            if intensity <= 0:
                return -np.inf
            ll += np.log(intensity)
        return ll

    def log_likelihood(params):
        c_mu = params[:n_coef]
        c_alpha = params[n_coef:2*n_coef]
        decay = params[-1]

        if np.any(c_mu < 0) or np.any(c_alpha < 0) or decay <= 0:
            return np.inf

        mu_vals = mu_vals_from_coef(c_mu)
        alpha_vals = alpha_vals_from_coef(c_alpha)

        ll = compute_log_likelihood_numba(times_train_np, mu_vals, alpha_vals, decay)

        mu_integral = mu_integral_coeffs @ c_mu
        hawkes_integral = (np.mean(alpha_vals) / decay) * len(times_train_np)
        penalty = lambda_reg * np.sum(c_alpha**2)

        return -(ll - (mu_integral + hawkes_integral)) + penalty

    initial_params = np.concatenate([
        np.full(n_coef, len(times_train_np) / T_train / n_coef),
        np.full(n_coef, 0.1),
        [1.0]
    ])
    bounds = [(0, None)] * (2 * n_coef) + [(1e-3, 100)]

    res = minimize(log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B', options={'maxiter': 2000})

    c_mu_fit, c_alpha_fit, decay_fit = res.x[:n_coef], res.x[n_coef:2*n_coef], res.x[-1]

    mu_interp_base = interp1d(time_grid, spline_grid @ c_mu_fit, kind='cubic', fill_value="extrapolate")
    alpha_interp = interp1d(time_grid, spline_grid @ c_alpha_fit, kind='cubic', fill_value="extrapolate")

    return t0_train, mu_interp_base, alpha_interp, decay_fit, T_train


def forecast_period_single_sim(
    t0_train, mu_interp_fn, alpha_interp, decay_fit,
    period_start, period_end, mu_boost=1.0, T_train=None
):
    T_ini = (pd.to_datetime(period_start) - t0_train).total_seconds() / (3600 * 24)
    T_fin = (pd.to_datetime(period_end) - t0_train).total_seconds() / (3600 * 24)

    def make_seasonal_fns(mu_fn, alpha_fn, t_train_end, t_start_sim, t_end_sim):
        days_in_year = 365
        n_years = 3
        t_sim_grid = np.linspace(t_start_sim, t_end_sim, 1000)
        seasonal_mu = np.zeros_like(t_sim_grid)
        seasonal_alpha = np.zeros_like(t_sim_grid)

        for y in range(1, n_years + 1):
            offset = y * days_in_year
            t_shifted = t_sim_grid - offset
            valid = (t_shifted >= 0) & (t_shifted <= t_train_end)
            seasonal_mu[valid] += mu_fn(t_shifted[valid])
            seasonal_alpha[valid] += alpha_fn(t_shifted[valid])

        seasonal_mu /= n_years
        seasonal_alpha /= n_years

        mu_fn = interp1d(t_sim_grid, mu_boost * seasonal_mu, fill_value="extrapolate")
        alpha_fn = interp1d(t_sim_grid, seasonal_alpha, fill_value="extrapolate")
        return mu_fn, alpha_fn

    mu_fn, alpha_fn = make_seasonal_fns(mu_interp_fn, alpha_interp, T_train, T_ini, T_fin)
    events_sim = simulate_hawkes_ogata(mu_fn, alpha_fn, decay_fit, T_ini, T_fin)
    return events_sim  # DEVOLVEMOS LOS EVENTOS SIMULADOS


def simulate_hawkes_ogata(mu_fn, alpha_fn, decay, t_start, t_end, max_jumps=10000):
    np.random.seed(42)
    t = t_start
    events = []
    while t < t_end and len(events) < max_jumps:
        excitation = np.sum([alpha_fn(s) * decay * np.exp(-decay * (t - s)) for s in events if s < t])
        lambda_t = mu_fn(t) + excitation
        if lambda_t <= 0:
            break
        u = np.random.uniform()
        w = -np.log(u) / lambda_t
        t_candidate = t + w
        if t_candidate > t_end:
            break
        excitation_candidate = np.sum([alpha_fn(s) * decay * np.exp(-decay * (t_candidate - s)) for s in events if s < t_candidate])
        lambda_candidate = mu_fn(t_candidate) + excitation_candidate
        if np.random.uniform() <= lambda_candidate / lambda_t:
            events.append(t_candidate)
        t = t_candidate
    return np.array(events)


def generar_calendario_eventos(t0_train, events_real, events_sim, start_date, end_date):
    dates_real = (t0_train + pd.to_timedelta(events_real, unit='D')).floor('D')
    dates_sim = (t0_train + pd.to_timedelta(events_sim, unit='D')).floor('D')

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    all_days = pd.date_range(start=start_date, end=end_date, freq='D')
    day_labels = pd.Series(0, index=all_days)

    for d in dates_real:
        if d in day_labels.index:
            day_labels[d] = 1

    for d in dates_sim:
        if d in day_labels.index:
            if day_labels[d] == 1:
                day_labels[d] = 3  # coincidencia
            elif day_labels[d] == 0:
                day_labels[d] = 2  # solo simulado

    colors = ['#ffffff', '#377eb8', '#e41a1c', '#984ea3']  # blanco, azul, rojo, morado
    cmap = ListedColormap(colors)

    plt.figure(figsize=(19, 9))
    calmap.calendarplot(
        day_labels,
        cmap=cmap,
        fillcolor='lightgrey',
        linewidth=0.5,
        linecolor='white',
        yearlabels=True,
        daylabels=['L', 'M', 'X', 'J', 'V', 'S', 'D'],
        yearascending=True
    )

    legend_elements = [
        Patch(facecolor='#377eb8', edgecolor='k', label='Real'),
        Patch(facecolor='#e41a1c', edgecolor='k', label='Simulado'),
        Patch(facecolor='#984ea3', edgecolor='k', label='Coinciden'),
        Patch(facecolor='#ffffff', edgecolor='k', label='Sin evento')
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=4)
    plt.title(f"Mapa de calor de feminicidios (reales, simulados y coincidencias)\n{start_date.date()} - {end_date.date()}")
    plt.tight_layout()
    plt.gcf().savefig("calmap_feminicidios.png", dpi=300)
    plt.show()

def obtener_fechas_simuladas(t0_train, events_sim):
    """
    Convierte los tiempos simulados (en días) a fechas reales del calendario.
    Devuelve un DataFrame ordenado con las fechas previstas.
    """
    fechas_sim = (t0_train + pd.to_timedelta(events_sim, unit='D')).floor('D')
    df_fechas = pd.DataFrame({"Fecha prevista": sorted(fechas_sim.unique())})
    return df_fechas

