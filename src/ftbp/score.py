import numpy as np
from ftbp.estimators import psi, psi_prime, estimate_theta, estimate_sigma
from scipy.stats import norm
from scipy.optimize import brentq
import itertools
from scipy.special import comb


# Standard normal PDF
def phi(a):
    return np.exp(-a**2 / 2) / np.sqrt(2 * np.pi)

# ----------- M-estimate of location -----------
def estimate_v(x, delta=1.0, loss_type='huber'):
    return np.sum(psi(x, delta, loss_type=loss_type)) / np.sqrt(len(x))

# ----------- Plug-in standard error -----------
def estimate_s(x, delta=1.0, null=False, loss_type='huber'):
    # Let's just use normal data for null (integration)
    if null:
        # integrand = lambda r: psi(r, delta, loss_type=loss_type) ** 2 * phi(r)
        # return np.sqrt(quad(integrand, -np.inf, np.inf)[0])
        return 0.8427126131022351  # This is the value for the Huber loss function with delta = 1.345
    else:
        return np.sqrt(np.sum(psi(x, delta, loss_type=loss_type) ** 2) / len(x))

# ----------- Wald test -----------
def wald_test(x, delta=1.0, alpha=0.05, null=False, loss_type='huber'):
    v_hat = estimate_v(x, delta, loss_type=loss_type)
    if null:
        sigma_hat = estimate_s(x, delta, null=null, loss_type=loss_type)
    else:
        sigma_hat = estimate_s(x, delta, loss_type=loss_type)
    T = v_hat / sigma_hat
    return int(np.abs(T) > norm.ppf(1 - alpha/2))

# ----------- One-sided optimal attacks (Corollary 5) -----------
def eta_v_plus(x, v_hat, m, delta=1.0, loss_type='huber'):
    x_sorted = np.sort(x)
    n = len(x)
    new_v = (np.sum(psi(x_sorted[m:], delta, loss_type=loss_type)) + m * psi(30, delta, loss_type=loss_type)) / np.sqrt(n)
    return new_v - v_hat

def eta_v_minus(x, v_hat, m, delta=1.0, loss_type='huber'):
    x_sorted = np.sort(x)
    n = len(x)
    new_v = (np.sum(psi(x_sorted[:n-m], delta, loss_type=loss_type)) + m * psi(-30, delta, loss_type=loss_type)) / np.sqrt(n)
    return v_hat - new_v

# --- Update eta_s_plus ---
def eta_s_plus(x, s_hat, m, delta=1.0, null=False, loss_type='huber'):
    if null:
        return 0
    else:
        y = np.abs(x)
        order = np.argsort(y)
        attacked = order[:m]
        remaining = order[m:]
        new_s = np.sum(psi(x[remaining], delta, loss_type=loss_type) ** 2) + m * (delta ** 2)
        new_s /= len(x)
        return np.sqrt(new_s) - s_hat

# --- Update eta_sigma_minus ---
def eta_s_minus(x, s_hat, m, delta=1.0, null=False, loss_type='huber'):
    if null:
        return 0
    else:
        y = np.abs(x)
        order = np.argsort(y)
        remaining = order[:len(x)-m]
        new_s = np.sum(psi(x[remaining], delta, loss_type=loss_type) ** 2)
        new_s /= len(x)
        return s_hat - np.sqrt(new_s)

# ----------- Real breakdown point (brute force) -----------
def real_bp(x, grid, delta=1.0, mode='power', max_ops=1e10, loss_type='huber'):
    n = len(x)
    phi0 = wald_test(x, delta, loss_type=loss_type)
    target = 0 if mode == 'power' else 1
    if phi0 != (1 - target):
        return np.nan
    for m in range(1, n+1):
        ops = comb(n, m) * len(grid)
        if ops > max_ops:
            continue
        for I in itertools.combinations(range(n), m):
            for t in grid:
                x_prime = x.copy()
                x_prime[list(I)] = t
                if wald_test(x_prime, delta, loss_type=loss_type) == target:
                    return m
    return np.nan

# ----------- Upper bound for power BP (Corollary 6) -----------
def bound_power_upper(x, delta=1.0, alpha=0.05, null=False, loss_type='huber'):
    n = len(x)
    v_hat = estimate_v(x, delta, loss_type=loss_type)
    s_hat = estimate_s(x, delta, null=null, loss_type=loss_type)
    z = norm.ppf(1 - alpha/2)
    x_sorted = np.sort(x)
    if v_hat - z * s_hat > 0:
        for m in range(1, n+1):
            x_new = x_sorted.copy()
            x_new[n-m:] = -30
            if wald_test(x_new, delta, null=null, loss_type=loss_type) == 0:
                return m
    else:
        for m in range(1, n+1):
            x_new = x_sorted.copy()
            x_new[:m] = 30
            if wald_test(x_new, delta, null=null, loss_type=loss_type) == 0:
                return m
    return np.nan

# ----------- Lower bound for power BP (Corollary 6) -----------
def bound_power_lower(x, delta=1.0, alpha=0.05, null=False, loss_type='huber'):
    n = len(x)
    v_hat = estimate_v(x, delta, loss_type=loss_type)
    s_hat = estimate_s(x, delta, null=null, loss_type=loss_type)
    z = norm.ppf(1 - alpha/2)
    if v_hat - z * s_hat > 0:
        for m in range(1, n+1):
            eta_vv_minus = eta_v_minus(x, v_hat, m, delta, loss_type=loss_type)
            eta_ss_plus = eta_s_plus(x, s_hat, m, delta, null=null, loss_type=loss_type)
            if eta_vv_minus + z * eta_ss_plus >= v_hat - z * s_hat:
                return m
    else:
        for m in range(1, n+1):
            eta_vv_plus = eta_v_plus(x, v_hat, m, delta, loss_type=loss_type)
            eta_ss_plus = eta_s_plus(x, s_hat, m, delta, null=null, loss_type=loss_type)
            if eta_vv_plus + z * eta_ss_plus >= -v_hat - z * s_hat:
                return m
    return np.nan

# ----------- Upper bound for level BP (Corollary 6) -----------
def bound_level_upper(x, delta=1.0, alpha=0.05, null=False, loss_type='huber'):
    n = len(x)
    x_sorted = np.sort(x)
    m1 = m2 = n
    for m in range(1, n+1):
        x_new = x_sorted.copy()
        x_new[n-m:] = x_new[0]
        if wald_test(x_new, delta, null=null, loss_type=loss_type) == 1:
            m1 = m
            break
    for m in range(1, n+1):
        x_new = x_sorted.copy()
        x_new[:m] = x_new[-1]
        if wald_test(x_new, delta, null=null, loss_type=loss_type) == 1:
            m2 = m
            break
    return min(m1, m2)

# ----------- Lower bound for level BP (Corollary 6) -----------
def bound_level_lower(x, delta=1.0, alpha=0.05, null=False, loss_type='huber'):
    n = len(x)
    v_hat = estimate_v(x, delta, loss_type=loss_type)
    s_hat = estimate_s(x, delta, null=null, loss_type=loss_type)
    z = norm.ppf(1 - alpha/2)
    m1 = m2 = n
    for m in range(1, n+1):
        eta_vv_minus = eta_v_minus(x, v_hat, m, delta, loss_type=loss_type)
        eta_ss_minus = eta_s_minus(x, s_hat, m, delta, null=null, loss_type=loss_type)
        if eta_vv_minus + z * eta_ss_minus >= v_hat + z * s_hat:
            m1 = m
            break
    for m in range(1, n+1):
        eta_vv_plus = eta_v_plus(x, v_hat, m, delta, loss_type=loss_type)
        eta_ss_minus = eta_s_minus(x, s_hat, m, delta, null=null, loss_type=loss_type)
        if eta_vv_plus + z * eta_ss_minus >= -v_hat + z * s_hat:
            m2 = m
            break
    return min(m1, m2)