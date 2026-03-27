import numpy as np
import pandas as pd
import itertools
from math import comb
from scipy.optimize import brentq
from scipy.stats import norm, cauchy, uniform
import scipy.stats
from ftbp.estimators import psi, estimate_theta, estimate_sigma

DELTA = 1.345  # Huber's proposal for 95% efficiency at the normal model
# In the experiments, we are using distributions located at 0 with std 1, so we set:
LARGE_SENTINEL = 30
LARGE_SENTINEL_2 = 100

def compute_scale(x, scale_type, dist):
    if scale_type == 'MAD':
        return scipy.stats.median_abs_deviation(x)
    elif scale_type == 'huber':
        n = len(x)
        def chi(u):
            u = np.asarray(u)
            return np.where(np.abs(u) <= DELTA, u**2, DELTA**2)

        def objective(sigma):
            if sigma <= 0:
                return np.inf  # avoid invalid scale
            u = x / sigma
            return np.sum(chi(u)) / n
        
        # Since chi >= 0, the sum is >= 0; we want to find the sigma where it's close to zero
        # Start with a large bracket
        lower = 0
        upper = max(100 * np.std(x), 10)

        if dist == 'normal':
            b = 0.71
        elif dist == 'Cauchy':
            b = 1
        elif dist == 'uniform':
            b = 1/3
        else:
            raise ValueError(f"Unknown distribution: {dist}")

        # Shift the function to zero crossing if you insist on root-finding
        def f(sigma):
            return objective(sigma) - b 

        try:
            solution = brentq(f, lower, upper, xtol=1e-6, maxiter=1000)
            flag = 0
        except:
            solution = 0 # sigma is 0 (exploded in this case)
            flag = 1
        if flag == 1:
            print(objective(lower) - b, objective(upper) - b)
        return solution
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")
    
def two_stage(x, delta=1.0, loss_type='huber', scale_type='MAD', dist='normal'):
    sigma_hat = compute_scale(x, scale_type, dist)
    new_x = x / sigma_hat
    theta_hat = estimate_theta(new_x, delta, loss_type=loss_type)
    return theta_hat * sigma_hat, sigma_hat

def two_stage_eta_lower_plus(x, m, delta=1.0, loss_type='huber', scale_type='MAD', dist='normal'):
    x_sorted = np.sort(x)
    theta = two_stage(x_sorted, delta=delta, loss_type=loss_type, scale_type=scale_type, dist=dist)[0]
    n = len(x_sorted)
    x_infty = np.concatenate([x_sorted[m:], np.ones(m)*LARGE_SENTINEL])
    x_n = np.concatenate([x_sorted[m:], np.ones(m)*x_sorted[-1]])
    theta_infty = two_stage(x_infty, delta=delta, loss_type=loss_type, scale_type=scale_type, dist=dist)[0]
    theta_n = two_stage(x_n, delta=delta, loss_type=loss_type, scale_type=scale_type, dist=dist)[0]
    eta_infty = theta_infty - theta
    eta_n = theta_n - theta
    return max(eta_infty, eta_n)

def two_stage_eta_lower_minus(x, m, delta=1.0, loss_type='huber', scale_type='MAD', dist='normal'):
    x_sorted = np.sort(x)
    theta = two_stage(x_sorted, delta=delta, loss_type=loss_type, scale_type=scale_type, dist=dist)[0]
    n = len(x_sorted)
    x_infty = np.concatenate([x_sorted[:n-m], np.ones(m)*-LARGE_SENTINEL])
    x_n = np.concatenate([x_sorted[:n-m], np.ones(m)*x_sorted[0]])
    theta_infty = two_stage(x_infty, delta=delta, loss_type=loss_type, scale_type=scale_type, dist=dist)[0]
    theta_n = two_stage(x_n, delta=delta, loss_type=loss_type, scale_type=scale_type, dist=dist)[0]
    eta_infty = theta - theta_infty
    eta_n = theta - theta_n
    return max(eta_infty, eta_n)

def two_stage_BP_upper_plus(x, eta, delta=1.0, loss_type='huber', scale_type='MAD', dist='normal'):
    n = len(x)
    theta_hat, sigma_hat = two_stage(x, delta=delta, loss_type=loss_type, scale_type=scale_type, dist=dist)
    psi_inf = delta
    x_sorted = np.sort(x)

    Delta_inf = n/2
    for m in range(1, n+1):
        data_inf = np.concatenate([x_sorted[m:], np.ones(m)*LARGE_SENTINEL])
        sigma_m_inf = compute_scale(data_inf, scale_type, dist)
        if sigma_m_inf == 0:
            continue

        r = (x_sorted[m:] - (theta_hat + eta)) / sigma_m_inf
        sum_psi = np.sum(psi(r, delta=delta, loss_type=loss_type))
        rhs = - sum_psi / psi_inf

        if m >= rhs:
            Delta_inf = m
            break

    Delta_n = n/2
    x_n = x_sorted[-1]
    for m in range(1, n+1):
        data_n = np.concatenate([x_sorted[m:], np.ones(m)*x_n])
        sigma_m_n = compute_scale(data_n, scale_type, dist)
        if sigma_m_n == 0:
            continue

        r_vec = (x_sorted[m:] - (theta_hat + eta)) / sigma_m_n
        num = np.sum(psi(r_vec, delta=delta, loss_type=loss_type))
        r_den = (x_n - (theta_hat + eta)) / sigma_m_n
        den = psi(r_den, delta=delta, loss_type=loss_type)
        if den <= 0:
            continue

        rhs = - num / den
        if m >= rhs:
            Delta_n = m
            break

    return min(Delta_inf, Delta_n)

def two_stage_BP_upper_minus(x, eta, delta=1.0, loss_type='huber', scale_type='MAD', dist='normal'):
    n = len(x)
    theta_hat, sigma_hat = two_stage(x, delta=delta, loss_type=loss_type, scale_type=scale_type, dist=dist)
    psi_minus_inf = -delta
    x_sorted = np.sort(x)

    Delta_minus_inf = n/2
    for m in range(1, n+1):
        data_inf = np.concatenate([x_sorted[:n-m], np.ones(m)*-LARGE_SENTINEL])
        sigma_m = compute_scale(data_inf, scale_type, dist)
        if sigma_m == 0:
            continue
        r = (x_sorted[:n-m] - (theta_hat - eta)) / sigma_m
        sum_psi = np.sum(psi(r, delta=delta, loss_type=loss_type))
        rhs = -sum_psi / psi_minus_inf
        if m >= rhs:
            Delta_minus_inf = m
            break

    x1 = x_sorted[0]
    Delta_minus_1 = n/2
    for m in range(1, n+1):
        data1 = np.concatenate([x_sorted[:n-m], np.ones(m)*x1])
        sigma_m = compute_scale(data1, scale_type, dist)
        if sigma_m == 0:
            continue
        r = (x_sorted[:n-m] - (theta_hat - eta)) / sigma_m
        sum_psi = np.sum(psi(r, delta=delta, loss_type=loss_type))
        r_den = (x1 - (theta_hat - eta)) / sigma_m
        den = psi(r_den, delta=delta, loss_type=loss_type)
        if den >= 0:
            continue
        rhs = -sum_psi / den
        if m >= rhs:
            Delta_minus_1 = m
            break

    return min(Delta_minus_inf, Delta_minus_1)

def huber_scale_range(x, m, delta=1.0, dist='normal'):
    new_x = np.abs(x)
    new_x = np.sort(new_x)
    n = len(new_x)
    x_plus = np.concatenate([new_x[m:], np.ones(m) * LARGE_SENTINEL_2])
    x_minus = np.concatenate([new_x[:n-m], np.ones(m) * 0])
    sigma_plus = compute_scale(x_plus, 'huber', dist)
    sigma_minus = compute_scale(x_minus, 'huber', dist)
    return sigma_plus, sigma_minus

def sign_split_rescalings(x, m, delta=1.0, scale_type='huber', dist='normal'):
    """
    Given data x and contamination size m, compute
      a = S_min(m; x), b = S_max(m; x),
      r_ab    = x_i/b if x_i>=0 else x_i/a,
      rprime  = x_i/a if x_i>=0 else x_i/b.
    """
    x_sorted = np.sort(x)
    if scale_type == 'huber':
        S_max, S_min = huber_scale_range(x_sorted, m, delta=delta, dist=dist)
    else:
        raise ValueError(f"Scale_type: {scale_type} not implemented")
    
    if S_min == 0:
        S_min = 1e-8  # Avoid division by zero
    a = S_min
    b = S_max
    r_ab      = np.where(x >= 0, x/b,      x/a)
    rprime_ab = np.where(x >= 0, x/a,      x/b)
    return a, b, r_ab, rprime_ab

def eta_theta_plus(x, theta_hat, m, delta=1.0, loss_type='huber'):
    x_sorted = np.sort(x)
    n = len(x)
    def f1(t):
        return np.sum(psi(x_sorted[m:] - t, delta, loss_type=loss_type)) + m * delta
    try:
        theta1 = brentq(f1, theta_hat - LARGE_SENTINEL, theta_hat + LARGE_SENTINEL)
        return theta1 - theta_hat
    except ValueError:
        return np.nan

def eta_theta_minus(x, theta_hat, m, delta=1.0, loss_type='huber'):
    x_sorted = np.sort(x)
    def f2(t):
        return np.sum(psi(x_sorted[:len(x)-m] - t, delta, loss_type=loss_type)) - m * delta
    try:
        theta2 = brentq(f2, theta_hat - LARGE_SENTINEL, theta_hat + LARGE_SENTINEL)
        return theta_hat - theta2
    except ValueError:
        return np.nan

# --- 3. two-stage estimator upper‐bounds from the Lemma ---
def two_stage_eta_upper_plus(x, m, delta=1.0, loss_type='huber', scale_type='MAD', dist='normal'):
    theta_hat, sigma_hat = two_stage(x, delta, loss_type, scale_type, dist)
    a, b, r_ab, rprime = sign_split_rescalings(x, m, delta, scale_type, dist)
    # theta_loc_hat = estimate_theta(rprime, delta, loss_type=loss_type)
    eta_loc = eta_theta_plus(rprime, theta_hat * sigma_hat, m, delta, loss_type)
    if theta_hat < 0:
        c = a
    else:
        c = b
    return ((c - sigma_hat) / sigma_hat) * theta_hat + b * eta_loc

def two_stage_eta_upper_minus(x, m, delta=1.0,
                              loss_type='huber', scale_type='MAD', dist='normal'):
    theta_hat, sigma_hat = two_stage(x, delta, loss_type, scale_type, dist)
    a, b, r, rprime = sign_split_rescalings(x, m, delta, scale_type, dist)
    # theta_loc_hat = estimate_theta(r, delta, loss_type=loss_type)
    eta_loc = eta_theta_minus(r, theta_hat * sigma_hat, m, delta, loss_type)
    if theta_hat < 0:
        c = b
    else:
        c = a
    return ((sigma_hat - c) / sigma_hat) * theta_hat + b * eta_loc

# --- 4. breakdown‐point lower bounds via those upper bounds ---
def two_stage_BP_lower_plus(x, eta, delta=1.0,
                  loss_type='huber', scale_type='MAD', dist='normal'):
    """
    Compute BP_{η+} ≥ (1/n) * min{ m : two_stage_eta_upper_plus ≥ η }.
    """
    n = len(x)
    for m in range(n+1):
        if two_stage_eta_upper_plus(x, m, delta, loss_type, scale_type, dist) >= eta:
            return m
    return 1.0

def two_stage_BP_lower_minus(x, eta, delta=1.0,
                   loss_type='huber', scale_type='MAD', dist='normal'):
    """
    Compute BP_{η-} ≥ (1/n) * min{ m : two_stage_bound_eta_minus ≥ η }.
    """
    n = len(x)
    for m in range(n+1):
        if two_stage_eta_upper_minus(x, m, delta, loss_type, scale_type, dist) >= eta:
            return m
    return 1.0
