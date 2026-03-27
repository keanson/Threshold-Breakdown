import numpy as np
from ftbp.estimators import psi, psi_prime, estimate_theta, estimate_sigma
from scipy.stats import norm
from scipy.optimize import brentq
import itertools
from scipy.special import comb

LARGE_SENTINEL = 30

def two_sample_sigma_hat(x, y, theta_hat_x, theta_hat_y, delta=1.0, loss_type='huber'):
    sigma_hat_x = estimate_sigma(x, theta_hat_x, delta, loss_type=loss_type)
    sigma_hat_y = estimate_sigma(y, theta_hat_y, delta, loss_type=loss_type)
    return np.sqrt(sigma_hat_x ** 2 + sigma_hat_y ** 2)

def estimate_theta_bootstrap(x, w_b, delta=1.0, loss_type='huber'):
    """
    Estimate the bootstrap M-estimate of location.
    """
    f = lambda t: np.sum(w_b * psi(x - t, delta, loss_type=loss_type))
    a, b = np.min(x) - LARGE_SENTINEL, np.max(x) + LARGE_SENTINEL
    return brentq(f, a, b)

# ========= BOOTSTRAP WALD TEST =========

def wald_test_bootstrap(x, y, w_x, w_y, delta=1.0, alpha=0.05, loss_type='huber'):
    """
    Two-sample bootstrap Wald test.
    Reject iff 0 ∉ (θ̂1_b − θ̂2_b) ± z_{1−α/2} * sqrt( σ̂1^2 + σ̂2^2 ),
    where θ̂k_b are weighted bootstrap M-estimates and σ̂k are plug-in SEs
    computed from *unweighted* sums at those θ̂k_b (as in your formula).
    Returns int( reject ).
    """
    # bootstrap location estimates (weighted)
    theta1_b = estimate_theta_bootstrap(x, w_x, delta, loss_type=loss_type)
    theta2_b = estimate_theta_bootstrap(y, w_y, delta, loss_type=loss_type)

    # plug-in (unweighted) σ-hats evaluated at bootstrap thetas
    theta1 = estimate_theta(x, delta, loss_type=loss_type)
    theta2 = estimate_theta(y, delta, loss_type=loss_type)
    sigma1 = estimate_sigma(x, theta1, delta, null=False, loss_type=loss_type)
    sigma2 = estimate_sigma(y, theta2, delta, null=False, loss_type=loss_type)
    sigma = np.sqrt(sigma1**2 + sigma2**2)

    z = norm.ppf(1 - alpha/2)
    diff = theta1_b - theta2_b
    return int(np.abs(diff) > z * sigma)


def wald_test_bootstrap_details(x, y, w_x, w_y, delta=1.0, alpha=0.05, loss_type='huber'):
    """
    Same as wald_test_bootstrap but also returns diagnostic details.
    """
    theta1_b = estimate_theta_bootstrap(x, w_x, delta, loss_type=loss_type)
    theta2_b = estimate_theta_bootstrap(y, w_y, delta, loss_type=loss_type)

    theta1 = estimate_theta(x, delta, loss_type=loss_type)
    theta2 = estimate_theta(y, delta, loss_type=loss_type)
    sigma1 = estimate_sigma(x, theta1, delta, null=False, loss_type=loss_type)
    sigma2 = estimate_sigma(y, theta2, delta, null=False, loss_type=loss_type)
    sigma = np.sqrt(sigma1**2 + sigma2**2)

    z = norm.ppf(1 - alpha/2)
    diff = theta1_b - theta2_b
    reject = int(np.abs(diff) > z * sigma)
    ci = (diff - z * sigma, diff + z * sigma)
    return {
        "reject": reject,
        "theta1_b": theta1_b,
        "theta2_b": theta2_b,
        "diff": diff,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma": sigma,
        "z": z,
        "alpha": alpha,
        "ci_for_diff": ci,
    }

# == Vanilla Breakdown Point at eta ==
def _sort_pairs_by_x(x, w):
    order = np.argsort(x)
    return x[order], w[order], order

# ----------- One-sided optimal attacks -----------
def eta_theta_plus(x, wx, theta_hat, m, delta=1.0, loss_type='huber'):
    x_sorted, wx_sorted, _ = _sort_pairs_by_x(x, wx)
    # split the weight ratio such that it corresponds to m/n for \sum_{i \in I} w_i / \sum_{i=1}^n w_i
    n = len(x)
    total_wx = np.sum(wx_sorted)
    desired_cum = m / n * total_wx
    if desired_cum == 0:
        return 0.0
    # check indices
    cs = np.cumsum(wx_sorted)
    idx = np.searchsorted(cs, desired_cum, side='right')
    prev = cs[idx-1] if idx > 0 else 0.0
    take = desired_cum - prev
    remainder = wx_sorted[idx] - take
    # create wx_sorted modified
    wx_sorted[idx] = remainder
    wx_sorted = np.insert(wx_sorted, idx, take)
    x_sorted = np.insert(x_sorted, idx, x_sorted[idx])
    def f1(t):
        return np.sum(wx_sorted[idx+1:] * psi(x_sorted[idx+1:] - t, delta, loss_type=loss_type)) + np.sum(wx_sorted[:idx+1]) * delta
    try:
        theta1 = brentq(f1, theta_hat - LARGE_SENTINEL, theta_hat + LARGE_SENTINEL)
        return theta1 - theta_hat
    except ValueError:
        return np.inf

def eta_theta_minus(x, wx, theta_hat, m, delta=1.0, loss_type='huber'):
    x_sorted, wx_sorted, _ = _sort_pairs_by_x(x, wx)
    # split the weight ratio such that it corresponds to m/n for \sum_{i \in I} w_i / \sum_{i=1}^n w_i
    n = len(x)
    total_wx = np.sum(wx_sorted)
    desired_cum = m / n * total_wx
    if desired_cum == 0:
        return 0.0
    # check indices
    cs = np.cumsum(wx_sorted[::-1])
    idx = np.searchsorted(cs, desired_cum, side='right')
    prev = cs[idx-1] if idx > 0 else 0.0
    take = desired_cum - prev
    remainder = wx_sorted[n-idx-1] - take
    # create wx_sorted modified
    wx_sorted[n-idx-1] = take
    wx_sorted = np.insert(wx_sorted, n-idx-1, remainder)
    x_sorted = np.insert(x_sorted, n-idx-1, x_sorted[n-idx-1])
    def f2(t):
        return np.sum(wx_sorted[:n-idx] * psi(x_sorted[:n-idx] - t, delta, loss_type=loss_type)) - np.sum(wx_sorted[n-idx:]) * delta
    try:
        theta2 = brentq(f2, theta_hat - LARGE_SENTINEL, theta_hat + LARGE_SENTINEL)
        return theta_hat - theta2
    except ValueError:
        return np.inf

def eta_at_m_bootstrap(x, y, w_x, w_y, m, delta=1.0, alpha=0.05, loss_type='huber', direction='positive'):
    """
    Implements the vanilla definition of the breakdown point at m for the bootstrap. Direction can be 'positive', 'negative', or 'both', with positive meaning we want to stretch the difference bewteen theta_x - theta_y, negative meaning we want to shrink it, and both meaning we want to consider both directions.
    Returns a dict with 'eta' and 'm'.
    """
    n1, n2 = len(x), len(y)
    if m >= min(n1, n2)/2:
        return {'eta': np.inf, 'm': m}
    info = _compute_basics_bootstrap(x, y, w_x, w_y, delta, alpha, loss_type)
    theta_x = info["theta1_b"]
    theta_y = info["theta2_b"]
    diff = info["diff"]

    k1_min, k1_max = max(0, m - n2), min(m, n1)
    eta_max = -np.inf
    for k1 in range(k1_min, k1_max + 1):
        k2 = m - k1
        if direction in ['positive', 'both']:
            eta_x = eta_theta_plus(x, w_x, theta_x, k1, delta, loss_type=loss_type)
            eta_y = eta_theta_minus(y, w_y, theta_y, k2, delta, loss_type=loss_type)
            if theta_x + eta_x - (theta_y - eta_y) - diff > eta_max:
                eta_max = theta_x + eta_x - (theta_y - eta_y) - diff
        if direction in ['negative', 'both']:
            eta_x = eta_theta_minus(x, w_x, theta_x, k1, delta, loss_type=loss_type)
            eta_y = eta_theta_plus(y, w_y, theta_y, k2, delta, loss_type=loss_type)
            if diff - (theta_x - eta_x - (theta_y + eta_y)) > eta_max:
                eta_max = diff - (theta_x - eta_x - (theta_y + eta_y))

    return {'eta': eta_max, 'm': m}

def bp_at_eta_bootstrap(x, y, w_x, w_y, eta, delta=1.0, alpha=0.05, loss_type='huber', m_max=None):
    """
    Implements the vanilla definition of the breakdown point at η for the bootstrap. 
    Returns a dict with 'bp' and 'm'.
    """
    n1, n2 = len(x), len(y)
    if m_max is None:
        m_max = n1 + n2

    info = _compute_basics_bootstrap(x, y, w_x, w_y, delta, alpha, loss_type)
    diff = info["diff"]

    for m in range(1, m_max+1):
        if m >= min(n1, n2)/2:
            feasible = True
            return {'bp': m/(n1 + n2), 'm': m, 'wbp': m/(n1 + n2)}
        feasible = False
        k1_min, k1_max = max(0, m - n2), min(m, n1)
        for k1 in range(k1_min, k1_max + 1):
            k2 = m - k1
            eta_x = eta_theta_plus(x, w_x, info["theta1_b"], k1, delta, loss_type=loss_type)
            eta_y = eta_theta_minus(y, w_y, info["theta2_b"], k2, delta, loss_type=loss_type)
            if eta_x + eta_y >= eta:
                feasible = True
                break
            eta_x = eta_theta_minus(x, w_x, info["theta1_b"], k1, delta, loss_type=loss_type)
            eta_y = eta_theta_plus(y, w_y, info["theta2_b"], k2, delta, loss_type=loss_type)
            if eta_x + eta_y >= eta:
                feasible = True
                break
            
        if feasible:
            return {'bp': m/(n1 + n2), 'm': m}


def _compute_basics_bootstrap(x, y, w_x, w_y, delta=1.0, alpha=0.05, loss_type='huber'):
    """Compute θ_b, θ (unweighted), σ (unweighted), diff, z, φ_b."""
    theta1_b = estimate_theta_bootstrap(x, w_x, delta, loss_type=loss_type)
    theta2_b = estimate_theta_bootstrap(y, w_y, delta, loss_type=loss_type)
    theta1    = estimate_theta(x, delta, loss_type=loss_type)
    theta2    = estimate_theta(y, delta, loss_type=loss_type)
    sigma1    = estimate_sigma(x, theta1, delta, null=False, loss_type=loss_type)
    sigma2    = estimate_sigma(y, theta2, delta, null=False, loss_type=loss_type)
    sigma     = np.sqrt(sigma1**2 + sigma2**2)
    z         = norm.ppf(1 - alpha/2)
    diff      = theta1_b - theta2_b
    reject    = int(np.abs(diff) > z * sigma)
    return {
        "theta1_b": theta1_b, "theta2_b": theta2_b,
        "theta1": theta1, "theta2": theta2,
        "sigma1": sigma1, "sigma2": sigma2,
        "sigma": sigma, "z": z, "diff": diff, "reject": reject
    }
