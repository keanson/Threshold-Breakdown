# src/ftbp/estimators.py
import numpy as np
from scipy.optimize import brentq

# ----------- Robust psi and derivatives -----------
def psi(r, delta=1.0, loss_type='huber'):
    if loss_type == 'huber':
        return np.where(np.abs(r) <= delta, r, delta * np.sign(r))
    elif loss_type == 'logcosh':
        return np.tanh(r / delta) * delta
    elif loss_type == 'concordant':
        # The following form is correct
        return (np.sqrt(4 * (r / delta) ** 2 + 1) - 1) / (2 * (r / delta)) * delta
        # but originally we use this equivalent form + some numerical stability tricks
        # if np.min(np.abs(r)) < 1e-8:
        #     return np.zeros_like(r / delta)
        # else:
        #     return - (np.sqrt(4 * (r / delta) ** 2 + 1) - 2 * (r / delta) ** 2 - 1)/((r / delta) * (np.sqrt(4 * (r / delta) ** 2 + 1) - 1)) * delta
    else:
        raise ValueError('Unknown loss_type')

def psi_prime(r, delta=1.0, loss_type='huber'):
    if loss_type == 'huber':
        return np.where(np.abs(r) <= delta, 1.0, 0.0)
    elif loss_type == 'logcosh':
        return 1 - np.tanh(r / delta) ** 2
    elif loss_type == 'concordant':
        # The following form is correct
        val = (1 - 1 / np.sqrt(4 * (r / delta) ** 2 + 1)) / (2 * (r / delta) ** 2) 
        # if nan, and a single element
        if val.size == 1:
            if np.isnan(val):
                val = 1.0
        else:
            val[np.isnan(val)] = 1.0
        return val
        # but originally we use this equivalent form + some numerical stability tricks
        # return np.where(np.abs(r / delta) < 3e-2, 1.0, ((4 * (r / delta) ** 2 + 1) ** (3/2) + (1 - 2 * (r / delta) ** 2) * np.sqrt(4 * (r / delta) ** 2 + 1) - 6 * (r / delta) ** 2 - 2)/((r / delta) ** 2 * np.sqrt(4 * (r / delta) ** 2 + 1) * (np.sqrt(4 * (r / delta) ** 2 + 1) - 1) ** 2))
    else:
        raise ValueError('Unknown loss_type')

# ----------- M-estimate of location -----------
def estimate_theta(x, delta=1.0, loss_type='huber'):
    f = lambda t: np.sum(psi(x - t, delta, loss_type=loss_type))
    a, b = np.min(x) - 20, np.max(x) + 20
    return brentq(f, a, b, maxiter=1000)

# ----------- Plug-in standard error -----------
def estimate_sigma(x, theta_hat, delta=1.0, null=False, loss_type='huber'):
    if null:
        r = x
    else:
        r = x - theta_hat
    num = np.sum(psi(r, delta, loss_type=loss_type) ** 2)
    den = np.sum(psi_prime(r, delta, loss_type=loss_type)) ** 2
    if den <= 1e-8:
        return np.inf
    return np.sqrt(num / den)