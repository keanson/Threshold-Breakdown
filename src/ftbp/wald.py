import numpy as np
from ftbp.estimators import psi, psi_prime, estimate_theta, estimate_sigma
from scipy.stats import norm
from scipy.optimize import brentq
import itertools
from scipy.special import comb

# In the experiments, we are using distributions located at 0 with std 1, so we set:
LARGE_SENTINEL = 20
# This is for the Delta function in the non-Huber cases, we use 3 as the bracketing range for quick computation
DELTA_BRACKETING_RANGE = 3

# ----------- Wald test -----------
def wald_test(x, delta=1.0, alpha=0.05, null=False, loss_type='huber'):
    theta_hat = estimate_theta(x, delta, loss_type=loss_type)
    if null:
        sigma_hat = estimate_sigma(x, 0, delta, null=null, loss_type=loss_type)
    else:
        sigma_hat = estimate_sigma(x, theta_hat, delta, loss_type=loss_type)
    T = theta_hat / sigma_hat
    return int(np.abs(T) > norm.ppf(1 - alpha/2))

# ----------- One-sided optimal attacks (Corollary 5) -----------
def eta_theta_plus(x, theta_hat, m, delta=1.0, loss_type='huber'):
    if m == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = len(x)
    def f1(t):
        return np.sum(psi(x_sorted[m:] - t, delta, loss_type=loss_type)) + m * delta
    try:
        theta1 = brentq(f1, theta_hat - LARGE_SENTINEL, theta_hat + LARGE_SENTINEL)
        return theta1 - theta_hat
    except ValueError:
        return np.inf

def eta_theta_minus(x, theta_hat, m, delta=1.0, loss_type='huber'):
    if m == 0:
        return 0.0
    x_sorted = np.sort(x)
    def f2(t):
        return np.sum(psi(x_sorted[:len(x)-m] - t, delta, loss_type=loss_type)) - m * delta
    try:
        theta2 = brentq(f2, theta_hat - LARGE_SENTINEL, theta_hat + LARGE_SENTINEL)
        return theta_hat - theta2
    except ValueError:
        return np.inf

def Delta(t, delta=1.0, loss_type='huber'):
    from scipy.optimize import minimize_scalar
    if t == 0:
        return 0.0
    t = float(t)
    def obj(x):
        return -abs(psi_prime(x + t, delta, loss_type) - psi_prime(x, delta, loss_type))
    br = (-DELTA_BRACKETING_RANGE, DELTA_BRACKETING_RANGE)
    res = minimize_scalar(obj, bracket=br, method='Brent')
    return -res.fun

def find_diff_psi_prime(x, theta_hat, m, delta=1.0):
    x_sorted = np.sort(x)
    y = x_sorted - theta_hat
    range_min = eta_theta_minus(x, theta_hat, m, delta)
    range_max = eta_theta_plus(x, theta_hat, m, delta)
    count_0 = np.sum((y <= delta) & (y >= -delta))
    counts = []
    # get the y_i between [-delta-range_min, -delta+range_max]
    starting_points = y[(y >= -delta - range_min) & (y <= -delta + range_max)]
    for point in starting_points:
        # find the number of points (sum) that lies in the range [point, point + 2 * delta]
        count = np.sum((y >= point) & (y <= point + 2 * delta))
        counts.append(count)
    # get the y_i between [delta-range_min, delta+range_max]
    ending_points = y[(y >= delta - range_min) & (y <= delta + range_max)]
    for point in ending_points:
        # find the number of points (sum) that lies in the range [point, point + 2 * delta]
        count = np.sum((y <= point) & (y >= point - 2 * delta))
        counts.append(count)
    if counts == []:
        return 0, 0
    # find the maximum count
    max_count = max(counts)
    min_count = min(counts)
    return max_count - count_0, min_count - count_0

import numpy as np

def upper_num_shift_envelope(x, center, m, eta, delta=1.0, loss_type='huber',
                             n_grid=401, refine=2, return_t=False,
                             psi_cap_sq=None):
    """
    Finite-sample upper bound for
        sum_i psi(\tilde x_i - \tilde theta)^2
    over all contaminations changing at most m points and all shifts
        \tilde theta - center in [-eta, eta].

    Numerical strategy.
    1. Build a scouting grid on [-eta, eta].
    2. Detect candidate piece boundaries where the active trimmed set changes.
    3. On each locally stable piece, use the derivative
           d/dt sum_{i in A} psi(r_i - t)^2
         = -2 sum_{i in A} psi(r_i - t) psi_prime(r_i - t)
       and solve for critical points.
    4. Evaluate only boundaries and critical points.

    For a fixed t, the exact upper bound is:
        m * psi_cap_sq + sum of the largest n-m values of psi(x_i - center - t)^2.

    Notes.
    - For Huber, explicit kinks t = r_i ± delta are added.
    - This is a derivative-assisted numerical envelope for general losses.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    k = n - m
    r = x - center

    if psi_cap_sq is None:
        psi_cap_sq = delta ** 2

    if m <= 0:
        val = float(np.sum(psi(r, delta, loss_type=loss_type) ** 2))
        return (val, 0.0) if return_t else val

    if m >= n:
        val = float(n * psi_cap_sq)
        return (val, 0.0) if return_t else val

    if eta <= 0:
        vals = psi(r, delta, loss_type=loss_type) ** 2
        vals.sort()
        val = float(m * psi_cap_sq + np.sum(vals[m:]))
        return (val, 0.0) if return_t else val

    try:
        from scipy.optimize import brentq
    except Exception:
        brentq = None

    piece_tol = 1e-10
    root_tol = 1e-10

    def sq_vals(t):
        u = r - t
        p = psi(u, delta, loss_type=loss_type)
        return p * p

    def dsq_vals(t):
        u = r - t
        p = psi(u, delta, loss_type=loss_type)
        pp = psi_prime(u, delta, loss_type=loss_type)
        return -2.0 * p * pp

    def fixed_t_upper(t):
        vals = sq_vals(t)
        vals.sort()
        return float(m * psi_cap_sq + np.sum(vals[m:]))

    def active_set(t):
        vals = sq_vals(t)
        idx = np.argsort(vals, kind='mergesort')[m:]
        return tuple(idx.tolist())

    def piece_derivative(t, idx):
        dv = dsq_vals(t)
        return float(np.sum(dv[list(idx)]))

    def solve_on_window(left, right, grid_size):
        pts = np.linspace(left, right, max(int(grid_size), 51))

        if loss_type == 'huber':
            kinks = np.concatenate([r - delta, r + delta])
            kinks = kinks[(kinks >= left) & (kinks <= right)]
            if kinks.size:
                pts = np.concatenate([pts, kinks])

        pts = np.unique(np.sort(pts))

        best_val = -np.inf
        best_t = left

        for t in pts:
            val = fixed_t_upper(float(t))
            if val > best_val:
                best_val = val
                best_t = float(t)

        for a, b in zip(pts[:-1], pts[1:]):
            if b - a <= piece_tol:
                continue

            tm = 0.5 * (a + b)
            idx = active_set(tm)

            val_mid = fixed_t_upper(tm)
            if val_mid > best_val:
                best_val = val_mid
                best_t = float(tm)

            a_in = np.nextafter(a, b)
            b_in = np.nextafter(b, a)
            if not (a_in < b_in):
                continue

            idx_left = active_set(a_in)
            idx_mid = idx
            idx_right = active_set(b_in)

            if not (idx_left == idx_mid == idx_right):
                continue

            fa = piece_derivative(a_in, idx)
            fb = piece_derivative(b_in, idx)

            for t in (a_in, b_in):
                val = fixed_t_upper(t)
                if val > best_val:
                    best_val = val
                    best_t = float(t)

            if brentq is not None and np.isfinite(fa) and np.isfinite(fb):
                if abs(fa) <= root_tol:
                    val = fixed_t_upper(a_in)
                    if val > best_val:
                        best_val = val
                        best_t = float(a_in)

                if abs(fb) <= root_tol:
                    val = fixed_t_upper(b_in)
                    if val > best_val:
                        best_val = val
                        best_t = float(b_in)

                if fa * fb < 0:
                    try:
                        t_star = brentq(lambda t: piece_derivative(t, idx), a_in, b_in)
                        val_star = fixed_t_upper(t_star)
                        if val_star > best_val:
                            best_val = float(val_star)
                            best_t = float(t_star)
                    except Exception:
                        pass

        return best_val, best_t

    best_val, best_t = solve_on_window(-eta, eta, n_grid)

    if refine is None:
        refine = 1
    refine = max(int(refine), 1)

    if refine > 1:
        width = 2.0 * eta / max(n_grid - 1, 1)
        for _ in range(refine - 1):
            left = max(-eta, best_t - 4.0 * width)
            right = min(eta, best_t + 4.0 * width)
            local_val, local_t = solve_on_window(left, right, max(2 * n_grid, 101))
            if local_val > best_val:
                best_val = local_val
                best_t = local_t
            width *= 0.5

    return (best_val, best_t) if return_t else best_val

def eta_sigma_plus(x, theta_hat, m, delta=1.0, null=False, loss_type='huber',
                   n_grid=401, refine=2, debug=False):
    n = len(x)
    sigma_hat = estimate_sigma(x, theta_hat, delta, loss_type=loss_type)

    if m == 0:
        return (0.0, {}) if debug else 0.0

    if loss_type == 'huber':
        y = np.abs(x - (0 if null else theta_hat))
        order = np.argsort(y)
        attacked = order[:m]
        remaining = order[m:]

        if null:
            num = np.sum(psi(x[remaining], delta, loss_type=loss_type) ** 2) + m * (delta ** 2)
            den = np.sum(psi_prime(x[remaining], delta, loss_type=loss_type)) ** 2
            if den <= 1e-8:
                return (np.inf, {"num": num, "den": den}) if debug else np.inf
            out = np.sqrt(num / den) - sigma_hat
            return (out, {"num": num, "den": den}) if debug else out

        eta_plus = eta_theta_plus(x, theta_hat, m, delta, loss_type=loss_type)
        eta_minus = eta_theta_minus(x, theta_hat, m, delta, loss_type=loss_type)
        eta = max(eta_plus, eta_minus)

        num, t_star = upper_num_shift_envelope(
            x=x,
            center=theta_hat,
            m=m,
            eta=eta,
            delta=delta,
            loss_type=loss_type,
            n_grid=n_grid,
            refine=refine,
            return_t=True,
            psi_cap_sq=delta ** 2,
        )

        _, lower = find_diff_psi_prime(x, theta_hat, m, delta)
        den = (
            max(
                np.sum(psi_prime(x[remaining] - theta_hat, delta, loss_type=loss_type))
                + (lower - m) * delta,
                0.0,
            )
        ) ** 2

        if den <= 1e-8:
            return (np.inf, {"num": num, "den": den, "eta": eta, "t_star": t_star}) if debug else np.inf

        out = np.sqrt(num / den) - sigma_hat
        info = {
            "num": num,
            "den": den,
            "eta_plus": eta_plus,
            "eta_minus": eta_minus,
            "eta": eta,
            "t_star": t_star,
        }
        return (out, info) if debug else out

    else:
        y = np.abs(x - (0 if null else theta_hat))
        order = np.argsort(y)
        attacked = order[:m]
        remaining = order[m:]

        if null:
            num = np.sum(psi(x[remaining], delta, loss_type=loss_type) ** 2) + m * (delta ** 2)
            den = np.sum(psi_prime(x[remaining], delta, loss_type=loss_type)) ** 2
            if den <= 1e-8:
                return (np.inf, {"num": num, "den": den}) if debug else np.inf
            out = np.sqrt(num / den) - sigma_hat
            return (out, {"num": num, "den": den}) if debug else out

        eta_plus = eta_theta_plus(x, theta_hat, m, delta, loss_type=loss_type)
        eta_minus = eta_theta_minus(x, theta_hat, m, delta, loss_type=loss_type)
        eta = max(eta_plus, eta_minus)

        num, t_star = upper_num_shift_envelope(
            x=x,
            center=theta_hat,
            m=m,
            eta=eta,
            delta=delta,
            loss_type=loss_type,
            n_grid=n_grid,
            refine=refine,
            return_t=True,
            psi_cap_sq=delta ** 2,
        )

        endpoints = 2 * x - theta_hat
        endpoints = np.sort(endpoints)
        p1 = np.sum(theta_hat + eta_plus >= endpoints[:n - m])
        p2 = np.sum(theta_hat - eta_minus <= endpoints[m:])
        p = max(p1, p2)
        diff = Delta(eta, delta, loss_type=loss_type)

        den = (
            max(
                np.sum(psi_prime(x[remaining] - theta_hat, delta, loss_type=loss_type))
                - p * diff,
                0.0,
            )
        ) ** 2

        if den <= 1e-8:
            return (np.inf, {"num": num, "den": den, "eta": eta, "t_star": t_star}) if debug else np.inf

        out = np.sqrt(num / den) - sigma_hat
        info = {
            "num": num,
            "den": den,
            "eta_plus": eta_plus,
            "eta_minus": eta_minus,
            "eta": eta,
            "t_star": t_star,
        }
        return (out, info) if debug else out

# # --- Update eta_sigma_plus ---
# def eta_sigma_plus(x, theta_hat, m, delta=1.0, null=False, loss_type='huber'):
#     n = len(x)
#     sigma_hat = estimate_sigma(x, theta_hat, delta, loss_type=loss_type)
#     if m == 0:
#         return 0.0
#     if loss_type == 'huber':
#         # (existing Huber logic)
#         y = np.abs(x - (0 if null else theta_hat))
#         order = np.argsort(y)
#         attacked = order[:m]
#         remaining = order[m:]
#         if null:
#             num = np.sum(psi(x[remaining], delta, loss_type=loss_type) ** 2) + m * (delta ** 2)
#             den = np.sum(psi_prime(x[remaining], delta, loss_type=loss_type)) ** 2
#         else:
#             eta = max(eta_theta_plus(x, theta_hat, m, delta, loss_type=loss_type), eta_theta_minus(x, theta_hat, m, delta, loss_type=loss_type))
#             num = np.sum(psi(x[remaining] - theta_hat, delta, loss_type=loss_type) ** 2) + 5 * m * (delta ** 2)
#             _, lower = find_diff_psi_prime(x, theta_hat, m, delta)
#             # print(f"outliers: {outliers}")
#             den = (max(np.sum(psi_prime(x[remaining] - theta_hat, delta, loss_type=loss_type)) + (lower - m) * delta, 0))**2
#         if den <= 1e-8:
#             return np.inf
#         return np.sqrt(num / den) - sigma_hat
#     else:
#         n = len(x)
#         y = np.abs(x - (0 if null else theta_hat))
#         order = np.argsort(y)
#         attacked = order[:m]
#         remaining = order[m:]
#         eta = max(eta_theta_plus(x, theta_hat, m, delta, loss_type=loss_type), eta_theta_minus(x, theta_hat, m, delta, loss_type=loss_type))
#         if null:
#             num = np.sum(psi(x[remaining], delta, loss_type=loss_type) ** 2) + m * (delta ** 2)
#             den = np.sum(psi_prime(x[remaining], delta, loss_type=loss_type)) ** 2
#         else:
#             num = np.sum(psi(x[remaining] - theta_hat, delta, loss_type=loss_type) ** 2) + 5 * m * (delta ** 2)
#             endpoints = 2 * x - theta_hat
#             endpoints = np.sort(endpoints)
#             p1 = np.sum(theta_hat + eta_theta_plus(x, theta_hat, m, delta, loss_type=loss_type) >= endpoints[:n-m])
#             p2 = np.sum(theta_hat - eta_theta_minus(x, theta_hat, m, delta, loss_type=loss_type) <= endpoints[m:])
#             p = max(p1, p2)
#             diff = Delta(eta, delta, loss_type=loss_type)
#             den = (max(np.sum(psi_prime(x[remaining] - theta_hat, delta, loss_type=loss_type)) - p * diff, 0)) ** 2
#         if den <= 1e-8:
#             return np.inf
#         return np.sqrt(num / den) - sigma_hat


import numpy as np

def lower_num_shift_envelope(x, center, m, eta, delta=1.0, loss_type='huber',
                             n_grid=401, refine=2, return_t=False):
    """
    Finite-sample lower bound for
        sum_i psi(\tilde x_i - \tilde theta)^2
    over all contaminations changing at most m points and all shifts
        \tilde theta - center in [-eta, eta].

    Numerical strategy.
    1. Build a scouting grid on [-eta, eta].
    2. Detect candidate piece boundaries where the active trimmed set changes.
    3. On each locally stable piece, use the derivative
           d/dt sum_{i in A} psi(r_i - t)^2
         = -2 sum_{i in A} psi(r_i - t) psi_prime(r_i - t)
       and solve for critical points.
    4. Evaluate only boundaries and critical points.

    Notes.
    - For Huber, explicit kinks t = r_i ± delta are added.
    - This is exact for Huber only if all true switching points are captured.
      For general smooth losses it is a derivative-assisted numerical envelope,
      not a symbolic exact solver.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    k = n - m
    r = x - center

    if m <= 0:
        val = float(np.sum(psi(r, delta, loss_type=loss_type) ** 2))
        return (val, 0.0) if return_t else val

    if m >= n:
        return (0.0, 0.0) if return_t else 0.0

    if eta <= 0:
        vals = psi(r, delta, loss_type=loss_type) ** 2
        vals.sort()
        val = float(np.sum(vals[:k]))
        return (val, 0.0) if return_t else val

    try:
        from scipy.optimize import brentq
    except Exception:
        brentq = None

    # Conservative tolerances.
    piece_tol = 1e-10
    root_tol = 1e-10

    def sq_vals(t):
        u = r - t
        p = psi(u, delta, loss_type=loss_type)
        return p * p

    def dsq_vals(t):
        u = r - t
        p = psi(u, delta, loss_type=loss_type)
        pp = psi_prime(u, delta, loss_type=loss_type)
        return -2.0 * p * pp

    def fixed_t_lower(t):
        vals = sq_vals(t)
        vals.sort()
        return float(np.sum(vals[:k]))

    def active_set(t):
        vals = sq_vals(t)
        # stable full sort is safer than argpartition near ties
        idx = np.argsort(vals, kind='mergesort')[:k]
        return tuple(idx.tolist())

    def piece_derivative(t, idx):
        dv = dsq_vals(t)
        return float(np.sum(dv[list(idx)]))

    def solve_on_window(left, right, grid_size):
        # Build scouting points.
        pts = np.linspace(left, right, max(int(grid_size), 51))

        # Add explicit kinks for Huber.
        if loss_type == 'huber':
            kinks = np.concatenate([r - delta, r + delta])
            kinks = kinks[(kinks >= left) & (kinks <= right)]
            if kinks.size:
                pts = np.concatenate([pts, kinks])

        pts = np.unique(np.sort(pts))

        cand_t = [left, right]
        best_val = np.inf
        best_t = left

        # Always check all scouting points themselves.
        for t in pts:
            val = fixed_t_lower(float(t))
            if val < best_val:
                best_val = val
                best_t = float(t)

        # Work interval by interval.
        for a, b in zip(pts[:-1], pts[1:]):
            if b - a <= piece_tol:
                continue

            tm = 0.5 * (a + b)
            idx = active_set(tm)

            # Evaluate midpoint as a cheap safety check.
            val_mid = fixed_t_lower(tm)
            if val_mid < best_val:
                best_val = val_mid
                best_t = float(tm)

            # Check whether the active set looks stable inside this interval.
            a_in = np.nextafter(a, b)
            b_in = np.nextafter(b, a)
            if not (a_in < b_in):
                continue

            idx_left = active_set(a_in)
            idx_mid = idx
            idx_right = active_set(b_in)

            if not (idx_left == idx_mid == idx_right):
                continue

            # On this interval, use the derivative of the fixed active-set sum.
            fa = piece_derivative(a_in, idx)
            fb = piece_derivative(b_in, idx)

            # Evaluate near boundaries too.
            for t in (a_in, b_in):
                val = fixed_t_lower(t)
                if val < best_val:
                    best_val = val
                    best_t = float(t)

            # Root of derivative inside the interval.
            if brentq is not None and np.isfinite(fa) and np.isfinite(fb):
                # Exact zero at an endpoint.
                if abs(fa) <= root_tol:
                    val = fixed_t_lower(a_in)
                    if val < best_val:
                        best_val = val
                        best_t = float(a_in)

                if abs(fb) <= root_tol:
                    val = fixed_t_lower(b_in)
                    if val < best_val:
                        best_val = val
                        best_t = float(b_in)

                # Sign change means a critical point.
                if fa * fb < 0:
                    try:
                        t_star = brentq(lambda t: piece_derivative(t, idx), a_in, b_in)
                        val_star = fixed_t_lower(t_star)
                        cand_t.append(float(t_star))
                        if val_star < best_val:
                            best_val = float(val_star)
                            best_t = float(t_star)
                    except Exception:
                        pass

        return best_val, best_t

    # Global pass.
    best_val, best_t = solve_on_window(-eta, eta, n_grid)

    # Optional local polishing around the current best point.
    # This is a numerical refinement only.
    if refine is None:
        refine = 1
    refine = max(int(refine), 1)

    if refine > 1:
        width = 2.0 * eta / max(n_grid - 1, 1)
        for _ in range(refine - 1):
            left = max(-eta, best_t - 4.0 * width)
            right = min(eta, best_t + 4.0 * width)
            local_val, local_t = solve_on_window(left, right, max(2 * n_grid, 101))
            if local_val < best_val:
                best_val = local_val
                best_t = local_t
            width *= 0.5

    return (best_val, best_t) if return_t else best_val


def eta_sigma_minus(x, theta_hat, m, delta=1.0, null=False, loss_type='huber',
                    n_grid=401, refine=2, debug=False):
    n = len(x)
    sigma_hat = estimate_sigma(x, theta_hat, delta, loss_type=loss_type)

    if m == 0:
        return (0.0, {}) if debug else 0.0

    if loss_type == 'huber':
        y = np.abs(x - (0 if null else theta_hat))
        order = np.argsort(y)
        remaining = order[:n - m]

        if null:
            num = np.sum(psi(x[remaining], delta, loss_type=loss_type) ** 2)
            den = (
                np.sum(psi_prime(x[remaining], delta, loss_type=loss_type)) + m * delta
            ) ** 2
            out = sigma_hat - np.sqrt(num / den)
            return (out, {"num": num, "den": den}) if debug else out

        # Use the true finite-sample lower envelope for the numerator.
        eta_plus = eta_theta_plus(x, theta_hat, m, delta, loss_type=loss_type)
        eta_minus = eta_theta_minus(x, theta_hat, m, delta, loss_type=loss_type)
        eta = max(eta_plus, eta_minus)

        num, t_star = lower_num_shift_envelope(
            x=x,
            center=theta_hat,
            m=m,
            eta=eta,
            delta=delta,
            loss_type=loss_type,
            n_grid=n_grid,
            refine=refine,
            return_t=True,
        )

        if num <= 0:
            return (sigma_hat, {"num": num, "den": None, "eta": eta, "t_star": t_star}) if debug else sigma_hat

        upper, _ = find_diff_psi_prime(x, theta_hat, m, delta)
        den = (
            m * delta
            + np.sum(psi_prime(x[remaining] - theta_hat, delta, loss_type=loss_type))
            + (upper + m) * delta
        ) ** 2

        out = sigma_hat - np.sqrt(num / den)
        info = {
            "num": num,
            "den": den,
            "eta_plus": eta_plus,
            "eta_minus": eta_minus,
            "eta": eta,
            "t_star": t_star,
        }
        return (out, info) if debug else out

    else:
        y = np.abs(x - (0 if null else theta_hat))
        order = np.argsort(y)
        remaining = order[:n - m]

        if null:
            num = np.sum(psi(x[remaining], delta, loss_type=loss_type) ** 2)
            den = (
                np.sum(psi_prime(x[remaining], delta, loss_type=loss_type))
                + m * delta
            ) ** 2
            out = sigma_hat - np.sqrt(num / den)
            return (out, {"num": num, "den": den}) if debug else out

        eta_plus = eta_theta_plus(x, theta_hat, m, delta, loss_type=loss_type)
        eta_minus = eta_theta_minus(x, theta_hat, m, delta, loss_type=loss_type)
        eta = max(eta_plus, eta_minus)

        num, t_star = lower_num_shift_envelope(
            x=x,
            center=theta_hat,
            m=m,
            eta=eta,
            delta=delta,
            loss_type=loss_type,
            n_grid=n_grid,
            refine=refine,
            return_t=True,
        )

        if num <= 0:
            return (sigma_hat, {"num": num, "den": None, "eta": eta, "t_star": t_star}) if debug else sigma_hat

        p1 = np.sum(x >= theta_hat)
        p2 = np.sum(x <= theta_hat)
        p = max(p1, p2) - m
        diff = Delta(eta, delta, loss_type=loss_type)

        den = (
            m * psi_prime(0, delta, loss_type=loss_type)
            + np.sum(psi_prime(x[remaining] - theta_hat, delta, loss_type=loss_type))
            + p * diff
        ) ** 2

        out = sigma_hat - np.sqrt(num / den)
        info = {
            "num": num,
            "den": den,
            "eta_plus": eta_plus,
            "eta_minus": eta_minus,
            "eta": eta,
            "t_star": t_star,
        }
        return (out, info) if debug else out

# # --- Update eta_sigma_minus ---
# def eta_sigma_minus(x, theta_hat, m, delta=1.0, null=False, loss_type='huber'):
#     n = len(x)
#     sigma_hat = estimate_sigma(x, theta_hat, delta, loss_type=loss_type)
#     if m == 0:
#         return 0.0
#     if loss_type == 'huber':
#         # (existing Huber logic)
#         y = np.abs(x - (0 if null else theta_hat))
#         order = np.argsort(y)
#         remaining = order[:len(x)-m]
#         if null:
#             num = np.sum(psi(x[remaining], delta, loss_type=loss_type) ** 2)
#             den = np.sum(psi_prime(x[remaining], delta, loss_type=loss_type) + m * delta) ** 2
#         else:
#             eta = max(eta_theta_plus(x, theta_hat, m, delta, loss_type=loss_type), eta_theta_minus(x, theta_hat, m, delta, loss_type=loss_type))
#             num = np.sum(psi(x[remaining] - theta_hat, delta, loss_type=loss_type) ** 2) - 4 * m * delta ** 2
#             if num <= 0:
#                 return sigma_hat
#             upper, _ = find_diff_psi_prime(x, theta_hat, m, delta)
#             den = (m * delta + np.sum(psi_prime(x[remaining] - theta_hat, delta, loss_type=loss_type)) +(upper + m) * delta)**2
#         return sigma_hat - np.sqrt(num / den)
#     else:
#         n = len(x)
#         y = np.abs(x - (0 if null else theta_hat))
#         order = np.argsort(y)
#         remaining = order[:n-m]
#         eta = max(eta_theta_plus(x, theta_hat, m, delta, loss_type=loss_type), eta_theta_minus(x, theta_hat, m, delta, loss_type=loss_type))
#         if null:
#             num = np.sum(psi(x[remaining], delta, loss_type=loss_type) ** 2)
#             den = np.sum(psi_prime(x[remaining], delta, loss_type=loss_type) + m * psi_prime(0, delta, loss_type=loss_type)) ** 2
#         else:
#             num = np.sum(psi(x[remaining] - theta_hat, delta, loss_type=loss_type) ** 2)  - 4 * m * delta ** 2
#             p1 = np.sum(x >= theta_hat)
#             p2 = np.sum(x <= theta_hat)
#             p = max(p1, p2) - m
#             diff = Delta(eta, delta, loss_type=loss_type)
#             den = (m * psi_prime(0, delta, loss_type=loss_type) + np.sum(psi_prime(x[remaining] - theta_hat, delta, loss_type=loss_type)) + p * diff) ** 2
#             if num <= 0:
#                 return sigma_hat
#         return sigma_hat - np.sqrt(num / den)
    

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
    theta_hat = estimate_theta(x, delta, loss_type=loss_type)
    sigma_hat = estimate_sigma(x, theta_hat, delta, loss_type=loss_type)
    z = norm.ppf(1 - alpha/2)
    x_sorted = np.sort(x)
    if theta_hat - z * sigma_hat > 0:
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
    theta_hat = estimate_theta(x, delta, loss_type=loss_type)
    sigma_hat = estimate_sigma(x, theta_hat, delta, null=null, loss_type=loss_type)
    z = norm.ppf(1 - alpha/2)
    if theta_hat - z * sigma_hat > 0:
        for m in range(1, n+1):
            eta_t_minus = eta_theta_minus(x, theta_hat, m, delta, loss_type=loss_type)
            eta_s_plus = eta_sigma_plus(x, theta_hat, m, delta, null, loss_type=loss_type)
            if eta_t_minus + z * eta_s_plus >= theta_hat - z * sigma_hat:
                return m
    else:
        for m in range(1, n+1):
            eta_t_plus = eta_theta_plus(x, theta_hat, m, delta, loss_type=loss_type)
            eta_s_plus = eta_sigma_plus(x, theta_hat, m, delta, null, loss_type=loss_type)
            if eta_t_plus + z * eta_s_plus >= -theta_hat - z * sigma_hat:
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
    theta_hat = estimate_theta(x, delta, loss_type=loss_type)
    sigma_hat = estimate_sigma(x, theta_hat, delta, null, loss_type=loss_type)
    z = norm.ppf(1 - alpha/2)
    m1 = m2 = n
    for m in range(1, n+1):
        eta_t_minus = eta_theta_minus(x, theta_hat, m, delta, loss_type=loss_type)
        eta_s_minus = eta_sigma_minus(x, theta_hat, m, delta, null, loss_type=loss_type)
        if eta_t_minus + z * eta_s_minus >= theta_hat + z * sigma_hat:
            m1 = m
            break
    for m in range(1, n+1):
        eta_t_plus = eta_theta_plus(x, theta_hat, m, delta, loss_type=loss_type)
        eta_s_minus = eta_sigma_minus(x, theta_hat, m, delta, null, loss_type=loss_type)
        if eta_t_plus + z * eta_s_minus >= -theta_hat + z * sigma_hat:
            m2 = m
            break
    return min(m1, m2)

# -- Two-sample stuff --
def wald_test_two_sample(x, y, delta=1.0, alpha=0.05, null=False, loss_type='huber'):
    theta_hat_x = estimate_theta(x, delta, loss_type=loss_type)
    theta_hat_y = estimate_theta(y, delta, loss_type=loss_type)
    if null:
        sigma_hat = two_sample_sigma_hat(x, y, 0, 0, delta=delta, loss_type=loss_type)
    else:
        sigma_hat = two_sample_sigma_hat(x, y, theta_hat_x, theta_hat_y, delta=delta, loss_type=loss_type)
    T = (theta_hat_x - theta_hat_y) / sigma_hat
    return int(np.abs(T) > norm.ppf(1 - alpha/2))

def two_sample_sigma_hat(x, y, theta_hat_x, theta_hat_y, delta=1.0, loss_type='huber'):
    sigma_hat_x = estimate_sigma(x, theta_hat_x, delta, loss_type=loss_type)
    sigma_hat_y = estimate_sigma(y, theta_hat_y, delta, loss_type=loss_type)
    return np.sqrt(sigma_hat_x ** 2 + sigma_hat_y ** 2)

# --- Two-sample eta_theta_plus ---
def eta_theta_plus_two_sample(x, y, theta_hat_x, theta_hat_y, k1, k2, delta=1.0, loss_type='huber'):
    return eta_theta_plus(x, theta_hat_x, k1, delta, loss_type=loss_type) + eta_theta_minus(y, theta_hat_y, k2, delta, loss_type=loss_type)

# --- Two-sample eta_theta_minus ---
def eta_theta_minus_two_sample(x, y, theta_hat_x, theta_hat_y, k1, k2, delta=1.0, loss_type='huber'):
    return eta_theta_minus(x, theta_hat_x, k1, delta, loss_type=loss_type) + eta_theta_plus(y, theta_hat_y, k2, delta, loss_type=loss_type)

# --- Two-sample eta_sigma_plus ---
def eta_sigma_plus_two_sample(x, y, theta_hat_x, theta_hat_y, sigma_hat, k1, k2, delta=1.0, loss_type='huber'):
    return np.sqrt((eta_sigma_plus(x, theta_hat_x, k1, delta, loss_type=loss_type) + estimate_sigma(x, theta_hat_x, delta, loss_type=loss_type)) ** 2 + (eta_sigma_plus(y, theta_hat_y, k2, delta, loss_type=loss_type) + estimate_sigma(y, theta_hat_y, delta, loss_type=loss_type)) ** 2) - sigma_hat

# --- Two-sample eta_sigma_minus ---
def eta_sigma_minus_two_sample(x, y, theta_hat_x, theta_hat_y, sigma_hat, k1, k2, delta=1.0, loss_type='huber'):
    return sigma_hat - np.sqrt((-eta_sigma_minus(x, theta_hat_x, k1, delta, loss_type=loss_type) + estimate_sigma(x, theta_hat_x, delta, loss_type=loss_type)) ** 2 + (-eta_sigma_minus(y, theta_hat_y, k2, delta, loss_type=loss_type) + estimate_sigma(y, theta_hat_y, delta, loss_type=loss_type)) ** 2)


# ----------- Upper bound for power BP (Corollary 6) -----------
def bound_power_upper_two_sample(x, y, delta=1.0, alpha=0.05, loss_type='huber'):
    n1 = len(x)
    n2 = len(y)
    theta_hat_x = estimate_theta(x, delta, loss_type=loss_type)
    theta_hat_y = estimate_theta(y, delta, loss_type=loss_type)
    theta_diff = theta_hat_x - theta_hat_y
    sigma_hat = two_sample_sigma_hat(x, y, theta_hat_x, theta_hat_y, delta, loss_type)
    z = norm.ppf(1 - alpha/2)
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    for m in range(1, min(n1, n2)+1):
        if theta_diff > 0:
            k1_min, k1_max = max(0, m - n2), min(m, n1)
            for k1 in range(k1_min, k1_max + 1):
                k2 = m - k1
                # Replace m largest y with +inf, m smallest x with -inf
                x_new = np.concatenate([x_sorted[:n1-k1], -np.ones(k1) * LARGE_SENTINEL])
                y_new = np.concatenate([y_sorted[k2:], np.ones(k2) * LARGE_SENTINEL])
                theta_hat_x_new = estimate_theta(x_new, delta, loss_type=loss_type)
                theta_hat_y_new = estimate_theta(y_new, delta, loss_type=loss_type)
                sigma_hat_new = two_sample_sigma_hat(x_new, y_new, theta_hat_x_new, theta_hat_y_new, delta, loss_type)
                if (theta_hat_x_new - theta_hat_y_new) - z * sigma_hat_new <= 0:
                    return m, (theta_hat_x_new - theta_hat_y_new) / sigma_hat_new
        else:
            k1_min, k1_max = max(0, m - n2), min(m, n1)
            for k1 in range(k1_min, k1_max + 1):
                k2 = m - k1
                # Replace m smallest x with +inf, m largest y with -inf
                x_new = np.concatenate([x_sorted[k1:], np.ones(k1) * LARGE_SENTINEL])
                y_new = np.concatenate([y_sorted[:n2-k2], -np.ones(k2) * LARGE_SENTINEL])
                theta_hat_x_new = estimate_theta(x_new, delta, loss_type=loss_type)
                theta_hat_y_new = estimate_theta(y_new, delta, loss_type=loss_type)
                sigma_hat_new = two_sample_sigma_hat(x_new, y_new, theta_hat_x_new, theta_hat_y_new, delta, loss_type)
                if (theta_hat_x_new - theta_hat_y_new) + z * sigma_hat_new >= 0:
                    return m, (theta_hat_x_new - theta_hat_y_new) / sigma_hat_new
    return np.nan

# ----------- Lower bound for power BP (Corollary 6) -----------
def bound_power_lower_two_sample(x, y, delta=1.0, alpha=0.05, loss_type='huber'):
    n1 = len(x)
    n2 = len(y)
    theta_hat_x = estimate_theta(x, delta, loss_type=loss_type)
    theta_hat_y = estimate_theta(y, delta, loss_type=loss_type)
    theta_diff = theta_hat_x - theta_hat_y
    sigma_hat = two_sample_sigma_hat(x, y, theta_hat_x, theta_hat_y, delta, loss_type)
    z = norm.ppf(1 - alpha/2)
    if theta_diff > 0:
        for m in range(1, min(n1, n2)+1):
            k1_min, k1_max = max(0, m - n2), min(m, n1)
            for k1 in range(k1_min, k1_max + 1):
                k2 = m - k1
                eta_t_minus = eta_theta_minus_two_sample(x, y, theta_hat_x, theta_hat_y, k1, k2, delta, loss_type=loss_type)
                eta_s_plus = eta_sigma_plus_two_sample(x, y, theta_hat_x, theta_hat_y, sigma_hat, k1, k2, delta, loss_type=loss_type)
                if eta_t_minus + z * eta_s_plus >= theta_diff - z * sigma_hat:
                    return m, (theta_diff - eta_t_minus) / (sigma_hat + eta_s_plus)
    else:
        for m in range(1, min(n1, n2)+1):
            k1_min, k1_max = max(0, m - n2), min(m, n1)
            for k1 in range(k1_min, k1_max + 1):
                k2 = m - k1
                eta_t_plus = eta_theta_plus_two_sample(x, y, theta_hat_x, theta_hat_y, k1, k2, delta, loss_type=loss_type)
                eta_s_plus = eta_sigma_plus_two_sample(x, y, theta_hat_x, theta_hat_y, sigma_hat, k1, k2, delta, loss_type=loss_type)
                if eta_t_plus + z * eta_s_plus >= -theta_diff - z * sigma_hat:
                    return m, (theta_diff + eta_t_plus) / (sigma_hat + eta_s_plus)
    return np.nan

    # ----------- Upper bound for level BP (two-sample) -----------
def bound_level_upper_two_sample(x, y, delta=1.0, alpha=0.05, loss_type='huber'):
    n1 = len(x)
    n2 = len(y)
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    z = norm.ppf(1 - alpha/2)
    m1 = m2 = min(n1, n2)
    for m in range(1, min(n1, n2)+1):
        k1_min, k1_max = max(0, m - n2), min(m, n1)
        for k1 in range(k1_min, k1_max + 1):
            k2 = m - k1
            # Case 1: replace m smallest x with x_min, m largest y with y_max
            x_new = np.concatenate([x_sorted[:n1-k1], np.full(k1, x_sorted[0])])
            y_new = np.concatenate([y_sorted[k2:], np.full(k2, y_sorted[-1])])
            theta_hat_x_new = estimate_theta(x_new, delta, loss_type=loss_type)
            theta_hat_y_new = estimate_theta(y_new, delta, loss_type=loss_type)
            sigma_hat_new = two_sample_sigma_hat(x_new, y_new, theta_hat_x_new, theta_hat_y_new, delta, loss_type)
            if (theta_hat_x_new - theta_hat_y_new) + z * sigma_hat_new <= 0:
                m1 = m 
                stat1 = (theta_hat_x_new - theta_hat_y_new) / sigma_hat_new
                break
    for m in range(1, min(n1, n2)+1):
        k1_min, k1_max = max(0, m - n2), min(m, n1)
        for k1 in range(k1_min, k1_max + 1):
            k2 = m - k1
        # Case 2: replace m largest x with x_max, m smallest y with y_min
            x_new = np.concatenate([x_sorted[k1:], np.full(k1, x_sorted[-1])])
            y_new = np.concatenate([y_sorted[:n2-k2], np.full(k2, y_sorted[0])])
            theta_hat_x_new = estimate_theta(x_new, delta, loss_type=loss_type)
            theta_hat_y_new = estimate_theta(y_new, delta, loss_type=loss_type)
            sigma_hat_new = two_sample_sigma_hat(x_new, y_new, theta_hat_x_new, theta_hat_y_new, delta, loss_type)
            if (theta_hat_x_new - theta_hat_y_new) - z * sigma_hat_new >= 0:
                m2 = m
                stat2 = (theta_hat_x_new - theta_hat_y_new) / sigma_hat_new
                break
    return min(m1, m2), stat1 if m1 < m2 else stat2

# ----------- Lower bound for level BP (two-sample) -----------
def bound_level_lower_two_sample(x, y, delta=1.0, alpha=0.05, loss_type='huber'):
    n1 = len(x)
    n2 = len(y)
    theta_hat_x = estimate_theta(x, delta, loss_type=loss_type)
    theta_hat_y = estimate_theta(y, delta, loss_type=loss_type)
    theta_diff = theta_hat_x - theta_hat_y
    sigma_hat = two_sample_sigma_hat(x, y, theta_hat_x, theta_hat_y, delta, loss_type)
    z = norm.ppf(1 - alpha/2)
    m1 = m2 = min(n1, n2)
    for m in range(1, min(n1, n2)+1):
        k1_min, k1_max = max(0, m - n2), min(m, n1)
        for k1 in range(k1_min, k1_max + 1):
            k2 = m - k1
            eta_t_minus = eta_theta_minus_two_sample(x, y, theta_hat_x, theta_hat_y, k1, k2, delta, loss_type=loss_type)
            eta_s_minus = eta_sigma_minus_two_sample(x, y, theta_hat_x, theta_hat_y, sigma_hat, k1, k2, delta, loss_type=loss_type)
            if eta_t_minus + z * eta_s_minus >= theta_diff + z * sigma_hat:
                m1 = m
                stat1 = (theta_diff - eta_t_minus) / (sigma_hat - eta_s_minus)
                break
    for m in range(1, min(n1, n2)+1):
        k1_min, k1_max = max(0, m - n2), min(m, n1)
        for k1 in range(k1_min, k1_max + 1):
            k2 = m - k1
            eta_t_plus = eta_theta_plus_two_sample(x, y, theta_hat_x, theta_hat_y, k1, k2, delta, loss_type=loss_type)
            eta_s_minus = eta_sigma_minus_two_sample(x, y, theta_hat_x, theta_hat_y, sigma_hat, k1, k2, delta, loss_type=loss_type)
            if eta_t_plus + z * eta_s_minus >= -theta_diff + z * sigma_hat:
                m2 = m
                stat2 = (theta_diff + eta_t_plus) / (sigma_hat - eta_s_minus)
                break
    return min(m1, m2), stat1 if m1 < m2 else stat2


def two_sample_test_statistic(x, y, m, delta=1.0, loss_type='huber', bp_type='power'):
    # computing upper and lower bounds for the statistic after m attacks
    # bp_type: 'power', 'level'
    n1 = len(x)
    n2 = len(y)
    theta_hat_x = estimate_theta(x, delta, loss_type=loss_type)
    theta_hat_y = estimate_theta(y, delta, loss_type=loss_type)
    theta_diff = theta_hat_x - theta_hat_y
    sigma_hat = two_sample_sigma_hat(x, y, theta_hat_x, theta_hat_y, delta, loss_type)
    original_stat = abs(theta_diff / sigma_hat)
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    # c_grid for three losses:
    if loss_type == 'huber':
        c_grid = list(0.1 * np.arange(1, 13) - 0.2)
    elif loss_type == 'logcosh':
        c_grid = [-0.121, 0.0, 0.121, 0.244, 0.373, 0.51, 0.662, 0.835, 1.045, 1.323, 1.774]
    elif loss_type == 'concordant':
        c_grid = [-0.15, 0.0, 0.15, 0.309, 0.488, 0.705, 0.987, 1.389, 2.033, 3.291, 7.016]
    if m == 0:
        return original_stat, original_stat, original_stat
    if bp_type in ['power']:
        k1_min, k1_max = max(0, m - n2), min(m, n1)
        # notice the lower here is for BP, so it is actually the upper for the statistic
        # and the upper here is for BP, so it is actually the lower for the statistic
        # upper bound for power BP
        power_upper_min = np.inf
        if theta_diff > 0:
            for k1 in range(k1_min, k1_max + 1):
                k2 = m - k1
                # Replace m largest x with +inf, m smallest y with -inf
                x_new = np.concatenate([x_sorted[k1:], np.ones(k1) * LARGE_SENTINEL])
                y_new = np.concatenate([y_sorted[:n2-k2], -np.ones(k2) * LARGE_SENTINEL])
                theta_hat_x_new = estimate_theta(x_new, delta, loss_type=loss_type)
                theta_hat_y_new = estimate_theta(y_new, delta, loss_type=loss_type)
                sigma_hat_new = two_sample_sigma_hat(x_new, y_new, theta_hat_x_new, theta_hat_y_new, delta, loss_type=loss_type)
                power_upper = abs((theta_hat_x_new - theta_hat_y_new) / sigma_hat_new)
                if power_upper < original_stat:
                    power_upper_min = min(power_upper_min, power_upper)
        else:
            for k1 in range(k1_min, k1_max + 1):
                k2 = m - k1
                # Replace m largest y with +inf, m smallest x with -inf
                x_new = np.concatenate([x_sorted[:n1-k1], -np.ones(k1) * LARGE_SENTINEL])
                y_new = np.concatenate([y_sorted[k2:], np.ones(k2) * LARGE_SENTINEL])
                theta_hat_x_new = estimate_theta(x_new, delta, loss_type=loss_type)
                theta_hat_y_new = estimate_theta(y_new, delta, loss_type=loss_type)
                sigma_hat_new = two_sample_sigma_hat(x_new, y_new, theta_hat_x_new, theta_hat_y_new, delta, loss_type=loss_type)
                power_upper = abs((theta_hat_x_new - theta_hat_y_new) / sigma_hat_new)
                if power_upper < original_stat:
                    power_upper_min = min(power_upper_min, power_upper)

        # lower bound for power BP
        power_lower_min = np.inf
        for k1 in range(k1_min, k1_max + 1):
            k2 = m - k1
            eta_t_minus = eta_theta_minus_two_sample(x, y, theta_hat_x, theta_hat_y, k1, k2, delta, loss_type=loss_type)
            eta_s_plus = eta_sigma_plus_two_sample(x, y, theta_hat_x, theta_hat_y, sigma_hat, k1, k2, delta, loss_type=loss_type)
            power_lower_1 = (theta_diff - eta_t_minus) / (sigma_hat + eta_s_plus)
            if power_lower_1 < original_stat:
                power_lower_min = min(power_lower_min, abs(power_lower_1))
        k1_min, k1_max = max(0, m - n2), min(m, n1)
        for k1 in range(k1_min, k1_max + 1):
            k2 = m - k1
            eta_t_plus = eta_theta_plus_two_sample(x, y, theta_hat_x, theta_hat_y, k1, k2, delta, loss_type=loss_type)
            eta_s_plus = eta_sigma_plus_two_sample(x, y, theta_hat_x, theta_hat_y, sigma_hat, k1, k2, delta, loss_type=loss_type)
            power_lower_2 = (theta_diff + eta_t_plus) / (sigma_hat + eta_s_plus)
            if power_lower_2 < original_stat:
                power_lower_min = min(power_lower_min, abs(power_lower_2))

    if bp_type in ['level']:
        # upper bound for level BP
        level_upper_max = -np.inf
        k1_min, k1_max = max(0, m - n2), min(m, n1)
        for k1 in range(k1_min, k1_max + 1):
            k2 = m - k1
            # Here we use a detailed attack strategy to get a tighter upper bound for level BP
            # Roughly speaking, we compute the attacked theta, and then move the attacked points to the closest point to the attacked theta within c * delta distance, where we prepick some c as candidates
            # And then we choose the optimal one
            attacked_theta_x = theta_hat_x - eta_theta_minus(x, theta_hat_x, k1, delta, loss_type=loss_type)
            attacked_theta_y = theta_hat_y + eta_theta_plus(y, theta_hat_y, k2, delta, loss_type=loss_type)
            for c in c_grid:  # c from -0.1 to 1.0
                x_target = attacked_theta_x - delta * c
                y_target = attacked_theta_y + delta * c
                x_new = np.concatenate([x_sorted[:n1-k1], np.full(k1, x_target)])
                y_new = np.concatenate([y_sorted[k2:], np.full(k2, y_target)])
                theta_hat_x_new = estimate_theta(x_new, delta, loss_type=loss_type)
                theta_hat_y_new = estimate_theta(y_new, delta, loss_type=loss_type)
                sigma_hat_new = two_sample_sigma_hat(x_new, y_new, theta_hat_x_new, theta_hat_y_new, delta, loss_type)
                level = abs((theta_hat_x_new - theta_hat_y_new) / sigma_hat_new)
                if level > original_stat:
                    level_upper_max = max(level_upper_max, level)
        
        k1_min, k1_max = max(0, m - n2), min(m, n1)
        for k1 in range(k1_min, k1_max + 1):
            k2 = m - k1
            # The same as above, just swap x and y
            attacked_theta_x = theta_hat_x + eta_theta_plus(x, theta_hat_x, k1, delta, loss_type=loss_type)
            attacked_theta_y = theta_hat_y - eta_theta_minus(y, theta_hat_y, k2, delta, loss_type=loss_type)
            for c in c_grid:  # c from -0.1 to 1.0
                x_target = attacked_theta_x + delta * c
                y_target = attacked_theta_y - delta * c
                x_new = np.concatenate([x_sorted[k1:], np.full(k1, x_target)])
                y_new = np.concatenate([y_sorted[:n2-k2], np.full(k2, y_target)])
                theta_hat_x_new = estimate_theta(x_new, delta, loss_type=loss_type)
                theta_hat_y_new = estimate_theta(y_new, delta, loss_type=loss_type)
                sigma_hat_new = two_sample_sigma_hat(x_new, y_new, theta_hat_x_new, theta_hat_y_new, delta, loss_type)
                level = abs((theta_hat_x_new - theta_hat_y_new) / sigma_hat_new)
                if level > original_stat:
                    level_upper_max = max(level_upper_max, level)

        # lower bound for level BP
        level_lower_max = -np.inf
        k1_min, k1_max = max(0, m - n2), min(m, n1)
        for k1 in range(k1_min, k1_max + 1):
            k2 = m - k1
            eta_t_minus = eta_theta_minus_two_sample(x, y, theta_hat_x, theta_hat_y, k1, k2, delta, loss_type=loss_type)
            eta_s_minus = eta_sigma_minus_two_sample(x, y, theta_hat_x, theta_hat_y, sigma_hat, k1, k2, delta, loss_type=loss_type)
            level = (theta_diff - eta_t_minus) / (sigma_hat - eta_s_minus)
            level_lower_max = max(level_lower_max, abs(level))

        k1_min, k1_max = max(0, m - n2), min(m, n1)
        for k1 in range(k1_min, k1_max + 1):
            k2 = m - k1
            eta_t_plus = eta_theta_plus_two_sample(x, y, theta_hat_x, theta_hat_y, k1, k2, delta, loss_type=loss_type)
            eta_s_minus = eta_sigma_minus_two_sample(x, y, theta_hat_x, theta_hat_y, sigma_hat, k1, k2, delta, loss_type=loss_type)
            level = (theta_diff + eta_t_plus) / (sigma_hat - eta_s_minus)
            level_lower_max = max(level_lower_max, abs(level))
    if bp_type == 'power':
        return original_stat, power_lower_min, power_upper_min
    else:
        return original_stat, level_lower_max, level_upper_max