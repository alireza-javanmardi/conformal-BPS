import numpy as np
import cvxpy as convex


# Optimization function with m constraints
def solve_b(p_set, nominal_coverage):
    """find the optimal b given the probabilities in p_set and the given nominal coverage

    Args:
        p_set (np.array): n(number of instances)*k(number of classes)*m(number of probability distributions  per instance) array of probability distributions
        nominal_coverage (float): the desired coverage

    Returns:
        np.array: n*k array of optimal Bernoulli parameters
    """
    n, m, k = p_set.shape
    b = convex.Variable((n, k))
    constraints = [b >= 0, b <= 1]
    for i in range(m):
        constraints.append(convex.sum(convex.multiply(b, p_set[:, i, :]), axis=1) >= nominal_coverage)
    problem = convex.Problem(convex.Minimize(convex.sum(b)), constraints)
    problem.solve()
    # problem.solve(solver=convex.SCS)
    return b.value

def solve_b_in_batches(p_set, nominal_coverage, batch_size=1000):
    """applying solve_b_multiple in batches
    """
    n = p_set.shape[0]
    all_b = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_batch = solve_b(p_set[start:end], nominal_coverage)
        all_b.append(b_batch)
    return np.vstack(all_b)

# Average b[true_label]
def avg_true_label_inclusion(b, labels):
    """average coverage given the expected sets and true labels

    Args:
        b (np.array): n*k array of optimal Bernoulli parameters
        labels (np.array): n array of true labels

    Returns:
        float: average of expected coverage
    """
    return np.mean([b[i, labels[i]] for i in range(len(labels))])

# Binary search for optimal nominal coverage
def find_optimal_lambda(p_set, labels, target, batch_size, tol=1e-3):
    """find the optimal lambda that satisfies the target coverage

    Args:
        p_set (np.array): n(number of instances)*k(number of classes)*m(number of probability distributions  per instance) array of probability distributions
        labels (np.array): n array of true labels
        target (float): desired coverage
        batch_size (int): batch_size for solving b
        tol (float, optional): resolution of finding optimal lambda. Defaults to 1e-3.

    Returns:
        best_lambda (float): optimal lambda that satisfies the desired target coverage
        best_b (np.array): n*k array of optimal Bernoulli parameters
    """
    lo, hi = 0, 1.0
    # lo, hi = target, 1.0
    best_lambda = None
    best_b = None
    while hi - lo > tol:
        mid = (lo + hi) / 2
        b = solve_b_in_batches(p_set, nominal_coverage=mid, batch_size=batch_size)
        avg_incl = avg_true_label_inclusion(b, labels)
        if avg_incl >= target:
            best_lambda = mid
            best_b = b
            hi = mid
        else:
            lo = mid
    return best_lambda, best_b

def boundary_point_with_upper(p, a, b, d):
    """
    p:   (n, K) array of distributions
    a,b: indices in [0..K)
    d:   scalar or (n,) array of radii

    Returns
    -------
    q : (n, K) array of boundary points
    lam : (n,) array of lambda_max for each instance
    """
    # compute the two candidate upper‐bounds on lambda
    lam1 = p[:, b] / d                # from nonnegativity at b
    lam2 = (1.0 - p[:, a]) / d        # from upper‐bound at a

    # combine them with 1.0
    lambda_max = np.minimum(np.minimum(lam1, lam2), 1.0)

    # shift p by lambda_max * h
    q = p.copy()
    q[:, a] += lambda_max * d
    q[:, b] -= lambda_max * d

    return q, lambda_max

def all_boundary_points(P, d):
    """
    P: (n, K) array of n distributions
    d: scalar or (n,) array of radii

    Returns
    -------
    Q : (n, K*(K-1), K) array of all boundary points
    L : (n, K*(K-1))    array of corresponding lambdas
    """
    n, K = P.shape
    m = K * (K - 1)

    Q = np.empty((n, m, K), dtype=P.dtype)
    L = np.empty((n, m),    dtype=P.dtype)

    idx = 0
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            q_ab, lam_ab = boundary_point_with_upper(P, a, b, d)
            Q[:, idx, :] = q_ab
            L[:, idx]      = lam_ab
            idx += 1

    return Q, L

def tv(p,q):
    """total variation distance of two discrete distribution

    Args:
        p (_type_): first distribution
        q (_type_): second distribution

    Returns:
        float: a number between 0 and 1
    """
    return 0.5*np.sum(np.abs(p-q), axis=1)
