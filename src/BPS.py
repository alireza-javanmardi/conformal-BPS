import numpy as np
import cvxpy as convex



def solve_b(credal_sets, lambda_):
    """find the optimal b given the probabilities in credal_sets and the given nominal coverage

    Args:
        credal_sets (np.array): n(number of instances)*m(number of probability distributions  per instance)*k(number of classes) array of probability distributions
        lambda_ (float): the desired coverage

    Returns:
        np.array: n*k array of optimal Bernoulli parameters
    """
    n, m, k = credal_sets.shape
    b = convex.Variable((n, k))
    constraints = [b >= 0, b <= 1]
    for i in range(m):
        constraints.append(convex.sum(convex.multiply(b, credal_sets[:, i, :]), axis=1) >= lambda_)
    problem = convex.Problem(convex.Minimize(convex.sum(b)), constraints)
    problem.solve()
    # problem.solve(solver=convex.SCS)
    return b.value

def solve_b_in_batches(credal_sets, lambda_, batch_size=1000):
    """applying solve_b_multiple in batches
    """
    n = credal_sets.shape[0]
    all_b = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_batch = solve_b(credal_sets[start:end], lambda_)
        all_b.append(b_batch)
    return np.vstack(all_b)


def lambda_optimizer(set_predictor_fn, inverse_risk_fn, target, tol=1e-5, lo=0.0, hi=1.0, set_predictor_kwargs=None, risk_kwargs=None, return_sets=False):
    """
    Generic binary-search lambda optimizer.
    """

    if set_predictor_kwargs is None:
        set_predictor_kwargs = {}

    if risk_kwargs is None:
        risk_kwargs = {}

    best_lambda = None
    best_sets = None

    while hi - lo > tol:
        mid = (lo + hi) / 2

        # --- call set predictor ---
        sets = set_predictor_fn(lambda_=mid, **set_predictor_kwargs)

        # --- evaluate risk ---
        risk_value = inverse_risk_fn(sets=sets, **risk_kwargs)

        if risk_value >= target :
            best_lambda = mid
            best_sets = sets
            hi = mid
        else:
            lo = mid

    if return_sets:
        return best_lambda, best_sets

    return best_lambda


def cond_cvg(sets, true_dists):
    return np.sum(np.multiply(sets, true_dists), axis=1)

def mean_cond_cvg(sets, true_dists):
    return np.mean(cond_cvg(sets, true_dists))

def mean_cond_cvg_satisfaction(sets, true_dists, desired_cond_cvg):
    return np.mean(cond_cvg(sets, true_dists) >= desired_cond_cvg)  

def marg_cvg(sets, labels):
    return np.mean(sets[np.arange(sets.shape[0]), labels])