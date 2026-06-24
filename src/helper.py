import numpy as np
from scipy.optimize import linprog
from tqdm import tqdm

def is_in_convex_hull_lp_batch_robust(points, vertices_batch, tol=1e-8):
    """
    points: (N, K)
    vertices_batch: (N, M, K)
    Returns: (N,) boolean array
    """
    N, K = points.shape
    M = vertices_batch.shape[1]
    results = np.zeros(N, dtype=bool)

    for i in tqdm(range(N)):
        p = points[i]
        V = vertices_batch[i]

        # 1. Drop the last dimension to ensure the system is full-rank 
        # (since sum(p) = 1 and sum(V) = 1 creates redundancy)
        A_eq = np.vstack([V[:, :-1].T, np.ones((1, M))])
        b_eq = np.concatenate([p[:-1], [1]])
        
        # 2. Feasibility check
        c = np.zeros(M)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method='highs')
        
        if res.success:
            # 3. Verify the dropped dimension was consistent
            w = res.x
            p_check = w @ V
            if np.allclose(p_check, p, atol=tol):
                results[i] = True
            else:
                results[i] = False
        else:
            results[i] = False
            
    return results
def one_hot(y, K):
    return np.eye(K)[y]


def compute_quantile(scores, alpha):
    """compute quantile from the scores

    Args:
        scores (list or np.array): scores of calibration data
        alpha (float): error rate in conformal prediction
    """
    n = len(scores)

    return np.quantile(scores, np.ceil((n+1)*(1-alpha))/n, method="inverted_cdf")

def get_tv_elementary_extreme_points_batch(p, d):
    p = np.asarray(p, dtype=float)
    n_instances, K = p.shape
    
    # Generate all i, j pairs where i != j
    ii, jj = np.where(~np.eye(K, dtype=bool))
    num_pairs = K * (K - 1)
    
    # Broadcast p to (n_instances, num_pairs, K)
    q = np.repeat(p[:, np.newaxis, :], num_pairs, axis=1)
    
    # Apply +d to index i and -d to index j for every instance
    inst_idx = np.arange(n_instances)[:, np.newaxis]
    pair_idx = np.arange(num_pairs)
    
    q[inst_idx, pair_idx, ii] += d
    q[inst_idx, pair_idx, jj] -= d
    
    return q

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
    shift = lambda_max * d

    q = p.copy()
    q[:, a] = p[:, a] + shift
    q[:, b] = p[:, b] - shift

    # numerical safeguard
    q = np.clip(q, 0.0, 1.0)
    q /= q.sum(axis=1, keepdims=True)

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



def get_last_point_on_simplex(p, q):
    """
    Finds the last point on the ray starting at p and passing through q
    (p + alpha * (q - p)) that remains within the probability simplex.
    
    Args:
        p: Categorical distribution (array-like, sums to 1, non-negative).
        q: Point in R^K (array-like, sums to 1).
        
    Returns:
        The boundary point on the simplex.
    """
    p = np.asanyarray(p)
    q = np.asanyarray(q)
    
    # Direction vector
    d = q - p
    
    # We need p + alpha * d >= 0 for all components.
    # For d_i >= 0, this is always true for alpha >= 0.
    # For d_i < 0, we need alpha <= -p_i / d_i.
    
    # Mask for components that are decreasing
    mask = d < 0
    
    if not np.any(mask):
        # If no components are decreasing, and since sum(d) = 0,
        # then d must be the zero vector (p == q).
        return p
    
    # Calculate the maximum alpha for each decreasing component
    alphas = -p[mask] / d[mask]
    
    # The first boundary hit is at the minimum of these alphas
    alpha_max = np.min(alphas)
    
    return p + alpha_max * d