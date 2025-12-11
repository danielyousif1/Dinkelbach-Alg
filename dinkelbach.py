import numpy as np
import cvxpy as cp

np.random.seed(33) 

def generate_random_pd_matrix(n, scale = 1.0):
    """Generates a random nxn symmetric positive definite matrix."""
    A = np.random.randn(n, n)
    # Multiply by A.T to make it symmetric PSD, add scaled identity to make it PD
    P = A @ A.T + np.eye(n) * 0.1
    return P * scale

def solve_fp_dinkelbach(n=3, max_iters=20, tol=1e-6):
    print(f"--- Solving Fractional Problem (n={n}) using Dinkelbach's Algorithm ---")
    
    # C,G in S^n_++
    C = generate_random_pd_matrix(n, scale = 1.0)
    
    # scale G (excesively) to have large eigenvalues
    G = generate_random_pd_matrix(n, scale = 100.0)

    log_det_G = np.linalg.slogdet(G)[1]
    
    # Initialization
    X0 = np.eye(n) / n
    f_val_0 = np.trace(C @ X0)
    g_val_0 = log_det_G + np.linalg.slogdet(X0)[1] # log(det(AB)) = log(det(A)det(B)) = log(det(A)) + log(det(B))
    
    lam = 1
    print(f"Initialization: lambda_0 = {lam:.4f}")
 
    # Iterate
    for k in range(max_iters):
        X = cp.Variable((n, n), symmetric = True)
        
        term_f = cp.trace(C @ X)
        
        term_g = log_det_G + cp.log_det(X)
        
        # Subproblem Objective: Minimize f(x) - lambda * g(x)
        objective = cp.Minimize(term_f - lam * term_g)
        
        constraints = [
            cp.trace(X) == 1,
            X >> 0  # PSD constraint
        ]
        
        prob = cp.Problem(objective, constraints)
        
        # Solve with built-in SCS engine
        try:
            result = prob.solve(solver = cp.SCS, verbose = False)
        except cp.error.SolverError:
            print("Solver failed. Trying default solver...")
            result = prob.solve()

        x_k = X.value
        
        f_xk = np.trace(C @ x_k)
        g_xk = log_det_G + np.linalg.slogdet(x_k)[1]
        
        # Objective value of subproblem
        F_lambda = f_xk - lam * g_xk
        
        print(f"Iter {k+1}: lambda={lam:.6f}, F(lambda)={F_lambda:.2e}, g(x)={g_xk:.4f}")
        
        # Stopping Condition
        if abs(F_lambda) < tol:
            print("\nConverged!")
            print(f"Optimal Value (min h(x)): {lam:.6f}")
            print(f"Trace(X): {np.trace(x_k):.4f}")
            return x_k, lam

        # Update lambda
        lam = f_xk / g_xk
        
    print("Max iterations reached.")
    return x_k, lam

if __name__ == "__main__":
    solve_fp_dinkelbach(n = 5)

