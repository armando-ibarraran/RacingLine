# ====================================================================================================
# Optimization functions
# ====================================================================================================
from scipy.linalg import solve, norm
import numpy as np

def apGrad(f, x):
    """
    Aproximates gradient of function f at point x
    """
    n = len(x)
    grad = np.zeros(n)
    eps = np.finfo(float).eps
    for i in range(n):
        h = np.zeros(n)
        h[i]= eps**(1.0/3)*(abs(x[i])+1)
        grad[i] = 0.5*(f(x+h)-f(x-h))/h[i]
        
    return grad

def rcBFGSLiMem(f, x0, itmax, m, tol=1e-5, eta=0.1, Delta_max=100.0):
    """
    L-BFGS trust region method.
    Stores only m latest (s,y) pairs, updates only if y^T s > 1e-3.
    """
    n = len(x0)
    x = x0.copy()
    Delta = 1.0
    its = 0
    S = np.zeros((n, m))  # Storage for s vectors
    Y = np.zeros((n, m))   # Storage for y vectors
    rho = np.zeros(m)
    m_current = 0          # Current number of stored pairs
    f_best = f(x)           # control por si se escapa de un valor bueno y la ultima iteracion es mala
    x_best = x.copy()       # lo mismo por delante
    def Hk_matvec(v):
        """Apply L-BFGS inverse Hessian approximation to vector v."""
        q = v.copy()
        alphas = np.zeros(m_current)
        
        # Two-loop recursion
        for i in reversed(range(m_current)):
            alphas[i] = rho[i] * np.dot(S[:,i], q)
            q -= alphas[i] * Y[:,i]
            
        if m_current > 0 and np.dot(Y[:,-1], Y[:,-1]) > 1e-12:  # Evita división por cero
            gamma = np.dot(S[:,-1], Y[:,-1]) / np.dot(Y[:,-1], Y[:,-1])
        else:
            gamma = 1.0  # Valor por defecto
        for i in range(m_current):
            beta = rho[i] * np.dot(Y[:,i], q)
            q += (alphas[i] - beta) * S[:,i]
            
        return q
    
    for k in range(itmax):
        g = apGrad(f, x)
        
        if norm(g, np.inf) <= tol:
            break
            
        # Compute direction using Hk_matvec
        p = -Hk_matvec(g)
        p_norm = norm(p)
        if p_norm > Delta:
            p = (Delta/p_norm) * p
            
        # Compute reduction ratios
        s = p
        x_new = x + s
        
        actual_reduction = f(x) - f(x_new)
        predicted_reduction = -np.dot(g, s)
        #predicted_reduction = -np.dot(g, s) - 0.5 * np.dot(s, -Hk_matvec(g))  # Using Hk as Hessian approx
        print(k,f(x_new))
        rho_k = actual_reduction / predicted_reduction if predicted_reduction != 0 else 0
        #print(f"Iter {k}: f = {f(x_new)}, rho = {rho_k}, ||p|| = {np.linalg.norm(p)}")
        if rho_k > eta:
            y = apGrad(f, x_new) - g
            if np.dot(y, s) > 1e-2:  # Only update if curvature condition holds
                if m_current < m:
                    S[:,m_current] = s
                    Y[:,m_current] = y
                    rho[m_current] = 1.0/np.dot(y, s)
                    m_current += 1
                else:
                    S = np.roll(S, -1, axis=1)
                    Y = np.roll(Y, -1, axis=1)
                    rho = np.roll(rho, -1)
                    S[:,-1] = s
                    Y[:,-1] = y
                    rho[-1] = 1.0/np.dot(y, s)
            x = x_new
            its += 1
            
        if rho_k > 0.75 and norm(s) > 0.8*Delta:
            Delta = min(2*Delta, Delta_max)
        elif rho_k < 0.1:
            Delta *= 0.5
        if f(x_new)<f_best:
            x_best=x_new
            f_best=f(x_new)
    return x_best, its