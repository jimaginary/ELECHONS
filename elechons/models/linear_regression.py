import numpy as np
from typing import Callable
from sklearn.linear_model import Lasso

class Prediction:
    data: np.ndarray
    prediction: np.ndarray
    delay: int
    predictor: Callable[[np.ndarray], np.ndarray]
    param_history: np.ndarray
    num_params: int

    def __init__(self, data, predictor, num_params, delay=1):
        self.data = data
        self.delay = delay
        self.predictor = predictor
        self.num_params = num_params
        self.prediction, self.param_history = self.predictor(self.data[..., :-1])
    
    def residuals(self):
        return self.data[..., self.delay:] - self.prediction
    
    def rmse(self):
        return np.sqrt(np.mean(np.pow(self.residuals(), 2)))
    
    def rmse_percent(self):
        return 100 * self.rmse() / np.sqrt(np.mean(np.pow(self.data[..., self.delay:], 2)))
    
    def BIC(self):
        return np.prod(self.data.shape) * 2 * np.log(self.rmse()) + self.num_params * np.log(np.prod(self.data.shape))
    
    # this function only works when param_history is a VAR autoregression matrix
    prE = None
    prEl0 = None
    Ecov = None
    def CMI(self, l0=False, omega0=0, omega1=np.pi):
        E = self.residuals()
        N = E.shape[0]
        p = self.param_history.shape[1] // self.param_history.shape[0]

        if self.Ecov is None:
            self.Ecov = np.cov(E)
        Ecov = self.Ecov
        if self.prE is None:
            self.prE = np.linalg.inv(Ecov)
        prE = self.prE
        if self.prEl0 is None and l0:
            self.prEl0 = FastGL0(Ecov, self.data.shape[1])
            prE = self.prEl0
        
        Atau = np.split(self.param_history, p, axis=1)
        def R(omega):
            Aomega = np.eye(N) - sum(Atau[i]*np.exp(-1j*omega*i) for i in range(p))
            Finvomega = Aomega.conj().T @ prE @ Aomega
            return - Finvomega / np.sqrt(np.outer(Finvomega.diagonal(), Finvomega.diagonal()))

        CMI = np.zeros((N,N))
        D = self.data.shape[1]
        dw = 2 * (omega1-omega0) / D
        for omega in np.linspace(omega0,omega1, int(D * (omega1 - omega0) / np.pi)):
            CMI += - 0.5 * (1/(2*np.pi)) * np.log(1 - np.pow(np.abs(R(omega)),2)) * dw
            CMI += - 0.5 * (1/(2*np.pi)) * np.log(1 - np.pow(np.abs(R(-omega)),2)) * dw
        
        return CMI

def VAR(data, p, percentile=0):
    Y = data[:, p:]
    Z = np.vstack([data[:,p-i-1:-i-1] for i in range(p)])

    A = np.linalg.inv(Z @ Z.T) @ Z @ Y.T
    if percentile != 0:
        A = np.where(A < np.percentile(A, percentile), A, 0)

    def predictor(data):
        Z = np.vstack([data[:,p-i-1:-i] if i != 0 else data[:,p-1:] for i in range(p)])

        return ((Z.T @ A).T, A.T)
    return Prediction(data, predictor, np.sum(A != 0), delay=p)

def FastGL0(S_, n):
    l = np.log(n)/(2*n)
    O_ = np.diag(1 / np.diag(S_))
    G_ = np.diag(np.diag(S_))
    N = S_.shape[0]

    def p_(A,i):
        A[[-1,i],:]=A[[i,-1],:]
        A[:,[-1,i]]=A[:,[i,-1]]

    def it(S,O,G):
        for i in range(N):
            # permute the matrices
            p_(S,i)
            p_(O,i)
            p_(G,i)

            O_not_i_inv = G[:-1,:-1] - np.outer(G[:-1,-1],G[:-1,-1])/G[-1,-1]
            for j in range(N-1):
                dot_ = np.dot(O_not_i_inv[:,j],O[:-1,-1])-O_not_i_inv[j,j]*O[j,-1]
                Bij_nz_opt = -(S[-1,j]+S[-1,-1]*dot_)/(S[-1,-1]*O_not_i_inv[j,j])
                F_nz_opt = 2*S[-1,j]+2*S[-1,-1]*Bij_nz_opt*dot_+S[-1,-1]*np.pow(Bij_nz_opt,2)*O_not_i_inv[j,j]
                if F_nz_opt > l:
                    O[-1,j] = Bij_nz_opt
                    O[j,-1] = Bij_nz_opt
                else:
                    O[-1,j] = 0
                    O[j,-1] = 0
            O[-1,-1] = O[-1,:-1] @ O_not_i_inv @ O[:-1,-1] + (1/S[-1,-1])

            prod_ = O_not_i_inv @ O[:-1,-1]
            s_ = O[-1,-1] - O[-1,:-1] @ O_not_i_inv @ O[:-1,-1]
            G[:-1,:-1] = O_not_i_inv + np.outer(prod_,prod_)/s_
            G[:-1,-1] = - prod_ / s_
            G[-1,:-1] = - prod_.T / s_
            G[-1,-1] = 1/s_

            # permute back
            p_(S,i)
            p_(O,i)
            p_(G,i)
        return O, G
    
    for i in range(200):
        O_, G_ = it(S_, O_, G_)
    return O_

def CMI(data, p):
    # Compute CMI matrix with entries CMI[i,j] = I(x_i; z_j | z_-ij)
    Y = data[:, p:]
    Z = np.vstack([data[:,p-i-1:-i-1] for i in range(p)])

    A = (np.linalg.inv(Z @ Z.T) @ Z @ Y.T).T
    N = A.shape[0]
    Atau = np.split(A, p, axis=1)

    sZ = np.cov(Z)
    E = Y - A @ Z
    sE = np.cov(E)
    prE = np.linalg.inv(sE)
    def R(omega):
        Aomega = np.eye(N) - sum(Atau[i]*np.exp(-1j*omega*i) for i in range(p))
        Finvomega = Aomega.conj().T @ prE @ Aomega
        return - Finvomega / np.sqrt(np.outer(Finvomega.diagonal(), Finvomega.diagonal()))
    
    CMI = np.zeros((N,N))
    D = 300
    dw = 2 * np.pi / D
    for omega in np.linspace(-np.pi,np.pi,D):
        CMI += - 0.5 * (1/(2*np.pi)) * np.log(1 - np.pow(np.abs(R(omega)),2)) * dw
    
    return CMI

    

def VAR_lasso(data, p, alpha):
    Y = data[:, p:]
    Z = np.vstack([data[:,p-i-1:-i-1] for i in range(p)])

    # A = np.linalg.inv(Z @ Z.T) @ Z @ Y.T

    A = np.zeros((data.shape[0] * p, data.shape[0]))
    
    for i in range(data.shape[0]):
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
        lasso.fit(Z.T, Y[i])
        A[:, i] = lasso.coef_

    def predictor(data):
        Z = np.vstack([data[:,p-i-1:-i] if i != 0 else data[:,p-1:] for i in range(p)])

        return ((Z.T @ A).T, A)
    return Prediction(data, predictor, np.sum(A != 0), delay=p)

def VAR_l0_coord_descent(data, p, init, max_steps=1000, threshhold=10, alpha=0.01):
    Y = data[:, p:]
    Z = np.vstack([data[:,p-i-1:-i-1] for i in range(p)])

    A = init.copy()

    obs = np.prod(Y.shape)
    def BICa(B):
        return obs * np.log(np.mean(np.pow(B @ Z - Y, 2))) + np.sum(B != 0) * np.log(obs)
    def l0(B):
        return np.mean(np.pow(B @ Z - Y, 2)) + np.sum(B != 0) * alpha

    sigma_ZZ = (Z @ Z.T) / Y.shape[1]
    sigma_YZ = (Y @ Z.T) / Y.shape[1]
    def coord_optimise(i, j):
        AiSZZj = np.dot(A[i], sigma_ZZ[j]) - A[i,j]*sigma_ZZ[j,j]
        Aij_opt = (sigma_YZ[i,j] - AiSZZj) / sigma_ZZ[j,j]
        err_with = Aij_opt**2 * sigma_ZZ[j,j] + 2 * Aij_opt * AiSZZj - 2 * Aij_opt * sigma_YZ[i,j] + alpha
        err_without = 0
        if err_with < err_without:
            A[i,j] = Aij_opt
        else:
            A[i,j] = 0

    def optimize_array():
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                coord_optimise(i, j)
    
    bic = BICa(A)
    for i in range(max_steps):
        # print()
        # print(f'step {i}')
        old_bic = bic
        optimize_array()
        bic = BICa(A)
        # print(bic)
        # print(f'l0 norm {l0(A)}')
        # print(np.sum(A != 0))
        if np.abs(old_bic - bic) < threshhold:
            break
    
    print(f'finished after {i} steps')

    def predictor(data):
        Z = np.vstack([data[:,p-i-1:-i] if i != 0 else data[:,p-1:] for i in range(p)])

        return ((A @ Z), A)
    return Prediction(data, predictor, np.sum(A != 0), delay=p)

def VAR_group_l0(data, p, init, max_steps=1000, threshhold=10, alpha=0.01):
    Y = data[:, p:]
    Z = np.empty((p*data.shape[0], data.shape[1] - p), dtype=data.dtype)
    for i in range(p):
        Z[i::p] = data[:,p-i-1:-i-1]

    A = init.copy()

    obs = np.prod(Y.shape)
    def BICa(B):
        return obs * np.log(np.mean(np.pow(B @ Z - Y, 2))) + np.sum(B != 0) * np.log(obs)
    def l0(B):
        return np.mean(np.pow(B @ Z - Y, 2)) + np.sum(B != 0) * alpha
    def m(M, i, j):
        return M[i*p:(i+1)*p,j*p:(j+1)*p]
    def v(M, i, j):
        return M[i,j*p:(j+1)*p]

    sigma_ZZ = (Z @ Z.T) / Y.shape[1]
    sigma_YZ = (Y @ Z.T) / Y.shape[1]
    def coord_optimise(i, j):
        sYZij = v(sigma_YZ, i, j)
        # print(f'sYZij {sYZij.shape}')
        sZZjj = m(sigma_ZZ, j, j)
        # print(f'sZZjj {sZZjj.shape}')
        AiSZZj = np.sum([m(sigma_ZZ, j, l) @ v(A, i, l) for l in range(A.shape[0]) if l != j], axis=0)
        # print(f'AiSZZj {AiSZZj.shape}')
        Aij_opt = np.linalg.inv(sZZjj) @ (sYZij - AiSZZj)
        # print(f'Aij_opt {Aij_opt.shape}')
        err_with = Aij_opt.T @ sZZjj @ Aij_opt + 2 * Aij_opt.T @ AiSZZj - 2 * sYZij.T @ Aij_opt + alpha
        err_without = 0
        if err_with < err_without:
            A[i,j*p:(j+1)*p] = Aij_opt
        else:
            A[i,j*p:(j+1)*p] = 0

    def optimize_array():
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                coord_optimise(i, j)
    
    bic = BICa(A)
    for i in range(max_steps):
        # print()
        # print(f'step {i}')
        old_bic = bic
        optimize_array()
        bic = BICa(A)
        # print(bic)
        # print(f'l0 norm {l0(A)}')
        # print(np.sum(A != 0))
        if np.abs(old_bic - bic) < threshhold:
            break
    
    print(f'finished after {i} steps')

    def predictor(data):
        Z = np.empty((p*data.shape[0], data.shape[1] - p + 1), dtype=data.dtype)
        for i in range(p):
            Z[i::p] = (data[:,p-i-1:-i] if i != 0 else data[:,p-1:])

        return ((A @ Z), A)
    return Prediction(data, predictor, np.sum(A != 0), delay=p)

def solve_VAR_with_mask(data, p, mask):
    Y = data[:, p:]  # shape: (n, T)
    Z = np.vstack([data[:, p - i - 1 : -i - 1] for i in range(p)])  # (pn, T)

    n, T = Y.shape
    pn = Z.shape[0]
    A = np.zeros((n, pn))

    for i in range(n):
        mask_i = mask[i]  # (pn,)
        if not np.any(mask_i):
            continue  # skip if entire row is masked

        Z_masked = Z[mask_i]  # select only allowed predictors
        y_i = Y[i]             # target vector

        # Least squares for this output dimension
        Ai_masked = np.linalg.lstsq(Z_masked.T, y_i, rcond=None)[0]
        A[i, mask_i] = Ai_masked  # fill only allowed entries

    def predictor(data):
        Z_pred = np.vstack([data[:, p - i - 1 : -i] if i != 0 else data[:, p - 1 :] for i in range(p)])
        return (A @ Z_pred, A)

    return Prediction(data, predictor, np.sum(A != 0), delay=p)

# rows are features, cols are time domain samples
def linear_auto_regression(data):
    precision = np.linalg.inv(data @ data.T)
    autocov = data[:, :-1] @ data[:, 1:].T

    A = precision @ autocov

    predictor = lambda data : ((data.T @ A).T, A)

    # return Prediction(data[:, 1:], (data[:, :-1].T @ precision @ autocov).T)
    return Prediction(data, predictor, np.prod(np.shape(A)))

# pred returns \hat{x}_{t+1} given x_t, params
# init_params the initial parameters
# grad returns \partial/(\partial{A_t})||x_{t+1}-\hat{x}_{t+1}||_2^2 given x_{t+1}, \hat{x}_{t+1}, and x_t
# prox optionally applies a proximal step to A_{t+1}
def exp_sgd(data, pred, init_params, grad, learning_coeff, prox=None, grad_learning_coeff=0):
    def func(data):
        param_history = np.empty((*init_params.shape, data.shape[-1]))
        param_history[..., 0] = init_params
        predictions = np.empty(data.shape)

        for i in range(data.shape[-1]):
            predictions[..., i] = pred(data[..., i], param_history[..., i])
            if i == data.shape[-1] - 1:
                break

            if grad_learning_coeff != 0 and not i == 0:
                gradient = (1-grad_learning_coeff) * gradient + grad_learning_coeff * grad(data[..., i+1], predictions[..., i], data[..., i])
            else: 
                gradient = grad(data[..., i+1], predictions[..., i], data[..., i])

            param_history[..., i+1] = param_history[..., i] - learning_coeff * gradient
            if prox is not None:
                param_history[..., i+1] = prox(param_history[..., i+1])
        
        return predictions, param_history
    return func


def linear_sgd(data, transition_learning_coefficient=0.1, div_by_zero_offset=1e-8, threshhold=0, grad_learning_coeff=0):
    pred = lambda x, params: params @ x
    init_params = np.outer(data[:, 1], data[:, 0]) / np.inner(data[:, 0], data[:, 0])
    grad = lambda x_dash, x_dash_hat, x: - np.outer(x_dash - x_dash_hat, x) / (np.sum(np.pow(x, 2)) + div_by_zero_offset)
    learning_coeff = transition_learning_coefficient
    prox = None
    if threshhold != 0:
        prox = lambda a: (np.abs(a) > threshhold) * a
        # prox = lambda a: np.maximum(np.abs(a) - lasso_coeff, 0) * np.sign(a)
    return Prediction(data, exp_sgd(data, pred, init_params, grad, learning_coeff, prox, grad_learning_coeff), 1 + (threshhold != 0) + (grad_learning_coeff != 0))

def var_sgd(data, p, learning_rate = 0.1, div_by_zero_offset=1e-8, threshhold=0, grad_learning_coeff=1):
    def func(data_in):
        param_history = np.empty((data_in.shape[0], data_in.shape[0] * p, data_in.shape[-1] - p + 1))
        param_history[..., 0] = np.zeros((data_in.shape[0], data_in.shape[0] * p))
        predictions = np.empty((data_in.shape[0], data_in.shape[-1] - p + 1))
        gradient = np.zeros((data_in.shape[0], data_in.shape[0] * p))

        for i in range(data_in.shape[-1] - p + 1):
            if i == 0:
                z = data_in[:,p-1::-1].reshape(-1,1).flatten()
            else:
                z = data_in[:,i+p-1:i-1:-1].reshape(-1,1).flatten()
            predictions[..., i] = param_history[..., i] @ z
            if i == data_in.shape[-1] - p:
                break

            error = data_in[..., i+p] - predictions[..., i]
            grad = - np.outer(error, z) / (np.sum(np.pow(z, 2)) + div_by_zero_offset)
            gradient = (1 - grad_learning_coeff) * gradient + grad_learning_coeff * grad

            param_history[..., i+1] = param_history[..., i] - learning_rate * gradient
            param_history[..., i+1] = (np.abs(param_history[..., i+1]) > threshhold) * param_history[..., i+1]
        
        return predictions, param_history
    return Prediction(data, func, 1 + (threshhold != 0) + (grad_learning_coeff != 0), delay=p)

# no history
def fast_var_sgd(data, p, learning_rate = 0.1, div_by_zero_offset=1e-8, threshhold=0, grad_learning_coeff=1):
    def func(data_in):
        params = np.zeros((data_in.shape[0], data_in.shape[0] * p))
        predictions = np.empty((data_in.shape[0], data_in.shape[-1] - p + 1))
        gradient = np.zeros((data_in.shape[0], data_in.shape[0] * p))

        for i in range(data_in.shape[-1] - p + 1):
            if i == 0:
                z = data_in[:,p-1::-1].reshape(-1,1).flatten()
            else:
                z = data_in[:,i+p-1:i-1:-1].reshape(-1,1).flatten()
            predictions[..., i] = params @ z
            if i == data_in.shape[-1] - p:
                break

            error = data_in[..., i+p] - predictions[..., i]
            grad = - np.outer(error, z) / (np.sum(np.pow(z, 2)) + div_by_zero_offset)
            gradient = (1 - grad_learning_coeff) * gradient + grad_learning_coeff * grad

            params -= learning_rate * gradient
            params = (np.abs(params) > threshhold) * params
        
        return predictions, None
    return Prediction(data, func, 1 + (threshhold != 0) + (grad_learning_coeff != 1), delay=p)
