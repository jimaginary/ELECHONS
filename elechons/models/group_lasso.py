import numpy as np
import tqdm
from group_lasso import GroupLasso
from elechons.models.linear_regression import Prediction, VAR
from tqdm import tqdm

GroupLasso.LOG_LOSSES = True

# input X data, column=time, row=station, n timesteps to consider
# output prediction, vector of coefficients
def gLasso(X, p, l1=0.08, l2=0.04, tol=1e-3):
    groups = np.array(p*list(range(X.shape[0])))
    # columns are first by old-new, then by station
    data = np.vstack([X[:,p-i-1:-i-1] if i != 0 else X[:,p-1:-1] for i in range(p)]).T
    Y = X[:, p:].T

    gl = GroupLasso(
        groups=groups,
        group_reg = l2,
        l1_reg = l1,
        frobenius_lipschitz=False,
        scale_reg="none",
        subsampling_scheme=None,
        supress_warning=True,
        n_iter=1,
        tol=tol,
        warm_start=True,
        fit_intercept=False
    )

    gl.fit(data, Y)
    gl.coef_ = VAR(X, p).param_history

    prev_loss = np.inf

    i = 0
    for _ in tqdm(range(200)):
        gl.fit(data, Y)
        print(np.std(gl.predict(data) - Y))
        print((gl.coef_ != 0).sum())

        if hasattr(gl, "losses_") and len(gl.losses_) > 0:
            current_loss = gl.losses_[-1]

            # Check relative improvement
            if abs(prev_loss - current_loss) < tol * (1 + abs(prev_loss)):
                print(f"Converged at iteration {i}, loss={current_loss:.6f}")
                break

            prev_loss = current_loss
        i += 1

    return Prediction(X, (lambda x : (gl.predict(data).T, gl.coef_)), (gl.coef_ != 0).sum(), delay=p)