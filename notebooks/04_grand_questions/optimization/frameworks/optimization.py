# ============================================================================
# TITAN UNIFIED OPTIMIZATION FRAMEWORK v2.0
# ============================================================================
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from scipy.optimize import minimize, differential_evolution
from scipy.stats import qmc, norm
import warnings
warnings.filterwarnings('ignore')

class UnifiedOptimizer:
    def __init__(self, objective_fn, bounds, constraints=None):
        self.objective_fn = objective_fn
        self.bounds = bounds
        self.constraints = constraints or []
        self.param_names = list(bounds.keys())
        self.history = {'params': [], 'values': [], 'method': []}

    def _dict_to_array(self, params_dict):
        return np.array([params_dict[k] for k in self.param_names])

    def _array_to_dict(self, params_array):
        return dict(zip(self.param_names, params_array))

    def _evaluate(self, params_dict, method_name):
        value = self.objective_fn(**params_dict)
        self.history['params'].append(params_dict)
        self.history['values'].append(value)
        self.history['method'].append(method_name)
        return value

    def bayesian_optimize(self, n_init=5, n_iter=20, maximize=True, verbose=True):
        if verbose:
            print("🔵 BAYESIAN OPTIMIZATION")

        X, y = [], []
        sampler = qmc.LatinHypercube(d=len(self.param_names))
        sample = sampler.random(n=n_init)

        for i, s in enumerate(sample):
            params_dict = {}
            for j, (name, (low, high)) in enumerate(self.bounds.items()):
                params_dict[name] = low + s[j] * (high - low)
            value = self._evaluate(params_dict, 'bayesian')
            X.append(self._dict_to_array(params_dict))
            y.append(value)

        best_idx = np.argmax(y) if maximize else np.argmin(y)
        best_value = y[best_idx]

        for i in range(n_iter):
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, normalize_y=True)
            gp.fit(np.array(X), np.array(y))

            # Expected Improvement
            best_acq = -np.inf
            best_candidate = None
            sampler = qmc.LatinHypercube(d=len(self.param_names))
            candidates = sampler.random(n=500)

            for candidate in candidates:
                x_dict = {}
                for j, (name, (low, high)) in enumerate(self.bounds.items()):
                    x_dict[name] = low + candidate[j] * (high - low)

                x_array = self._dict_to_array(x_dict).reshape(1, -1)
                mu, sigma = gp.predict(x_array, return_std=True)

                if maximize:
                    improvement = mu - best_value
                    Z = improvement / (sigma + 1e-9)
                else:
                    improvement = best_value - mu
                    Z = improvement / (sigma + 1e-9)

                ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

                if ei > best_acq:
                    best_acq = ei
                    best_candidate = x_dict

            value = self._evaluate(best_candidate, 'bayesian')
            X.append(self._dict_to_array(best_candidate))
            y.append(value)

            if (maximize and value > best_value) or (not maximize and value < best_value):
                best_value = value
                if verbose: print(f" BO Iter {i+1}: {value:.6f} ⭐")

        best_idx = np.argmax(y) if maximize else np.argmin(y)
        return self._array_to_dict(X[best_idx]), y[best_idx]

    def optimize_ensemble(self, methods=['bayesian', 'gradient', 'evolutionary'], verbose=True):
        print("🎯 ENSEMBLE OPTIMIZATION")
        results = {}

        if 'bayesian' in methods:
            params, value = self.bayesian_optimize(n_init=5, n_iter=15, maximize=False, verbose=verbose)
            results['bayesian'] = (params, value)

        best_method = min(results.items(), key=lambda x: x[1][1])
        return best_method[1][0], best_method[1][1], results
