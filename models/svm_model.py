import os
import json
import joblib
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import optuna

class SVMTrainer:
    def __init__(self, n_trials=40, cv_splits=5, random_state=42, out_dir="results/svm"):
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.study = None
        self.final_model = None

    def evaluate_params(self, C, gamma, epsilon, X, y):
        """Cross-validated MAE"""
        kf = KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        maes = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            maes.append(mean_absolute_error(y_val, preds))
        return float(np.mean(maes))

    def make_objective(self, X, y):
        """Closure for Optuna objective"""
        def objective(trial):
            C = trial.suggest_float('C', 1e-3, 1e3, log=True)
            gamma_mode = trial.suggest_categorical('gamma_mode', ['scale', 'float'])
            gamma = 'scale' if gamma_mode == 'scale' else trial.suggest_float('gamma', 1e-5, 1.0, log=True)
            epsilon = trial.suggest_float('epsilon', 1e-4, 1.0, log=True)
            return self.evaluate_params(C=C, gamma=gamma, epsilon=epsilon, X=X, y=y)
        return objective

    def fit_optuna(self, X, y):
        """Run Optuna optimization"""
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction='minimize', sampler=sampler)
        self.study.optimize(self.make_objective(X, y), n_trials=self.n_trials)
        # Save best params
        best_params_path = os.path.join(self.out_dir, "best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(self.study.best_params, f, indent=2)

    def fit_final_model(self, X, y):
        """Fit final SVR with best parameters"""
        best = self.study.best_params
        gamma = 'scale' if best.get('gamma_mode', 'scale')=='scale' else best.get('gamma', 1e-3)
        self.final_model = SVR(kernel='rbf', C=best['C'], gamma=gamma, epsilon=best['epsilon'])
        self.final_model.fit(X, y)
        model_path = os.path.join(self.out_dir, "svr_final.joblib")
        joblib.dump({"model": self.final_model, "best_params": best}, model_path)
        return self.final_model, model_path
