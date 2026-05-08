"""
model.py  —  do not edit
"""

import numpy as np


class SVM:

    def __init__(self, C=1.0, tol=1e-3, max_iter=200):
        self.C        = C
        self.tol      = tol
        self.max_iter = max_iter

    def fit(self, K_train, y):
        from svm import violates_kkt, compute_bias
        self.y_train = y.astype(float)
        n            = len(y)
        self.K       = K_train
        self.alphas  = np.zeros(n)
        self.b       = 0.0
        self._smo(n)
        sv            = self.alphas > 1e-5
        self.sv_y     = self.y_train[sv]
        self.sv_alpha = self.alphas[sv]
        self.sv_idx   = np.where(sv)[0]
        self.K_sv     = K_train[np.ix_(sv, sv)]
        self.b        = compute_bias(self.sv_y, self.sv_alpha,
                                     self.K_sv, self.C)
        return self

    def predict(self, K_test):
        s = self.scores(K_test)
        s[s == 0] = 1          # break ties — sign(0)=0 would give wrong dtype
        return np.sign(s).astype(int)

    def scores(self, K_test):
        from svm import decision_function
        if len(self.sv_idx) == 0:
            return np.full(K_test.shape[0], self.b)
        K_sv_te = K_test[:, self.sv_idx].T    # (n_sv, M)
        return decision_function(self.sv_y, self.sv_alpha,
                                 self.b, K_sv_te)

    def _f_train(self, i):
        return float(np.dot(self.alphas * self.y_train,
                            self.K[:, i])) + self.b

    def _smo(self, n):
        from svm import violates_kkt
        y = self.y_train
        K = self.K
        for _ in range(self.max_iter):
            changed = 0
            for i in range(n):
                Ei = self._f_train(i) - y[i]
                if not violates_kkt(y[i], self.alphas[i],
                                    Ei, self.C, self.tol):
                    continue
                j = i
                while j == i:
                    j = np.random.randint(0, n)
                Ej       = self._f_train(j) - y[j]
                ai0, aj0 = self.alphas[i], self.alphas[j]
                if y[i] == y[j]:
                    L = max(0.0, ai0 + aj0 - self.C)
                    H = min(self.C,  ai0 + aj0)
                else:
                    L = max(0.0, aj0 - ai0)
                    H = min(self.C,  self.C + aj0 - ai0)
                if L >= H:
                    continue
                eta = 2*K[i,j] - K[i,i] - K[j,j]
                if eta >= 0:
                    continue
                self.alphas[j] -= y[j] * (Ei - Ej) / eta
                self.alphas[j]  = np.clip(self.alphas[j], L, H)
                if abs(self.alphas[j] - aj0) < 1e-5:
                    continue
                self.alphas[i] += y[i]*y[j] * (aj0 - self.alphas[j])
                b1 = (self.b - Ei
                      - y[i]*(self.alphas[i]-ai0)*K[i,i]
                      - y[j]*(self.alphas[j]-aj0)*K[i,j])
                b2 = (self.b - Ej
                      - y[i]*(self.alphas[i]-ai0)*K[i,j]
                      - y[j]*(self.alphas[j]-aj0)*K[j,j])
                if   0 < self.alphas[i] < self.C: self.b = b1
                elif 0 < self.alphas[j] < self.C: self.b = b2
                else:                              self.b = (b1+b2)/2
                changed += 1
            if changed == 0:
                break