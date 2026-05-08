import numpy as np

EPS = 1e-12

# ======================================================
# METRICS
# ======================================================

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred, cls):
    tp = np.sum((y_pred == cls) & (y_true == cls))
    fp = np.sum((y_pred == cls) & (y_true != cls))
    return tp / (tp + fp + EPS)

def recall(y_true, y_pred, cls):
    tp = np.sum((y_pred == cls) & (y_true == cls))
    fn = np.sum((y_pred != cls) & (y_true == cls))
    return tp / (tp + fn + EPS)

def f1_score(y_true, y_pred, cls):
    p = precision(y_true, y_pred, cls)
    r = recall(y_true, y_pred, cls)
    return 2*p*r/(p+r+EPS)


# ======================================================
# UTILITIES
# ======================================================

def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1/(1+np.exp(-z))

def softmax(scores):
    scores -= scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


# ======================================================
# SAMPLERS
# ======================================================

class Sampler:
    def sample(self, n): raise NotImplementedError
    def prob(self, x): raise NotImplementedError


class GaussianSampler(Sampler):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.inv = np.linalg.inv(cov)
        self.det = np.linalg.det(cov)
        self.norm = 1/np.sqrt(((2*np.pi)**len(mean))*self.det)

    def sample(self, n):
        return np.random.multivariate_normal(self.mean, self.cov, n)

    def prob(self, x):
        diff = x - self.mean
        expo = np.sum(diff @ self.inv * diff, axis=1)
        return self.norm*np.exp(-0.5*expo)


class CategoricalSampler(Sampler):
    def __init__(self, probs):
        self.probs = probs

    def sample(self, n):
        return np.random.choice(len(self.probs), size=n, p=self.probs)

    def prob(self, y):
        return self.probs[y]


# ======================================================
# SOFTMAX (EXACT NORMALIZATION)
# ======================================================

#it is very important to choose a smaller learning rate to get stable answers, higher LRs result in unstable learning, please make this change 

class SoftmaxGenerativeClassifier:

    def __init__(self, num_classes, lr=1e-2,
                 batch_size=64, max_epochs=50):
        self.K = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.W = None
        self.b = None

        self.mu = None
        self.Sigma = None
        self.pi = None

    def score(self, X):
        return X @ self.W.T + self.b

    def gradients(self, X, y):
        B = X.shape[0]
        probs = softmax(self.score(X))
        one_hot = np.eye(self.K)[y]

        diff = probs - one_hot

        gW = diff.T @ X / B
        gb = diff.mean(axis=0)
        return gW, gb

    def fit(self, X, y):
        N, D = X.shape
        self.W = np.zeros((self.K, D))
        self.b = np.zeros(self.K)

        for _ in range(self.max_epochs):
            perm = np.random.permutation(N)
            X, y = X[perm], y[perm]

            for i in range(0, N, self.batch_size):
                xb = X[i:i+self.batch_size]
                yb = y[i:i+self.batch_size]

                gW, gb = self.gradients(xb, yb)
                self.W -= self.lr*gW
                self.b -= self.lr*gb

    def predict_proba(self, X):
        return softmax(self.score(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def recover_parameters(self):
         # ---- recover Sigma^{-1} ----
        Sigma_inv = np.linalg.lstsq(self.mu, self.W, rcond=None)[0]

        # ---- recover covariance ----
        self.Sigma = np.linalg.pinv(Sigma_inv)

        # ---- recover priors ----
        quad = np.sum(self.mu @ Sigma_inv * self.mu, axis=1)  # μᵀΣ⁻¹μ

        log_pi = self.b + 0.5 * quad

        # stabilize before exponentiating
        # log_pi -= np.max(log_pi)

        self.pi = np.exp(log_pi)
        # self.pi /= self.pi.sum()

        return {
            "mu": self.mu,
            "Sigma": self.Sigma,
            "pi": self.pi
        }
    #please note that i made a mistake here, the priors cannot be directly calculated like this, there are some terms missing which mess up the calculations, as such I will only test the covariance matrix obtained

# ======================================================
# IMPORTANCE SAMPLING
# ======================================================

class ImportanceSamplingClassifier:

    def __init__(self, num_classes, lr=1e-2,
                 batch_size=64, num_samples=20,
                 max_epochs=50, class_sampler=None):
        self.K = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.M = num_samples
        self.max_epochs = max_epochs
        self.class_sampler = class_sampler
        self.W = None
        self.b = None

    def score(self, X):
        return X @ self.W.T + self.b

    def estimate_normalizer(self, X, y_samples, q):
        scores = np.exp(self.score(X)[:, y_samples])
        return np.mean(scores / q, axis=1)

    def gradients(self, X, y):
        B, D = X.shape

        y_samples = self.class_sampler.sample(self.M)
        q = self.class_sampler.prob(y_samples)

        Z_hat = self.estimate_normalizer(X, y_samples, q)

        scores = np.exp(self.score(X))
        probs = scores / Z_hat[:, None]

        one_hot = np.eye(self.K)[y]
        diff = probs - one_hot

        gW = diff.T @ X / B
        gb = diff.mean(axis=0)
        return gW, gb

    def fit(self, X, y):
        N, D = X.shape
        self.W = np.zeros((self.K, D))
        self.b = np.zeros(self.K)

        for _ in range(self.max_epochs):
            perm = np.random.permutation(N)
            X, y = X[perm], y[perm]

            for i in range(0, N, self.batch_size):
                xb = X[i:i+self.batch_size]
                yb = y[i:i+self.batch_size]

                gW, gb = self.gradients(xb, yb)
                self.W -= self.lr*gW
                self.b -= self.lr*gb

    def predict_proba(self, X):
        scores = np.exp(self.score(X))
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ======================================================
# NOISE CONTRASTIVE ESTIMATION
# ======================================================

class NCEClassifier:

    def __init__(self, num_classes, lr=1e-2,
                 batch_size=64, noise_ratio=5,
                 max_epochs=50, x_sampler=None,
                 y_sampler=None):

        self.K = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.k = noise_ratio
        self.max_epochs = max_epochs
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler

        self.W = None
        self.b = None
        self.c = 0.0

    def f(self, X, y):
        return np.sum(self.W[y]*X, axis=1) + self.b[y]

    def sample_noise(self, n):
        return self.x_sampler.sample(n), self.y_sampler.sample(n)

    def gradients(self, X, y):
        B = X.shape[0]

        f_data = self.f(X, y)

        q_data = self.x_sampler.prob(X) * self.y_sampler.prob(y)
        s_data = sigmoid(f_data - self.c - np.log(self.k * q_data + EPS))

        data_weights = (1 - s_data)   # shape (B,)

        Xn, yn = self.sample_noise(B * self.k)

        f_noise = self.f(Xn, yn)
        q_noise = self.x_sampler.prob(Xn) * self.y_sampler.prob(yn)
        s_noise = sigmoid(-f_noise + self.c + np.log(self.k * q_noise + EPS))

        noise_weights = s_noise       # shape (B*k,)


        gW = np.zeros_like(self.W)
        gb = np.zeros_like(self.b)

        # accumulate data contributions
        np.add.at(gW, y, data_weights[:, None] * X)
        np.add.at(gb, y, data_weights)

        # accumulate noise contributions
        np.add.at(gW, yn, -noise_weights[:, None] * Xn)
        np.add.at(gb, yn, -noise_weights)

        # normalize by batch size
        gW /= B
        gb /= B

        # normalization constant gradient
        gc = -np.mean(data_weights) + np.mean(noise_weights)

        return gW, gb, gc


    def fit(self, X, y):
        N, D = X.shape
        self.W = np.zeros((self.K, D))
        self.b = np.zeros(self.K)

        for _ in range(self.max_epochs):
            perm = np.random.permutation(N)
            X, y = X[perm], y[perm]

            for i in range(0, N, self.batch_size):
                xb = X[i:i+self.batch_size]
                yb = y[i:i+self.batch_size]

                gW, gb, gc = self.gradients(xb, yb)
                self.W += self.lr*gW
                self.b += self.lr*gb
                self.c += self.lr*gc

    def predict_proba(self, X):
        scores = np.exp(X @ self.W.T + self.b)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
