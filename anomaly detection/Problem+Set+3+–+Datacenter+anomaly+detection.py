# coding: utf-8

# # Datacenter anomaly detection


# In[3]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[4]:

from matplotlib.patches import Ellipse

# ## Initialization

# Log file with VM usage statistics.

# In[6]:

input_filename = "system-load.csv"

# Number of Gaussians.

# In[7]:

num_gaussians = 3

# Convergence threshold.

# In[8]:

convergence_eps = 1e-2

# Random seed.

# In[9]:

random_seed = 42

# ## Loading the data

# In[10]:

df_load = pd.read_csv(input_filename)

# In[11]:

X = df_load.values

# Normalizing the data to the range [0..1].

# In[13]:

X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# In[14]:

m, n = X.shape

# In[15]:

# plt.figure(figsize=(6, 6))
# plt.plot(X.T[0], X.T[1], "o", c="#144C59", alpha=0.5, markersize=3)
#
# plt.xlabel("CPU usage")
# plt.ylabel("RAM usage")
# plt.show()


# ## Training misture of Gaussians

# In[16]:

np.random.seed(random_seed)

# Initializing Gaussians at random data points.

# In[17]:

means = X[np.random.choice(m, size=num_gaussians, replace=False)]

# In[19]:

covariances = np.ones((num_gaussians, n, n)) * np.identity(n)

# Initializing source probability for every Gaussian.


phis = np.ones(num_gaussians) / num_gaussians

# Initializing correspondence probability matrix for every point.

# In[21]:

W = np.zeros((num_gaussians, m))


# Implement multivariate Gaussian PDF.

# In[22]:

def multivariate_gaussian_pdf(X, mu, sigma):
    # =============== TODO: Your code here ===============
    # Compute the probability density function for the multivariate normal distribution.
    # NOTE: this function should also work when X is a 2D matrix (m x n).
    from scipy.stats import multivariate_normal
    m,n = X.shape
    pdf = np.zeros(m)

    det = np.linalg.det(sigma)
    two_pi_pow = np.power(2*np.pi,n)
    inv_sigma = np.linalg.inv(sigma)
    X_minus_mu = (X - mu).transpose() # 2x936
    const = 1.0/(np.sqrt(two_pi_pow*det))
    for i in range(m):
        pdf[i] = const*np.exp(-1/2*((np.transpose(X_minus_mu[:,i])).dot(inv_sigma).dot(X_minus_mu[:,i])))
    return pdf #multivariate_normal(mu, sigma).pdf(X)
    # ====================================================


# Implement E step of EM algorithm.

# In[23]:

def e_step(X, means, covariances, phis):
    num_gaussians = len(means)
    W = np.zeros((num_gaussians, len(X)))

    # =============== TODO: Your code here ===============
    # Compute W for each gaussian.
    # phis[j] is the probability of z to be of Gaussian distribution j

    for j in range(num_gaussians):
        W[j, :] = multivariate_gaussian_pdf(X, means[j], covariances[j]) * phis[j]

    W /= W.sum(axis=0)
    # ====================================================

    return W


# Implement M step of EM algorithm.

# In[24]:

def m_step(X, W):
    num_gaussians = len(W)
    m, n = X.shape

    phis = np.zeros(num_gaussians)
    means = np.zeros((num_gaussians, n))
    covariances = np.zeros((num_gaussians, n, n))

    # =============== TODO: Your code here ===============
    # Compute phi, mean, and covariance for each gaussian.
    sum_W = W.sum(axis=1)
    sum_W_X = W.dot(X)

    phis = 1 / m * sum_W
    means = np.divide(sum_W_X, sum_W.reshape((num_gaussians, 1)))

    for j in range(num_gaussians):
        x_centered = (X - means[j]).transpose()
        covariances[j] = np.array([W[j,i]*x_centered[:,i, None].dot(x_centered[:,i,None].transpose()) for i in range(m)]).sum(0)
        covariances[j] /= W.sum(axis=1)[j, None]

    # ====================================================

    return phis, means, covariances


# **Run EM until convergence**

# In[25]:

def gmm_log_likelihood(x, phis, means, covariances, num_gaussians):
    # =============== TODO: Your code here ===============
    # Compute log likelihood of the mixture of Gaussians.
    l = np.zeros(num_gaussians)
    mult_prob = np.array([multivariate_gaussian_pdf(X, means[j], covariances[j]) * phis[j, None] for j in range(num_gaussians)])
    log_sum = np.log(mult_prob.sum(0))
    l = log_sum.sum()
    # ====================================================
    return l


# In[26]:

log_likelihoods = []

# In[27]:

max_em_iterations = 200

# In[123]:

for iteration in range(max_em_iterations):
    # E-step.
    W = e_step(X, means, covariances, phis)

    # Compute log-likelihood.
    log_likelihood = gmm_log_likelihood(X, phis, means, covariances, num_gaussians)
    log_likelihoods.append(log_likelihood)
    print("Iteration: {0:3d}    Log-likelihood: {1:10.4f}".format(iteration, log_likelihood))

    # M-step.
    phis, means, covariances = m_step(X, W)
    # print(means)
    # print(covariances)
    # Check log-likelihood for convergence.
    if len(log_likelihoods) > 2 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < convergence_eps:
        print("EM has converged. Stopping early.")
        break


# In[ ]:

plt.figure(figsize=(12, 6))
plt.plot(log_likelihoods)

plt.xlabel("# Iteration")
plt.ylabel("Log-likelihood")
plt.title("Learning Progress")

plt.show()


# ## Visualizing the results

# In[ ]:

def plot_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 * nstd * np.sqrt(abs(vals))
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


# In[ ]:

def plot_gaussians(X, means, covs, ax=None):
    n = len(means)
    colors = ["#1F77B4", "#2CA02C", "#FFBB78", "#4C529B"]

    plt.cla()
    plt.plot(X.T[0], X.T[1], "o", c="#144C59", alpha=0.5, markersize=3)

    for k in range(num_gaussians):
        plot_ellipse(means[k], covs[k], ax=ax, alpha=0.4, color=colors[k % len(colors)])


# In[ ]:

plt.figure(figsize=(8, 8))
plot_gaussians(X, means, covariances)
plt.xlabel("CPU usage")
plt.ylabel("RAM usage")
plt.show()
