from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from methods.meta_template import MetaTemplate

BayesianMAMLModelState = namedtuple("BayesianMAMLModelState", ["X", "y"])
BayesianMAMLSVGDState = namedtuple("BayesianMAMLSVGDState", ["theta"])


# from: https://github.com/cnguyen10/few_shot_meta_learning
def get_kernel(particle_tensor):
    """
    Compute the RBF kernel for the input particles
    Input: particles = tensor of shape (N, M)
    Output: kernel_matrix = tensor of shape (N, N)
    """
    num_particles = particle_tensor.size(0)

    pairwise_d_matrix = get_pairwise_distance_matrix(particle_tensor)

    median_dist = torch.median(
        pairwise_d_matrix
    )  # tf.reduce_mean(euclidean_dists) ** 2
    h = median_dist / np.log(num_particles)

    kernel_matrix = torch.exp(-pairwise_d_matrix / h)
    kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
    grad_kernel = -torch.matmul(kernel_matrix, particle_tensor)
    grad_kernel += particle_tensor * kernel_sum
    grad_kernel /= h
    return kernel_matrix, grad_kernel, h


# from: https://github.com/cnguyen10/few_shot_meta_learning
def get_pairwise_distance_matrix(particle_tensor):
    """
    Input: tensors of particles
    Output: matrix of pairwise distances
    """
    num_particles = particle_tensor.shape[0]
    euclidean_dists = torch.nn.functional.pdist(
        input=particle_tensor, p=2
    )  # shape of (N)

    # initialize matrix of pairwise distances as a N x N matrix
    pairwise_d_matrix = torch.zeros(
        (num_particles, num_particles), device=particle_tensor.device
    )

    # assign upper-triangle part
    triu_indices = torch.triu_indices(row=num_particles, col=num_particles, offset=1)
    pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

    # assign lower-triangle part
    pairwise_d_matrix = torch.transpose(pairwise_d_matrix, dim0=0, dim1=1)
    pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

    return pairwise_d_matrix


class BayesianMAML(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, num_draws, num_steps):
        super(BayesianMAML, self).__init__(model_func, n_way, n_support)

        self.D = model_func().final_feat_dim

        self.C = n_way
        self.inner_lr = 1e-2

        self.num_steps = num_steps
        self.num_draws = num_draws

        self.theta = nn.Parameter(self.init_theta(self.D, self.C, self.num_draws))

        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def init_theta(self, D, n_way, num_particles):
        ret = []

        for _ in range(num_particles):
            w = torch.empty(D, n_way)
            b = torch.empty(n_way)
            log_λ = torch.empty(1)

            torch.nn.init.xavier_normal_(w)
            torch.nn.init.zeros_(b)
            torch.nn.init.ones_(log_λ)

            ret.append(
                torch.cat([w.view(-1), b.view(-1), log_λ.view(-1)], 0).view(1, -1)
            )

        return torch.cat(ret, 0)

    def theta_forward(self, theta, X):
        W = theta[:, : -(self.n_way + 1)].view(-1, self.D, self.C)
        b = theta[:, -(self.n_way + 1) : -1]
        return X.matmul(W) + b.unsqueeze(-2)

    def fit(self, X, Y):
        return BayesianMAMLModelState(X=X, y=Y.argmax(-1))

    def predict(self, model_state, svgd_state, X_star):
        logits = self.theta_forward(svgd_state.theta, X_star)
        log_proba = F.log_softmax(logits, -1)
        return torch.logsumexp(log_proba, 0) - np.log(log_proba.size(0))

    def svgd_update(self, model_state):
        svgd_state = BayesianMAMLSVGDState(self.theta)

        for _ in range(self.num_steps):
            svgd_state = self.next_svgd_state(model_state, svgd_state, self.inner_lr)

        return svgd_state

    def theta_log_prior(self, theta_flat):
        log_λ = theta_flat[-1]

        λ = torch.exp(log_λ)
        λ_inv = torch.exp(-log_λ)

        w_prior = torch.distributions.Normal(0.0, λ_inv)

        λ_prior = torch.distributions.Gamma(1.0, 0.1)

        return w_prior.log_prob(theta_flat[:-1]).sum() + λ_prior.log_prob(λ).to(
            theta_flat.device
        )

    def next_svgd_state(self, model_state, svgd_state, inner_lr):
        grads = []

        for particle_ind in range(self.num_draws):
            particle = svgd_state.theta[particle_ind]
            logits = self.theta_forward(particle.unsqueeze(0), model_state.X)[0]

            loss = self.loss_fn(
                input=logits, target=model_state.y
            ) - self.theta_log_prior(particle)

            grads.append(
                torch.autograd.grad(outputs=loss, inputs=particle, create_graph=True)[0]
            )

        grads = torch.stack(grads)
        kernel_matrix, grad_kernel, _ = get_kernel(particle_tensor=svgd_state.theta)

        return BayesianMAMLSVGDState(
            svgd_state.theta - inner_lr * (kernel_matrix.matmul(grads) - grad_kernel)
        )

    def extract_dataset(self, X):
        # X: C x shot x D
        C = X.size(0)
        shot = X.size(1)

        Y = torch.repeat_interleave(
            torch.eye(C, device=X.device, dtype=X.dtype), shot, 0
        )
        X = X.reshape(-1, X.size(-1))

        return X, Y

    def encode(self, x, is_feature=False):
        return self.parse_feature(x, is_feature)

    def set_forward(self, X, is_feature=False, verbose=False):
        X_support, X_query = self.encode(X, is_feature=is_feature)

        X_support, Y_support = self.extract_dataset(X_support)
        X_query, Y_query = self.extract_dataset(X_query)

        model_state = self.fit(X_support, Y_support)
        svgd_state = self.svgd_update(model_state)

        return self.predict(model_state, svgd_state, X_query)

    def set_forward_loss(self, X):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()
        scores = self.set_forward(X)
        return self.loss_fn(scores, y_query) / y_query.size(0)


class ChaserBayesianMAML(BayesianMAML):
    def __init__(self, *args, **kwargs):
        super(ChaserBayesianMAML, self).__init__(*args, **kwargs)
        self.leader_inner_lr = 5e-3

    def theta_log_prior(self, theta_flat):
        λ_inv = 1.0
        w_prior = torch.distributions.Normal(0.0, λ_inv)
        return w_prior.log_prob(theta_flat[:-1]).sum().to(theta_flat.device)

    def set_forward_loss(self, X):
        X_support, X_query = self.encode(X)

        X_support, Y_support = self.extract_dataset(X_support)
        X_query, Y_query = self.extract_dataset(X_query)

        # chaser
        model_state = self.fit(X_support, Y_support)
        chaser_svgd_state = self.svgd_update(model_state)

        # leader
        X_merged = torch.cat([X_support, X_query], 0)
        Y_merged = torch.cat([Y_support, Y_query], 0)
        merged_model_state = self.fit(X_merged, Y_merged)

        leader_svgd_state = BayesianMAMLSVGDState(chaser_svgd_state.theta)
        for _ in range(self.num_steps):
            leader_svgd_state = self.next_svgd_state(
                merged_model_state, leader_svgd_state, self.leader_inner_lr
            )

        loss = torch.pow(
            chaser_svgd_state.theta - leader_svgd_state.theta.detach(), 2
        ).sum()

        return loss
