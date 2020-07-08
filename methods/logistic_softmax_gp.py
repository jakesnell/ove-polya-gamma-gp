from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

import pypolyagamma
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D

from methods.meta_template import MetaTemplate
from methods.ove_polya_gamma_gp import load_kernel, psd_safe_cholesky, sample_mvn

LogisticSoftmaxModelState = namedtuple(
    "LogisticSoftmaxModelState",
    ["N", "C", "mu", "K_block", "K", "L", "Kinv_mu", "X", "Y", "kernel"],
)
LogisticSoftmaxGibbsState = namedtuple(
    "LogisticSoftmaxGibbsState", ["λ", "n", "ω", "f"]
)


class LogisticSoftmaxGP(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(LogisticSoftmaxGP, self).__init__(model_func, n_way, n_support)
        self.n_way = n_way
        self.kernel = load_kernel(n_way)
        self.noise = 0.0
        self.ppg = pypolyagamma.PyPolyaGamma()
        self.num_steps = 1
        self.num_draws = 1
        self.quad = GaussHermiteQuadrature1D()
        self.eps = 1e-6

        self.loss_fn = nn.CrossEntropyLoss()

    def extract_dataset(self, X):
        # X: C x shot x D
        C = X.size(0)
        shot = X.size(1)

        Y = torch.repeat_interleave(
            torch.eye(C, device=X.device, dtype=X.dtype), shot, 0
        )
        X = X.reshape(-1, X.size(-1))

        return X, Y

    def merged_encode(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature=is_feature)
        return torch.cat([z_support, z_query], 1)

    def set_forward_loss(self, X):
        X, Y = self.extract_dataset(self.merged_encode(X))
        model_state = self.fit(X, Y)

        gibbs_state = self.gibbs_sample(model_state)

        return -self.log_marginal_likelihood(model_state, gibbs_state).mean(0) / X.size(
            0
        )

    def encode(self, x, is_feature=False):
        return self.parse_feature(x, is_feature)

    def set_forward(self, X, is_feature=False, verbose=False):
        X_support, X_query = self.encode(X, is_feature=is_feature)

        X_support, Y_support = self.extract_dataset(X_support)
        X_query, Y_query = self.extract_dataset(X_query)

        model_state = self.fit(X_support, Y_support)

        gibbs_state = self.gibbs_sample(model_state)

        f_post = self.predict_posterior(model_state, gibbs_state, X_query)

        f_samples = f_post.rsample((1000,))
        f_samples = f_samples.reshape(-1, *Y_query.size())

        return F.log_softmax(F.logsigmoid(f_samples).mean(0), -1)

    def sample_polya_gamma(self, b, c):
        c_device = c.device

        b = b.detach().double().cpu()
        c = c.detach().double().cpu()

        ret = torch.zeros_like(c)

        self.ppg.pgdrawv(
            b.reshape(-1).numpy(), c.reshape(-1).numpy(), ret.view(-1).numpy()
        )
        return ret.float().to(c_device)

    def fit(self, X, Y):
        N = X.size(0)
        C = Y.size(-1)

        mu = self.kernel.mean_function(X).view(C, N)
        K_block = self.kernel.cov_function(X)
        K_block = K_block + self.noise * torch.eye(
            K_block.size(-1), dtype=K_block.dtype, device=K_block.device
        )
        K = K_block[:N, :N]

        L = psd_safe_cholesky(K)

        K = K.unsqueeze(0).expand(C, N, N)
        L = L.unsqueeze(0).expand(C, N, N)

        Kinv_mu = torch.cholesky_solve(mu.unsqueeze(-1), L).squeeze(-1)

        return LogisticSoftmaxModelState(
            N=N,
            C=C,
            mu=mu,
            K_block=K_block,
            K=K,
            L=L,
            Kinv_mu=Kinv_mu,
            X=X,
            Y=Y,
            kernel=self.kernel,
        )

    def gibbs_sample(self, model_state):
        gibbs_state = self.initial_gibbs_state(model_state)

        for _ in range(self.num_steps):
            gibbs_state = self.next_gibbs_state(model_state, gibbs_state)

        return gibbs_state

    def initial_gibbs_state(self, model_state):
        # sample from prior
        f_init = sample_mvn(model_state.mu, model_state.L, batch_size=self.num_draws)
        # Galy-Fajou uses 1 as initialization for λ
        λ_init = torch.ones(
            self.num_draws, model_state.N, device=f_init.device, dtype=f_init.dtype
        )

        # draws x C x N
        n_init = (
            torch.distributions.Poisson(λ_init).sample((model_state.C,)).transpose(0, 1)
        )

        ω_init = self.sample_polya_gamma(
            n_init.float() + model_state.Y.t().unsqueeze(0).float(),
            torch.zeros(
                self.num_draws,
                model_state.C,
                model_state.N,
                dtype=model_state.K.dtype,
                device=model_state.K.device,
            ),
        )

        return LogisticSoftmaxGibbsState(λ_init, n_init, ω_init, f_init)

    def next_gibbs_state(self, model_state, gibbs_state):
        n_new = torch.distributions.Poisson(
            gibbs_state.λ.unsqueeze(-2) * F.sigmoid(-gibbs_state.f)
        ).sample()
        λ_new = torch.distributions.Gamma(1 + n_new.sum(1), model_state.C).sample()
        ω_new = self.sample_polya_gamma(
            n_new.float() + model_state.Y.t().unsqueeze(0).float(), gibbs_state.f
        )
        f_new = self.gaussian_conditional(model_state, ω_new, n_new).sample()

        return LogisticSoftmaxGibbsState(λ_new, n_new, ω_new, f_new)

    def gaussian_conditional(self, model_state, ω, n):
        ω = ω.clamp(min=self.eps)

        kappa = 0.5 * (model_state.Y.t().unsqueeze(0) - n)  # draws x C x N
        Ω_inv = torch.diag_embed(1.0 / ω)  # draws x C x N x N

        L_noisy = psd_safe_cholesky(model_state.K + Ω_inv)
        Sigma = model_state.K - model_state.K.matmul(
            torch.cholesky_solve(model_state.K, L_noisy)
        )

        mu_tilde = Sigma.matmul(
            (model_state.Kinv_mu.unsqueeze(0) + kappa).unsqueeze(-1)
        ).squeeze(-1)

        return MultivariateNormal(mu_tilde, scale_tril=psd_safe_cholesky(Sigma))

    def predict_posterior(self, model_state, gibbs_state, X_star):
        ω = gibbs_state.ω.view(-1, model_state.C * model_state.N).clamp(min=self.eps)

        kappa = 0.5 * (model_state.Y.t().unsqueeze(0) - gibbs_state.n).reshape(
            -1, model_state.C * model_state.N
        )  # draws x CN
        Ω_inv = torch.diag_embed(1.0 / ω)  # draws x CN x CN
        L_noisy = psd_safe_cholesky(model_state.K_block + Ω_inv)
        z = kappa / ω

        mu_star = model_state.kernel.batch_mean_function(X_star)
        K_star = model_state.kernel.batch_cov_function(X_star, model_state.X)

        mu_pred = mu_star.unsqueeze(0) + K_star.matmul(
            torch.cholesky_solve((z - model_state.mu.view(-1)).unsqueeze(-1), L_noisy)
            .squeeze(-1)
            .t()
        ).permute([2, 0, 1])

        K_star_diag = model_state.kernel.batch_cov_function_diag(X_star)

        Sigma_pred = K_star_diag.unsqueeze(0) - torch.cholesky_solve(
            K_star.transpose(-1, -2).unsqueeze(0), L_noisy.unsqueeze(1)
        ).transpose(-1, -2).matmul(K_star.transpose(-1, -2))

        return MultivariateNormal(mu_pred, scale_tril=psd_safe_cholesky(Sigma_pred))

    def log_marginal_likelihood(self, model_state, gibbs_state):
        ω = gibbs_state.ω.view(-1, model_state.C * model_state.N).clamp(min=self.eps)

        M = ω.shape[-1]

        kappa = 0.5 * (model_state.Y.t().unsqueeze(0) - gibbs_state.n).reshape(
            -1, model_state.C * model_state.N
        )  # draws x CN

        z = kappa / ω

        Ω_inv = torch.diag_embed(1.0 / ω)

        z_mu = model_state.mu.view(model_state.C * model_state.N).unsqueeze(0)
        z_Sigma = model_state.K_block + Ω_inv
        p_z_marginal = MultivariateNormal(z_mu, scale_tril=psd_safe_cholesky(z_Sigma))

        return (
            p_z_marginal.log_prob(z)
            + 0.5 * M * np.log(2 * np.pi)
            - 0.5 * torch.log(ω).sum(-1)
            + 0.5 * torch.sum((kappa ** 2) / ω, -1)
            - M * np.log(2.0)
        )


class PredictiveLogisticSoftmaxGP(LogisticSoftmaxGP):
    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()
        scores = self.set_forward(x)
        return self.loss_fn(scores, y_query)
