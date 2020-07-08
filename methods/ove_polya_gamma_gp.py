import sys
import copy
from collections import namedtuple
import warnings

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal

# from gpytorch.utils.cholesky import psd_safe_cholesky

from methods.meta_template import MetaTemplate

import pypolyagamma

from gpytorch.utils.quadrature import GaussHermiteQuadrature1D

from sacred import Ingredient

kernel_ingredient = Ingredient("kernel")


@kernel_ingredient.config
def get_config():
    name = "LinearKernel"
    learn_params = True


@kernel_ingredient.capture
def load_kernel(num_classes, name, learn_params):
    return getattr(sys.modules[__name__], name)(
        num_classes=num_classes, learn_params=learn_params
    )


# modified from:
# https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/utils/cholesky.py
def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        L = torch.cholesky(A, upper=upper, out=out)
        return L
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise NanError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(5):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(
                    f"A not p.d., added jitter of {jitter_new} to the diagonal",
                    RuntimeWarning,
                )
                return L
            except RuntimeError:
                continue
        raise e


class Kernel(nn.Module):
    def __init__(self, num_classes, learn_params):
        self.num_classes = num_classes
        self.learn_params = learn_params
        super(Kernel, self).__init__()

    def mean_function(self, X):
        raise NotImplementedError("mean_function not yet implemented")

    def cov_block(self, x1, x2):
        raise NotImplementedError("cov_block not yet implemented")

    def cov_block_diag(self, x1, x2):
        # naive implementation
        return torch.diag(self.cov_block_wrapper(x1, x2))

    def cov_block_wrapper(self, x1, x2=None):
        # x1: N x D
        # x2: N x D (or None)
        if x2 is None:
            x2 = x1
        return self.cov_block(x1, x2)

    def batch_mean_function(self, X):
        # X: N x D
        return self.mean_function(X).reshape(X.size(0), self.num_classes)

    def cov_function(self, x1, x2=None):
        return block_matrix(self.cov_block_wrapper(x1, x2), self.num_classes)

    def batch_cov_function(self, x1, x2=None):
        return batch_block_matrix(self.cov_block_wrapper(x1, x2), self.num_classes)

    def batch_cov_function_diag(self, x1, x2=None):
        ret = self.cov_block_diag(x1, x2).unsqueeze(-1)
        ret = ret.expand(ret.size(0), self.num_classes)
        return torch.diag_embed(ret)


class LinearKernel(Kernel):
    def __init__(self, *args, **kwargs):
        super(LinearKernel, self).__init__(*args, **kwargs)
        if self.learn_params:
            self.register_parameter("output_scale_raw", nn.Parameter(torch.zeros(1)))
        else:
            self.register_buffer("output_scale_raw", torch.zeros(1))

    def mean_function(self, X):
        return torch.zeros(X.size(0) * self.num_classes, dtype=X.dtype, device=X.device)

    def normalize(self, X):
        D = X.size(-1)
        return X / math.sqrt(D)

    def cov_block(self, x1, x2=None):
        x1 = self.normalize(x1)
        x2 = self.normalize(x2)
        return torch.exp(self.output_scale_raw) * (x1.mm(x2.t()))


class ConstantMeanLinearKernel(LinearKernel):
    def __init__(self, *args, **kwargs):
        super(ConstantMeanLinearKernel, self).__init__(*args, **kwargs)
        assert self.learn_params == True, "if not learning, just use LinearKernel"
        self.register_parameter("mean_value", nn.Parameter(torch.zeros(1)))

    def mean_function(self, X):
        return self.mean_value * torch.ones(
            X.size(0) * self.num_classes, dtype=X.dtype, device=X.device
        )


class L2LinearKernel(LinearKernel):
    def normalize(self, X):
        return F.normalize(X)


class ConstantMeanL2LinearKernel(ConstantMeanLinearKernel):
    def normalize(self, X):
        return F.normalize(X)


class QuadraticKernel(Kernel):
    def __init__(self, *args, **kwargs):
        super(QuadraticKernel, self).__init__(*args, **kwargs)
        if self.learn_params:
            self.register_parameter("output_scale_raw", nn.Parameter(torch.zeros(1)))
        else:
            self.register_buffer("output_scale_raw", torch.zeros(1))

    def mean_function(self, X):
        return torch.zeros(X.size(0) * self.num_classes, dtype=X.dtype, device=X.device)

    def normalize(self, X):
        D = X.size(-1)
        return X / math.sqrt(D)

    def cov_block(self, x1, x2=None):
        x1 = self.normalize(x1)
        x2 = self.normalize(x2)
        return torch.exp(self.output_scale_raw) * (x1.mm(x2.t()) ** 2)


class L2QuadraticKernel(QuadraticKernel):
    def normalize(self, X):
        return F.normalize(X)


class RBFKernel(Kernel):
    def __init__(self, *args, **kwargs):
        super(RBFKernel, self).__init__(*args, **kwargs)
        if self.learn_params:
            self.register_parameter("output_scale_raw", nn.Parameter(torch.zeros(1)))
            self.register_parameter("lengthscale_raw", nn.Parameter(torch.zeros(1)))
        else:
            self.register_buffer("output_scale_raw", torch.zeros(1))
            self.register_buffer("lengthscale_raw", torch.zeros(1))

    def mean_function(self, X):
        return torch.zeros(X.size(0) * self.num_classes, dtype=X.dtype, device=X.device)

    def normalize(self, X):
        D = X.size(-1)
        return X / math.sqrt(D)

    def cov_block(self, x1, x2=None):
        x1 = self.normalize(x1)
        x2 = self.normalize(x2)
        dists = (
            (x1 ** 2).sum(-1).view(-1, 1)
            + (x2 ** 2).sum(-1).view(1, -1)
            - 2 * x1.mm(x2.t())
        )
        return torch.exp(self.output_scale_raw) * torch.exp(
            -0.5 * dists / torch.exp(2.0 * self.lengthscale_raw)
        )


class L2RBFKernel(RBFKernel):
    def normalize(self, X):
        return F.normalize(X)


OVEGibbsState = namedtuple("OVEGibbsState", ["ω", "f"])
OVEModelState = namedtuple(
    "OVEModelState",
    ["mu", "K", "L", "Kinv_mu", "A", "AK", "AKA_T", "X", "kernel", "C", "batch_A"],
)


def sample_mvn(loc, scale_tril, batch_size=None):
    if batch_size is None:
        u = torch.randn(loc.shape, dtype=loc.dtype, device=loc.device)
        return loc + scale_tril.mv(u)
    else:
        u = torch.randn(batch_size, *loc.shape, dtype=loc.dtype, device=loc.device)
        return loc.unsqueeze(0) + scale_tril.matmul(u.unsqueeze(-1)).squeeze(-1)


def affine_matrix(y, C, dtype, device):
    N = y.size(0)

    A = torch.zeros(N * (C - 1), N * C, dtype=dtype, device=device)

    row_ind = torch.arange(N * (C - 1))
    cond = torch.ne(
        torch.arange(C, device=device).repeat(N), torch.repeat_interleave(y, C)
    )

    A[
        row_ind,
        torch.repeat_interleave(y, C - 1) * N
        + torch.repeat_interleave(torch.arange(N, device=device), C - 1),
    ] = 1.0
    A[
        row_ind,
        torch.arange(C, device=device).repeat(N)[cond] * N
        + torch.repeat_interleave(torch.arange(N, device=device), C - 1),
    ] = -1.0

    return A


def block_matrix(block, num_blocks):
    arr = []
    for i in range(num_blocks):
        row = []
        for j in range(num_blocks):
            if i == j:
                row.append(block)
            else:
                row.append(torch.zeros_like(block))
        row = torch.cat(row, 1)
        arr.append(row)
    return torch.cat(arr, 0)


def batch_block_matrix(block, num_blocks):
    # block: N x M
    # num_blocks = C
    # ret: N x C x M
    block = block.unsqueeze(1)
    arr = []
    for i in range(num_blocks):
        row = []
        for j in range(num_blocks):
            if i == j:
                row.append(block)
            else:
                row.append(torch.zeros_like(block))
        row = torch.cat(row, 2)
        arr.append(row)
    return torch.cat(arr, 1)


def batch_ove_softmax(f, A, axis):
    # f: B x N x C
    assert f.ndim == 3
    assert axis == -1

    return F.softmax(F.logsigmoid(torch.einsum("bnk,jhk->bnjh", f, A)).sum(-1), -1)


def ove_softmax(f, A, axis):
    # f: N x C
    assert f.ndim == 2
    assert axis == -1

    return F.softmax(F.logsigmoid(torch.einsum("nk,jhk->njh", f, A)).sum(-1), -1)


def gh_ove_log_softmax(dist, A, quad):
    # dist.mu: N x K
    mu = dist.loc
    Sigma = dist.covariance_matrix

    mu_out = A.matmul(mu.t()).permute(2, 0, 1)
    Sigma_out = torch.einsum(
        "jil,njil->nji", A, torch.einsum("jik,nlk->njil", A, Sigma)
    )

    log_prob_out = quad.forward(
        F.logsigmoid, torch.distributions.Normal(mu_out, Sigma_out)
    ).sum(-1)

    return F.log_softmax(log_prob_out, -1)


def gh_ove_log_prob(dist, A, quad):
    # dist.mu: D x N x K
    mu = dist.loc
    Sigma = dist.covariance_matrix

    mu_out = torch.einsum("dnl,jil->dnji", mu, A)
    Sigma_out = torch.einsum(
        "jil,dnjil->dnji", A, torch.einsum("jik,dnlk->dnjil", A, Sigma)
    )

    log_prob_out = quad.forward(
        F.logsigmoid, torch.distributions.Normal(mu_out, Sigma_out)
    ).sum(-1)

    return log_prob_out


class OVEPolyaGammaGP(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, fast_inference=False):
        super(OVEPolyaGammaGP, self).__init__(model_func, n_way, n_support)
        self.n_way = n_way
        self.kernel = load_kernel(n_way)
        self.noise = 0.0
        self.ppg = pypolyagamma.PyPolyaGamma()
        self.num_steps = 1
        self.num_draws = 1
        self.quad = GaussHermiteQuadrature1D()

        self.loss_fn = nn.CrossEntropyLoss()

        self.fast_inference = fast_inference

    def extract_dataset(self, X):
        # X: C x shot x D
        C = X.size(0)
        shot = X.size(1)

        Y = torch.repeat_interleave(torch.eye(C, device=X.device), shot, 0)
        X = X.reshape(-1, X.size(-1))

        return X, Y

    def encode(self, x, is_feature=False):
        return self.parse_feature(x, is_feature)

    def merged_encode(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature=is_feature)
        return torch.cat([z_support, z_query], 1)

    def set_forward(self, X, is_feature=False, verbose=False):
        X_support, X_query = self.encode(X, is_feature=is_feature)

        X_support, Y_support = self.extract_dataset(X_support)
        X_query, Y_query = self.extract_dataset(X_query)

        model_state = self.fit(X_support, Y_support)

        gibbs_state = self.gibbs_sample(model_state)
        return gh_ove_log_prob(
            self.predict_posterior(model_state, gibbs_state, X_query),
            model_state.batch_A,
            self.quad,
        ).mean(0)

    def fit(self, X, Y):
        C = Y.size(-1)
        A = affine_matrix(Y.argmax(-1), C, dtype=X.dtype, device=X.device)

        mu = self.kernel.mean_function(X)

        K = self.kernel.cov_function(X)
        K = K + self.noise * torch.eye(K.size(-1), dtype=K.dtype, device=K.device)

        AK = A.mm(K)
        AKA_T = A.mm(K.mm(A.t()))

        L = psd_safe_cholesky(K)
        Kinv_mu = torch.cholesky_solve(mu.unsqueeze(-1), L).squeeze(-1)

        batch_A = []
        for i in range(self.n_way):
            batch_A.append(
                affine_matrix(
                    torch.LongTensor([i]).to(X.device), self.n_way, X.dtype, X.device
                ).unsqueeze(0)
            )
        batch_A = torch.cat(batch_A, 0)

        return OVEModelState(
            mu=mu,
            K=K,
            L=L,
            Kinv_mu=Kinv_mu,
            A=A,
            AK=AK,
            AKA_T=AKA_T,
            X=X,
            kernel=self.kernel,
            C=C,
            batch_A=batch_A,
        )

    def sample_polya_gamma(self, c):
        c_device = c.device

        c = c.detach().double().cpu()
        b = torch.ones_like(c)
        ret = torch.zeros_like(c)

        self.ppg.pgdrawv(
            b.reshape(-1).numpy(), c.reshape(-1).numpy(), ret.view(-1).numpy()
        )

        return ret.float().to(c_device)

    def initial_gibbs_state(self, model_state):
        # sample from prior
        ω_init = self.sample_polya_gamma(
            torch.zeros(
                self.num_draws,
                model_state.A.shape[0],
                dtype=model_state.A.dtype,
                device=model_state.A.device,
            )
        )
        f_init = sample_mvn(model_state.mu, model_state.L, batch_size=self.num_draws)
        return OVEGibbsState(ω_init, f_init)

    def next_gibbs_state(self, model_state, gibbs_state):
        ψ_prev = gibbs_state.f.mm(model_state.A.t())
        ω_new = self.sample_polya_gamma(ψ_prev)
        f_new = self.gaussian_conditional(model_state, ω_new).sample()

        return OVEGibbsState(ω_new, f_new)

    def gaussian_conditional(self, model_state, ω):
        kappa = 0.5 * torch.ones_like(ω)
        Ω_inv = torch.diag_embed(1.0 / ω)

        L_noisy = psd_safe_cholesky(model_state.AKA_T + Ω_inv)
        Sigma = model_state.K.unsqueeze(0) - model_state.AK.t().unsqueeze(0).matmul(
            torch.cholesky_solve(model_state.AK.unsqueeze(0), L_noisy)
        )

        mu_tilde = Sigma.matmul(
            (model_state.Kinv_mu.unsqueeze(0) + kappa.matmul(model_state.A)).unsqueeze(
                -1
            )
        ).squeeze(-1)

        return MultivariateNormal(mu_tilde, scale_tril=psd_safe_cholesky(Sigma))

    def log_marginal_likelihood(self, model_state, ω):
        M = ω.shape[-1]

        kappa = 0.5 * torch.ones_like(ω)
        z = kappa / ω

        Ω_inv = torch.diag_embed(1.0 / ω)

        z_mu = model_state.A.mv(model_state.mu)
        z_Sigma = model_state.AKA_T + Ω_inv
        p_z_marginal = MultivariateNormal(z_mu, z_Sigma)

        return (
            p_z_marginal.log_prob(z)
            + 0.5 * M * np.log(2 * np.pi)
            - 0.5 * torch.log(ω).sum(-1)
            + 0.5 * torch.sum((kappa ** 2) / ω, -1)
            - M * np.log(2.0)
        )

    # use expression derived from partitioned Gaussian directly on f* and z.
    def predict_mu(self, model_state, omega, X_star):
        kappa = 0.5 * torch.ones_like(omega)
        Ω_inv = torch.diag(1.0 / omega)
        L_noisy = psd_safe_cholesky(model_state.AKA_T + Ω_inv)
        z = kappa / omega

        mu_star = model_state.kernel.batch_mean_function(X_star)
        K_star = model_state.kernel.batch_cov_function(X_star, model_state.X)

        mu_pred = mu_star + K_star.matmul(model_state.A.T).matmul(
            torch.cholesky_solve(
                (z - model_state.A.mv(model_state.mu)).unsqueeze(-1), L_noisy
            ).squeeze(-1)
        )

        return mu_pred

    def predict_batch_mu(self, model_state, batch_omega, X_star):
        kappa = 0.5 * torch.ones_like(batch_omega)
        Ω_inv = torch.diag_embed(1.0 / batch_omega)
        L_noisy = psd_safe_cholesky(model_state.AKA_T + Ω_inv)
        z = kappa / batch_omega

        mu_star = model_state.kernel.batch_mean_function(X_star)
        K_star = model_state.kernel.batch_cov_function(X_star, model_state.X)

        mu_pred = mu_star.unsqueeze(0) + K_star.matmul(model_state.A.T).matmul(
            torch.cholesky_solve(
                (z - model_state.A.mv(model_state.mu)).unsqueeze(-1), L_noisy
            )
            .squeeze(-1)
            .T
        ).permute([2, 0, 1])

        return mu_pred

    # use expression derived from partitioned Gaussian directly on f* and z.
    def predict_posterior(self, model_state, gibbs_state, X_star):
        omega = gibbs_state.ω
        kappa = 0.5 * torch.ones_like(omega)
        Ω_inv = torch.diag_embed(1.0 / omega)
        L_noisy = psd_safe_cholesky(model_state.AKA_T + Ω_inv)
        z = kappa / omega

        mu_star = model_state.kernel.batch_mean_function(X_star)
        K_star = model_state.kernel.batch_cov_function(X_star, model_state.X)

        mu_pred = mu_star.unsqueeze(0) + K_star.matmul(
            torch.cholesky_solve(
                (z - model_state.A.mv(model_state.mu)).unsqueeze(-1), L_noisy
            )
            .squeeze(-1)
            .matmul(model_state.A)
            .t()
        ).permute([2, 0, 1])

        K_star_diag = model_state.kernel.batch_cov_function_diag(X_star)
        return MultivariateNormal(
            mu_pred,
            scale_tril=psd_safe_cholesky(
                K_star_diag.unsqueeze(0)
                - torch.cholesky_solve(
                    K_star.matmul(model_state.A.T).transpose(-1, -2).unsqueeze(0),
                    L_noisy.unsqueeze(1),
                )
                .transpose(-1, -2)
                .matmul(model_state.A)
                .matmul(K_star.transpose(-1, -2))
            ),
        )

    def gibbs_sample(self, model_state):
        gibbs_state = self.initial_gibbs_state(model_state)

        for _ in range(self.num_steps):
            gibbs_state = self.next_gibbs_state(model_state, gibbs_state)

        return gibbs_state

    def gibbs_sample_all(self, model_state):
        gibbs_states = [self.initial_gibbs_state(model_state)]

        for _ in range(self.num_steps):
            gibbs_states.append(self.next_gibbs_state(model_state, gibbs_states[-1]))

        # skip initial state
        gibbs_states = gibbs_states[1:]

        return OVEGibbsState(
            torch.cat([state.ω.unsqueeze(0) for state in gibbs_states], 0),
            torch.cat([state.f.unsqueeze(0) for state in gibbs_states], 0),
        )

    def set_forward_loss(self, X):
        X, Y = self.extract_dataset(self.merged_encode(X))
        model_state = self.fit(X, Y)

        gibbs_state = self.gibbs_sample(model_state)

        return (
            -self.log_marginal_likelihood(model_state, gibbs_state.ω).mean(0)
            / X.size(0)
        )


class PredictiveOVEPolyaGammaGP(OVEPolyaGammaGP):
    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()
        scores = self.set_forward(x)
        return self.loss_fn(scores, y_query)
