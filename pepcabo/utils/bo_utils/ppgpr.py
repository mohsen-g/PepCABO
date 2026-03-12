# ppgpr
from .base import DenseNetwork
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from botorch.posteriors.gpytorch import GPyTorchPosterior
import torch
from gpytorch.mlls import VariationalELBO
from gpytorch.kernels import *
import torch

class WeightedVariationalELBO(VariationalELBO):
    def _log_likelihood_term(self, variational_dist_f, target,weight, **kwargs):
        return (self.likelihood.expected_log_prob(target, variational_dist_f,**kwargs)*weight).sum(-1)

    def forward(self, approximate_dist_f, target,weight, **kwargs):
        # Get likelihood term and KL term
        num_batch = approximate_dist_f.event_shape[0]
        log_likelihood = self._log_likelihood_term(approximate_dist_f, target,weight, **kwargs).div(num_batch)
        kl_divergence = self.model.variational_strategy.kl_divergence().div(self.num_data / self.beta)

        # Add any additional registered loss terms
        added_loss = torch.zeros_like(log_likelihood)
        had_added_losses = False
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
            had_added_losses = True

        # Log prior term
        log_prior = torch.zeros_like(log_likelihood)
        for name, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))

        if self.combine_terms:
            return log_likelihood - kl_divergence + log_prior - added_loss, log_likelihood
        else:
            if had_added_losses:
                return log_likelihood, kl_divergence, log_prior, added_loss
            else:
                return log_likelihood, kl_divergence, log_prior
class WeightedPredictiveLogLikelihood(WeightedVariationalELBO):
    def _log_likelihood_term(self, approximate_dist_f, target,weight, **kwargs):
        return (self.likelihood.log_marginal(target, approximate_dist_f,**kwargs)*weight).sum(-1)


class GPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0) )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1
        self.likelihood = likelihood 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode
            # self.model.eval()
            self.likelihood.eval()
            dist = self.likelihood(self(X)) 

            return GPyTorchPosterior(mvn=dist)

class GPModelDKL(ApproximateGP):
    def __init__(self, inducing_points, likelihood, hidden_dims=(256, 256),dropout_prob=0.1):
        inducing_points=inducing_points[:,:64]
        feature_extractor = DenseNetwork(
            input_dim=inducing_points.size(-1),
            hidden_dims=hidden_dims,
            dropout_prob=dropout_prob
            ).to(inducing_points.device) # MLP
        inducing_points = feature_extractor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(GPModelDKL, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1 
        self.likelihood = likelihood
        self.feature_extractor = feature_extractor
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) # GP

    def __call__(self, x, *args, **kwargs):
        #print('xshape', x.shape)
        x=x[:,:64]
        x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()
            self.likelihood.eval()
            dist = self.likelihood(self(X))

            return GPyTorchPosterior(mvn=dist)

class CustomRBFKernelDeep(RBFKernel):
    has_lengthscale = True
    def __init__(self, input_dim=None, hidden_dims=None, ard_num_dims = None, batch_shape = None, active_dims = None, lengthscale_prior = None, lengthscale_constraint = None, eps = 0.000001, **kwargs):
        super().__init__(ard_num_dims, batch_shape, active_dims, lengthscale_prior, lengthscale_constraint, eps, **kwargs)
        self.feature_extract = DenseNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims
            ) # MLP
        self.last_norm = None
        self.inducing_norm = -1
        self.input_norm = -1
    def forward(self, x1, x2, diag=False, **params):
        E =  super().forward(self.feature_extract(x1), self.feature_extract(x2), diag, **params)
        return E

class GPModelDKLExtended(ApproximateGP):
    def __init__(self, inducing_points, likelihood, hidden_dims=(128, 128)):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(GPModelDKLExtended, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean() 
        kernel = CustomRBFKernelDeep(input_dim=inducing_points.shape[-1],hidden_dims=hidden_dims,ard_num_dims=hidden_dims[-1])
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
        self.num_outputs = 1 #must be one
        self.likelihood = likelihood
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) # GP

    def __call__(self, x, *args, **kwargs):
        return super().__call__(x, *args, **kwargs)
    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()
            self.likelihood.eval()
            dist = self.likelihood(self(X))
            return GPyTorchPosterior(mvn=dist)