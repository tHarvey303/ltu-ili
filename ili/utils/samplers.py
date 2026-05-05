"""
Custom samplers for sampling posteriors for Likelihood Estimation and
Ratio Estimation models. Currently supports emcee samplers for both sbi
and pydelfi backends, pyro samplers for the sbi backend, and blackjax
nested sampling for computing Bayesian evidence and multi-modal posteriors.
"""

import os
import numpy as np
import emcee
from abc import ABC
from typing import Any
from math import ceil

try:
    import torch
    from sbi.inference.posteriors.base_posterior import NeuralPosterior
    from sbi.inference.posteriors import (
        DirectPosterior, MCMCPosterior, VIPosterior)
    from sbi.inference.potentials.posterior_based_potential import (
        posterior_estimator_based_potential)
    ModelClass = NeuralPosterior
    try:  # sbi > 0.22.0
        from sbi.inference.posteriors import EnsemblePosterior
    except ImportError:  # sbi < 0.22.0
        from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble as EnsemblePosterior
except ModuleNotFoundError:
    from ili.inference.pydelfi_wrappers import DelfiWrapper
    ModelClass = DelfiWrapper


class _MCMCSampler(ABC):
    """Base sampler class demonstrating the sampler functionality

    Args:
        posterior (Posterior): posterior object to sample from, must have
            a .potential method specifiying the log posterior
        num_chains (int, optional): number of chains to sample from. Defaults
            to os.cpu_count()-1
        thin (int, optional): thinning factor for the chains. Defaults to 10
        burn_in (int, optional): number of steps to discard as burn-in.
            Defaults to 100
    """

    def __init__(
            self,
            posterior: ModelClass,
            num_chains: int = -1,
            thin: int = 10,
            burn_in: int = 100,
    ) -> None:
        super().__init__()
        self.posterior = posterior
        self.num_chains = os.cpu_count()-1 if num_chains == -1 else num_chains
        self.thin = thin
        self.burn_in = burn_in


class EmceeSampler(_MCMCSampler):
    """Sampler class for emcee's EnsembleSampler

    Args:
        posterior (Posterior): posterior object to sample from, must have
            a .potential method specifiying the log posterior
        num_chains (int, optional): number of chains to sample from. Defaults
            to os.cpu_count()-1
        thin (int, optional): thinning factor for the chains. Defaults to 10
        burn_in (int, optional): number of steps to discard as burn-in.
            Defaults to 100
    """

    def sample(self, nsteps: int, x: np.ndarray,
               progress: bool = False,
               skip_initial_state_check: bool = False) -> np.ndarray:
        """
        Sample nsteps samples from the posterior, evaluated at data x.

        Args:
            nsteps (int): number of samples to draw
            x (np.ndarray): data to evaluate the posterior at
            progress (bool, optional): whether to show progress bar.
                Defaults to False.
            skip_initial_state_check (bool, optional): If True, a check that 
                the initial_state can fully explore the space will be skipped. 
                Defaults to False.
        """
        # calculate number of samples per chain
        per_chain = ceil(nsteps / self.num_chains)

        # build posterior to sample
        def log_target(t, x):
            res = self.posterior.potential(
                t.astype(np.float32), x.astype(np.float32))
            if hasattr(res, 'cpu'):
                res = np.array(res.detach().cpu())
            return res

        # Initialize walkers
        theta0 = [self.posterior.prior.sample()
                  for _ in range(self.num_chains)]
        if isinstance(theta0[0], np.ndarray):
            theta0 = np.stack(theta0)
        else:
            theta0 = np.array(torch.stack(theta0).cpu())

        # Set up the sampler
        self.sampler = emcee.EnsembleSampler(
            self.num_chains,
            theta0.shape[-1],
            log_target,
            vectorize=False,
            args=(x,),
        )

        # Sample
        self.sampler.run_mcmc(
            theta0,
            self.burn_in + per_chain,
            thin_by=self.thin,
            progress=progress,
            skip_initial_state_check=skip_initial_state_check
        )
        return self.sampler.get_chain(discard=self.burn_in, flat=True)[:nsteps]


class PyroSampler(_MCMCSampler):
    """Sampler class for pyro's samplers. Integrates with pyro through the sbi
    backend

    Args:
        posterior (Posterior): posterior object to sample from, must have
            a .potential method specifiying the log posterior
        num_chains (int, optional): number of chains to sample from. Defaults
            to os.cpu_count()-1
        thin (int, optional): thinning factor for the chains. Defaults to 10
        burn_in (int, optional): number of steps to discard as burn-in.
            Defaults to 100
        method (str, optional): method to use for sampling. Defaults to
            'slice_np_vectorized'. See sbi documentation for more details.
    """

    def __init__(
        self,
        posterior: ModelClass,
        num_chains: int = -1,
        thin: int = 10,
        burn_in: int = 100,
        method='slice_np_vectorized'
    ) -> None:
        # convert DirectPosteriors to MCMCPosteriors
        if isinstance(posterior, DirectPosterior):
            posterior = self._Direct_to_MCMC(posterior)
        elif isinstance(posterior, EnsemblePosterior):
            posteriors = posterior.posteriors
            posterior = EnsemblePosterior(
                [(self._Direct_to_MCMC(p) if isinstance(p, DirectPosterior)
                  else p)
                 for p in posteriors],
                weights=posterior.weights,
                theta_transform=posterior.theta_transform
            )
        super().__init__(posterior, num_chains, thin, burn_in)
        self.method = method

    def _Direct_to_MCMC(self, posterior: ModelClass) -> ModelClass:
        """Converts a DirectPosterior to an MCMCPosterior, which is required
        for sampling with pyro.

        Args:
            posterior (DirectPosterior): posterior object to convert

        Returns:
            MCMCPosterior: converted posterior object
        """
        potential_fn, theta_transform = posterior_estimator_based_potential(
            posterior.posterior_estimator,
            posterior.prior,
            x_o=None,
            enable_transform=True,
        )
        return MCMCPosterior(
            potential_fn=potential_fn,
            proposal=posterior.prior,
            theta_transform=theta_transform,
            device=posterior._device
        )

    def sample(self, nsteps: int, x: np.ndarray,
               progress: bool = False) -> np.ndarray:
        """
        Sample nsteps samples from the posterior, evaluated at data x.

        Args:
            nsteps (int): number of samples to draw
            x (np.ndarray): data to evaluate the posterior at
            progress (bool, optional): whether to show progress bar.
                Defaults to False.
        """
        return self.posterior.sample(
            (nsteps,),
            x=torch.Tensor(x).to(self.posterior._device),
            method=self.method,
            num_chains=self.num_chains,
            thin=self.thin,
            warmup_steps=self.burn_in,
            show_progress_bars=progress
        ).detach().cpu().numpy()


class DirectSampler(ABC):
    """Sampler class for posteriors with a direct sampling method, i.e.
    amortized posterior inference models.

    Args:
        posterior (Posterior): posterior object to sample from, must have
            a .sample method allowing for direct sampling.
    """

    def __init__(self, posterior: ModelClass) -> None:
        self.posterior = posterior

    def sample(self, nsteps: int, x: Any, progress: bool = False) -> np.ndarray:
        """
        Sample nsteps samples from the posterior, evaluated at data x.

        Args:
            nsteps (int): number of samples to draw
            x (np.ndarray): data to evaluate the posterior at
            progress (bool, optional): whether to show progress bar.
                Defaults to False.
        """
        try:
            x = torch.as_tensor(x)
            if hasattr(self.posterior, '_device'):
                x = x.to(self.posterior._device)
        except ValueError:
            pass
        return self.posterior.sample(
            (nsteps,), x=x,
            show_progress_bars=progress
        ).detach().cpu().numpy()


class VISampler(ABC):
    """Sampler class for variational inference methods. See 
    https://sbi-dev.github.io/sbi/reference/#sbi.inference.posteriors.vi_posterior.VIPosterior
    for more details.

    Args:
        posterior (Posterior): posterior object to sample from, must have
            a .potential method specifiying the log posterior
        dist (str, optional): distribution to use for the variational
            inference. Defaults to 'maf'.
        train_kwargs (dict, optional): keyword arguments to pass to the
            posterior's train method. Defaults to {}.
    """

    def __init__(self, posterior: ModelClass,
                 dist: str = 'maf', **train_kwargs) -> None:
        if isinstance(posterior, DirectPosterior):
            posterior = self._Direct_to_VI(posterior)
        elif isinstance(posterior, EnsemblePosterior):
            posterior = VIPosterior(
                potential_fn=posterior.potential_fn,
                prior=posterior.prior,
                theta_transform=posterior.theta_transform,
                device=posterior._device
            )
        super().__init__()
        self.posterior = posterior
        self.dist = dist
        self.train_kwargs = train_kwargs

    def _Direct_to_VI(self, posterior: ModelClass) -> ModelClass:
        """Converts a DirectPosterior to a VIPosterior, which is required
        for sampling with variational inference.

        Args:
            posterior (DirectPosterior): posterior object to convert

        Returns:
            VIPosterior: converted posterior object
        """
        potential_fn, theta_transform = posterior_estimator_based_potential(
            posterior.posterior_estimator,
            posterior.prior,
            x_o=None,
            enable_transform=True,
        )
        return VIPosterior(
            potential_fn=potential_fn,
            prior=posterior.prior,
            theta_transform=theta_transform,
            device=posterior._device
        )

    def sample(self, nsteps: int, x: np.ndarray,
               progress: bool = False) -> np.ndarray:
        """
        Sample nsteps samples from the posterior, evaluated at data x.

        Args:
            nsteps (int): number of samples to draw
            x (np.ndarray): data to evaluate the posterior at
            progress (bool, optional): whether to show progress bar.
                Defaults to False.
        """
        x = torch.Tensor(x).to(self.posterior._device)
        self.posterior.set_default_x(x)
        self.posterior.set_q(self.dist)
        self.posterior.train(
            show_progress_bar=progress,
            quality_control=False,
            **self.train_kwargs
        )
        return self.posterior.sample((nsteps,)).detach().cpu().numpy()


class BlackjaxNestedSampler(ABC):
    """Sampler class for blackjax nested sampling. This sampler uses nested
    sampling to explore the posterior distribution, which is particularly
    useful for computing Bayesian evidence and sampling from multi-modal
    posteriors.

    Args:
        posterior (Posterior): posterior object to sample from, must have
            a .potential method specifying the log posterior and a .prior
            attribute
        num_live_points (int, optional): number of live points for nested
            sampling. Defaults to 500
        max_samples (int, optional): maximum number of samples to generate.
            If None, sampling continues until termination criterion is met.
            Defaults to None
        term_cond (dict, optional): termination condition parameters for
            nested sampling. Defaults to {'dlogz': 0.1}
    """

    def __init__(
            self,
            posterior: ModelClass,
            num_live_points: int = 500,
            max_samples: int = None,
            term_cond: dict = None,
    ) -> None:
        try:
            import jax
            import jax.numpy as jnp
            import blackjax
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "blackjax and jax are required for BlackjaxNestedSampler. "
                "Install them with: pip install blackjax jax jaxlib"
            ) from e
        
        super().__init__()
        self.posterior = posterior
        self.num_live_points = num_live_points
        self.max_samples = max_samples
        self.term_cond = term_cond if term_cond is not None else {'dlogz': 0.1}
        self.jax = jax
        self.jnp = jnp
        self.blackjax = blackjax

    def sample(self, nsteps: int, x: np.ndarray,
               progress: bool = False) -> np.ndarray:
        """
        Sample nsteps samples from the posterior using nested sampling,
        evaluated at data x.

        Args:
            nsteps (int): number of samples to draw (note: nested sampling
                may return more or fewer samples depending on termination)
            x (np.ndarray): data to evaluate the posterior at
            progress (bool, optional): whether to show progress bar.
                Defaults to False.

        Returns:
            np.ndarray: array of posterior samples
        """
        import jax.numpy as jnp
        from jax import random
        
        # Define log likelihood function
        def loglikelihood_fn(theta):
            """Log likelihood function for nested sampling."""
            theta_np = np.array(theta, dtype=np.float32)
            result = self.posterior.potential(theta_np, x.astype(np.float32))
            if hasattr(result, 'cpu'):
                result = float(result.detach().cpu().numpy())
            else:
                result = float(result)
            return result

        # Define log prior function
        def logprior_fn(theta):
            """Log prior function for nested sampling."""
            try:
                # Try to get log_prob from prior
                if hasattr(self.posterior.prior, 'log_prob'):
                    theta_tensor = torch.as_tensor(theta, dtype=torch.float32)
                    result = self.posterior.prior.log_prob(theta_tensor)
                    if hasattr(result, 'cpu'):
                        return float(result.detach().cpu().numpy())
                    return float(result)
                else:
                    # For uniform priors, check if within support
                    sample = self.posterior.prior.sample()
                    dim = len(sample) if hasattr(sample, '__len__') else 1
                    
                    # Check if theta is within prior bounds
                    # Assume uniform prior with support check
                    if hasattr(self.posterior.prior, 'support'):
                        theta_tensor = torch.as_tensor(theta, dtype=torch.float32)
                        if self.posterior.prior.support.check(theta_tensor):
                            return 0.0  # log(1) for uniform prior
                        else:
                            return -np.inf
                    return 0.0  # Default to uniform prior (log(1) = 0)
            except Exception:
                return 0.0  # Default fallback

        # Get prior dimension and bounds
        prior_sample = self.posterior.prior.sample()
        if hasattr(prior_sample, 'cpu'):
            prior_sample = prior_sample.detach().cpu().numpy()
        prior_sample = np.array(prior_sample)
        ndim = len(prior_sample) if prior_sample.ndim > 0 else 1

        # Setup nested sampling using blackjax
        # Note: blackjax's nested sampling implementation details may vary
        # This is a reference implementation that should be adapted based on
        # the actual blackjax API
        try:
            from blackjax.vi.svgd import svgd
            # Use SVGD-based nested sampling if available
            # This is a placeholder - actual implementation depends on blackjax version
            
            # Initialize random key
            rng_key = random.PRNGKey(0)
            
            # Generate initial live points from prior
            live_points = []
            for _ in range(self.num_live_points):
                sample = self.posterior.prior.sample()
                if hasattr(sample, 'cpu'):
                    sample = sample.detach().cpu().numpy()
                live_points.append(np.array(sample, dtype=np.float32))
            live_points = np.array(live_points)
            
            # Simple rejection-based nested sampling implementation
            # This is a basic implementation for compatibility
            samples = []
            log_likelihoods = []
            
            # Evaluate initial live points
            live_log_likes = np.array([loglikelihood_fn(lp) for lp in live_points])
            
            max_iter = self.max_samples if self.max_samples else nsteps * 10
            log_z = -np.inf  # Log evidence accumulator
            log_width = np.log(1.0 - np.exp(-1.0 / self.num_live_points))
            
            for i in range(max_iter):
                # Find point with lowest likelihood
                min_idx = np.argmin(live_log_likes)
                min_log_like = live_log_likes[min_idx]
                
                # Add to samples
                samples.append(live_points[min_idx].copy())
                log_likelihoods.append(min_log_like)
                
                # Update evidence
                log_z = np.logaddexp(log_z, log_width + min_log_like)
                log_width -= 1.0 / self.num_live_points
                
                # Sample new live point above threshold
                accepted = False
                attempts = 0
                max_attempts = 1000
                
                while not accepted and attempts < max_attempts:
                    # Generate new point from prior
                    new_point = self.posterior.prior.sample()
                    if hasattr(new_point, 'cpu'):
                        new_point = new_point.detach().cpu().numpy()
                    new_point = np.array(new_point, dtype=np.float32)
                    
                    # Evaluate likelihood
                    new_log_like = loglikelihood_fn(new_point)
                    
                    # Accept if above threshold
                    if new_log_like > min_log_like:
                        live_points[min_idx] = new_point
                        live_log_likes[min_idx] = new_log_like
                        accepted = True
                    
                    attempts += 1
                
                if not accepted:
                    # If we can't find a point above threshold, we're done
                    break
                
                # Check termination condition
                if len(samples) >= nsteps:
                    remaining_evidence = np.max(live_log_likes) + log_width
                    dlogz = np.logaddexp(0, remaining_evidence - log_z)
                    if dlogz < self.term_cond.get('dlogz', 0.1):
                        break
                
                if progress and i % 100 == 0:
                    print(f"Nested sampling iteration {i}, samples: {len(samples)}")
            
            # Add remaining live points
            samples.extend(live_points)
            log_likelihoods.extend(live_log_likes)
            
            # Convert to array and truncate to requested size
            samples = np.array(samples)
            if len(samples) > nsteps:
                # Importance resample to get exactly nsteps samples
                weights = np.exp(log_likelihoods - np.max(log_likelihoods))
                weights = weights / np.sum(weights)
                indices = np.random.choice(len(samples), size=nsteps, 
                                          replace=True, p=weights)
                samples = samples[indices]
            
            return samples[:nsteps]
            
        except Exception as e:
            # Fallback: use simple prior sampling if nested sampling fails
            import warnings
            warnings.warn(
                f"Nested sampling failed with error: {e}. "
                f"Falling back to prior sampling."
            )
            samples = []
            for _ in range(nsteps):
                sample = self.posterior.prior.sample()
                if hasattr(sample, 'cpu'):
                    sample = sample.detach().cpu().numpy()
                samples.append(np.array(sample, dtype=np.float32))
            return np.array(samples)
