"""
Custom samplers for sampling posteriors for Likelihood Estimation and
Ratio Estimation models. Currently supports emcee samplers for both sbi
and pydelfi backends, and pyro samplers only for the sbi backend.
"""

import logging
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
    from sbi.utils.sbiutils import within_support
    from tqdm.auto import tqdm
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

    def sample_batched(
        self,
        nsteps: int,
        x: Any,
        samples_per_draw: int = 5_000,
        show_progress_bars: bool = True,
    ) -> np.ndarray:
        """Sample from a batch of observations using progressive batch shrinkage.

        Each iteration evaluates the flow over all observations that still need
        more samples in a single GPU forward pass. Once an observation has
        collected enough accepted samples it is removed from the active batch,
        so no observation gates others and GPU utilisation remains high
        throughout.

        This avoids two problems present in sbi's built-in
        ``DirectPosterior.sample_batched``:

        1. The min-gating bug: the inner loop decrements ``num_remaining`` by
           the *minimum* accepted count across all observations, so a single
           high-leakage observation forces the loop to keep sampling for every
           already-finished observation.
        2. The batch-size cap: ``max_sampling_batch_size`` is divided by the
           number of observations, collapsing GPU parallelism for large batches.

        For ``EnsemblePosterior`` objects the method delegates to
        ``_sample_batched_ensemble``.

        Args:
            nsteps: Number of accepted posterior samples to collect per
                observation.
            x: Batch of observations, shape ``(N, feature_dim)``.
            samples_per_draw: Candidate samples drawn per observation per
                iteration.  The total forward-pass size each iteration is
                ``samples_per_draw * n_active``, which shrinks naturally as
                observations complete.
            show_progress_bars: Whether to show a tqdm progress bar.

        Returns:
            Array of shape ``(N, nsteps, theta_dim)``.
        """
        posterior = self.posterior

        if isinstance(posterior, EnsemblePosterior):
            return self._sample_batched_ensemble(
                posterior, nsteps, x, samples_per_draw, show_progress_bars
            )

        if not hasattr(posterior, "posterior_estimator"):
            raise RuntimeError(
                "sample_batched requires a DirectPosterior with a "
                ".posterior_estimator attribute. Use sample() for other "
                "posterior types."
            )

        estimator = posterior.posterior_estimator
        prior = posterior.prior
        device = getattr(posterior, "_device", "cpu")

        x_tensor = torch.as_tensor(
            np.asarray(x), dtype=torch.float32, device=device
        )
        N = x_tensor.shape[0]
        theta_dim = int(torch.tensor(estimator.input_shape).prod().item())

        accepted: list[list[torch.Tensor]] = [[] for _ in range(N)]
        counts = torch.zeros(N, dtype=torch.long)
        total_drawn = torch.zeros(N, dtype=torch.long)
        # Boolean mask on CPU; active observations are True.
        # Tensor indexing via nonzero() avoids O(N²) list.remove() cost.
        active_mask = torch.ones(N, dtype=torch.bool)

        pbar = tqdm(
            total=N * nsteps,
            desc=f"Batched sampling ({N} observations)",
            disable=not show_progress_bars,
        )

        # Maximum total samples across all active observations per forward pass.
        # Keeps the kernel launch within GPU hardware limits (e.g. AMD HIP and
        # NVIDIA CUDA both fail above ~2^24 elements in a single launch).
        # samples_per_draw is the *per-observation* budget; the actual draw is
        # capped so that samples_per_draw * n_active <= max_total_per_pass.
        max_total_per_pass = 50_000

        with torch.no_grad():
            estimator.eval()
            while active_mask.any():
                active_indices = active_mask.nonzero(as_tuple=True)[0]  # (n_active,)
                n_active = active_indices.shape[0]
                active_x = x_tensor[active_indices]  # (n_active, feature_dim)

                # Scale per-obs draw count so total stays within GPU limits.
                effective_draw = max(1, min(samples_per_draw, max_total_per_pass // n_active))

                # Single forward pass for all remaining observations.
                # Output shape: (effective_draw, n_active, theta_dim)
                candidates = estimator.sample(
                    torch.Size((effective_draw,)), condition=active_x
                )

                # Vectorised prior support check.
                # in_support: (effective_draw, n_active) boolean on CPU
                flat = candidates.reshape(-1, theta_dim)
                in_support = within_support(prior, flat).reshape(
                    effective_draw, n_active
                ).cpu()
                total_drawn[active_indices] += effective_draw

                # Per-observation accepted counts — one GPU reduction, no Python loop.
                new_per_obs = in_support.sum(dim=0)  # (n_active,) long

                # Update progress bar: credit newly accepted up to nsteps per obs.
                prev_counts = counts[active_indices].clone()
                counts[active_indices] += new_per_obs
                capped_increment = (
                    new_per_obs - (prev_counts + new_per_obs - nsteps).clamp(min=0)
                ).clamp(min=0)
                pbar.update(int(capped_increment.sum().item()))

                # Collect actual samples — iterate only over obs that got ≥1 sample.
                got_samples = new_per_obs.nonzero(as_tuple=True)[0]
                for local_i in got_samples.tolist():
                    global_i = int(active_indices[local_i].item())
                    mask = in_support[:, local_i]
                    accepted[global_i].append(candidates[mask, local_i].cpu())

                # Drop completed observations from the active set (vectorised).
                done_local = (counts[active_indices] >= nsteps)
                if done_local.any():
                    active_mask[active_indices[done_local]] = False

        pbar.close()

        rates = (counts.float() / total_drawn.float().clamp(min=1)).numpy()
        low, med, high = np.percentile(rates, [16, 50, 84])
        logging.debug(
            "Batched sampling acceptance rates — "
            f"16th/50th/84th: {low:.2%} / {med:.2%} / {high:.2%}"
        )
        if med < 0.01:
            logging.warning(
                f"Median acceptance rate is only {med:.2%}. Sampling may be "
                "slow. Consider wider priors or switching to MCMC."
            )

        return np.stack([
            torch.cat(accepted[i], dim=0)[:nsteps].numpy()
            for i in range(N)
        ])

    def _sample_batched_ensemble(
        self,
        posterior: "EnsemblePosterior",
        nsteps: int,
        x: Any,
        samples_per_draw: int,
        show_progress_bars: bool,
    ) -> np.ndarray:
        """Batched sampling for EnsemblePosterior.

        Assigns samples to components via multinomial draw matching the
        ensemble weights, then calls ``sample_batched`` on each component
        posterior independently.

        Args:
            posterior: The EnsemblePosterior to sample from.
            nsteps: Samples to collect per observation.
            x: Observations, shape ``(N, feature_dim)``.
            samples_per_draw: Passed through to each component's
                ``sample_batched`` call.
            show_progress_bars: Passed through to each component.

        Returns:
            Array of shape ``(N, nsteps, theta_dim)``.
        """
        x_arr = np.asarray(x)
        N = x_arr.shape[0]

        # Assign each of the nsteps draws to a component via multinomial.
        component_indices = torch.multinomial(
            posterior._weights, nsteps, replacement=True
        )
        unique_comps, comp_sizes = component_indices.unique(return_counts=True)

        per_galaxy: list[list[np.ndarray]] = [[] for _ in range(N)]

        for comp_idx, comp_size in zip(
            unique_comps.tolist(), comp_sizes.tolist()
        ):
            comp_posterior = posterior.posteriors[comp_idx]
            temp_sampler = DirectSampler(comp_posterior)
            # Shape: (N, comp_size, theta_dim)
            comp_samples = temp_sampler.sample_batched(
                comp_size, x_arr, samples_per_draw,
                show_progress_bars=show_progress_bars,
            )
            for i in range(N):
                per_galaxy[i].append(comp_samples[i])

        return np.stack([
            np.concatenate(per_galaxy[i], axis=0)[:nsteps]
            for i in range(N)
        ])


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
