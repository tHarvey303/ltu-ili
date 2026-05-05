"""
Example script demonstrating the usage of BlackjaxNestedSampler.

This script shows how to use the BlackjaxNestedSampler with an sbi posterior
for Neural Likelihood Estimation (NLE).

Requirements:
    pip install blackjax jax jaxlib

Usage:
    python example_blackjax_nested_sampler.py
"""

import numpy as np
import torch
import ili

# Create synthetic data
def simulator(params):
    """Simple toy simulator."""
    x = np.linspace(0, 10, 20)
    y = 3 * params[0] * np.sin(x) + params[1] * x ** 2 - 2 * params[2] * x
    y += 1 * np.random.randn(len(x))
    return y

# Generate training data
theta = np.random.rand(200, 3)  # 200 simulations, 3 parameters
x = np.array([simulator(t) for t in theta])

# Create dataloader
loader = ili.dataloaders.NumpyLoader(x=x, theta=theta)

# Define prior
device = 'cuda' if torch.cuda.is_available() else 'cpu'
prior = ili.utils.Uniform(low=[0, 0, 0], high=[1, 1, 1], device=device)

# Define and train NLE model
nets = [ili.utils.load_nde_sbi(engine='NLE', model='maf', hidden_features=16)]

runner = ili.inference.InferenceRunner.load(
    backend='sbi',
    engine='NLE',
    prior=prior,
    nets=nets,
    device=device,
    train_args={'max_num_epochs': 5},
)

posterior, _ = runner(loader)

# Create a test observation
x_obs = simulator(theta[0])

# Sample using BlackjaxNestedSampler
print("\nSampling with BlackjaxNestedSampler...")

# Method 1: Direct sampler usage
from ili.utils.samplers import BlackjaxNestedSampler

sampler = BlackjaxNestedSampler(
    posterior=posterior,
    num_live_points=500,
    max_samples=10000,
    term_cond={'dlogz': 0.1}
)

samples = sampler.sample(
    nsteps=1000,
    x=x_obs,
    progress=True
)

print(f"Generated {len(samples)} samples")
print(f"Sample shape: {samples.shape}")
print(f"Sample mean: {np.mean(samples, axis=0)}")
print(f"Sample std: {np.std(samples, axis=0)}")
print(f"True parameters: {theta[0]}")

# Method 2: Using validation metrics
print("\nUsing BlackjaxNestedSampler via PosteriorSamples metric...")

metric = ili.validation.metrics.PosteriorSamples(
    num_samples=1000,
    sample_method='blackjax_nested',
    sample_params={
        'num_live_points': 500,
        'max_samples': 10000,
        'term_cond': {'dlogz': 0.1}
    },
    labels=['theta_0', 'theta_1', 'theta_2']
)

samples2 = metric(
    posterior=posterior,
    x_obs=x_obs,
    theta_fid=theta[0],
    x=x[:5],
    theta=theta[:5]
)

print(f"Generated samples shape: {samples2.shape}")
print("\nBlackjaxNestedSampler example completed successfully!")
