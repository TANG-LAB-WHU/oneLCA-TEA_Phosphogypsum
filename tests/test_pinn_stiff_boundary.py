"""
Unit tests for Stiff Boundary PINNs and Adaptive Collocation Sampler.
"""

import importlib.util
import pytest
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def test_stiff_pinn_imports():
    """Verify that PyTorch dependencies can be imported if present."""
    if torch is None:
        pytest.skip("PyTorch not installed, skipping PINN tests.")
    
    from pgloop.stochastic_dynamics.stiff_pinn import StiffBoundaryPINN
    from pgloop.stochastic_dynamics.acr_sampler import AdaptiveCollocationSampler
    from pgloop.stochastic_dynamics.pinn_trainer import StiffPINNTrainer
    
    assert StiffBoundaryPINN is not None
    assert AdaptiveCollocationSampler is not None
    assert StiffPINNTrainer is not None


def test_stiff_pinn_forward_and_residual():
    """Verify forward pass and residual computation of StiffBoundaryPINN."""
    if torch is None:
        pytest.skip("PyTorch not installed.")
    
    from pgloop.stochastic_dynamics.stiff_pinn import StiffBoundaryPINN
    
    torch.set_default_dtype(torch.float64)
    model = StiffBoundaryPINN(hidden=[16, 16], activation="sine")
    
    x = torch.linspace(-1.0, 1.0, 32).unsqueeze(1).requires_grad_(True)
    t = torch.linspace(0.0, 1.0, 32).unsqueeze(1).requires_grad_(True)
    
    # Forward pass
    p = model(x, t)
    assert p.shape == (32, 1)
    assert (p >= 0).all(), "Outputs must be non-negative"
    
    # Residual computation
    def drift_fn(x_in, _t):
        return -x_in
        
    def diffusion_fn(x_in, _t):
        return 0.5 * torch.ones_like(x_in)
        
    res = model.residual(x, t, drift_fn, diffusion_fn)
    assert res.shape == (32, 1)
    assert not torch.isnan(res).any()


def test_adaptive_collocation_sampler():
    """Verify that the ACR sampler generates and refines collocation points."""
    if torch is None:
        pytest.skip("PyTorch not installed.")
        
    from pgloop.stochastic_dynamics.acr_sampler import AdaptiveCollocationSampler
    from pgloop.stochastic_dynamics.stiff_pinn import StiffBoundaryPINN
    
    torch.set_default_dtype(torch.float64)
    model = StiffBoundaryPINN(hidden=[16, 16], activation="tanh")
    
    sampler = AdaptiveCollocationSampler(
        x_min=-1.0,
        x_max=1.0,
        t_max=0.5,
        n_initial=100,
    )
    
    # Sample initial
    x_init, t_init = sampler.sample_initial()
    assert x_init.shape == (100, 1)
    assert t_init.shape == (100, 1)
    
    def drift_fn(x_in, _t):
        return -x_in
        
    def diffusion_fn(x_in, _t):
        return 0.5 * torch.ones_like(x_in)
        
    # Refine points
    x_ref, t_ref = sampler.refine_points(
        model,
        drift_fn,
        diffusion_fn,
        n_new_points=20,
        candidate_pool_size=100,
    )
    assert x_ref.shape == (120, 1)
    assert t_ref.shape == (120, 1)


def test_stiff_pinn_trainer_smoke():
    """Verify that StiffPINNTrainer training loop runs without errors."""
    if torch is None:
        pytest.skip("PyTorch not installed.")
        
    from pgloop.stochastic_dynamics.stiff_pinn import StiffBoundaryPINN
    from pgloop.stochastic_dynamics.pinn_trainer import StiffPINNTrainer
    from pgloop.stochastic_dynamics.fokker_planck import FokkerPlanck1DSolver
    from pgloop.stochastic_dynamics.eval import compare_with_numerical_solver
    
    torch.set_default_dtype(torch.float64)
    
    # Define simple OU drift/diffusion
    def drift_torch(x, _t):
        return -x
        
    def diff_torch(x, _t):
        return 0.5 * torch.ones_like(x)
        
    model = StiffBoundaryPINN(hidden=[16, 16], activation="sine")
    trainer = StiffPINNTrainer(
        model=model,
        drift_fn=drift_torch,
        diffusion_fn=diff_torch,
        x_min=-1.0,
        x_max=1.0,
        t_max=0.2,
        bc_type="no-flux",
        n_initial=100,
    )
    
    # Initial condition: narrow Gaussian
    def p0_fn(x):
        return torch.exp(-20.0 * x**2) / (np.sqrt(np.pi / 20.0))
        
    # Run short training (smoke test)
    history = trainer.train(
        p0_fn=p0_fn,
        n_epochs_adam=10,
        n_epochs_lbfgs=2,
        adaptive_sampling_freq=5,
        n_new_collocation_points=10,
        use_adaptive_weights=True,
    )
    
    assert len(history["loss_history"]) == 12
    assert history["n_collocation_points"] == 120
    assert "w_pde" in history["final_weights"]
    
    # Generate numerical baseline to verify compare_with_numerical_solver runs
    solver = FokkerPlanck1DSolver(x_min=-1.0, x_max=1.0, n_x=51)
    
    def drift_numpy(x, _t):
        return -x
        
    def diff_numpy(x, _t):
        return 0.5 * np.ones_like(x)
        
    p0_numpy = np.exp(-20.0 * solver.x**2) / (np.sqrt(np.pi / 20.0))
    
    trajectory = solver.evolve(
        p0=p0_numpy,
        drift_fn=drift_numpy,
        diffusion_fn=diff_numpy,
        dt=0.001,
        n_steps=200,
    )
    
    eval_results = compare_with_numerical_solver(model, trajectory)
    assert "rel_l2_error" in eval_results
    assert eval_results["p_pinn"].shape == (201, 51)
