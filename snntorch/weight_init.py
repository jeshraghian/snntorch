"""Spiking-aware weight initialization for snnTorch.

Provides fluctuation-driven initialization methods that account for neuron
membrane time constants, synaptic time constants, and presynaptic firing rates.

Based on:
    Rossbroich, J., Gygax, J. & Zenke, F. (2022).
    *Fluctuation-driven initialization for spiking neural network training.*
    Neuromorphic Computing and Engineering, 2(4), 044016.
    https://iopscience.iop.org/article/10.1088/2634-4386/ac97bb
"""

import math
import warnings
from typing import Optional, Tuple

import torch


def _epsilon_analytical(tau_mem: float, tau_syn: float) -> Tuple[float, float]:
    """Compute PSP kernel integrals analytically.

    From Rossbroich et al. (2022) Supplementary Material Section S1.
    Assumes exponential current-based synapses.

    Args:
        tau_mem: Membrane time constant in seconds.
        tau_syn: Synaptic time constant in seconds.

    Returns:
        Tuple of (epsilon_bar, epsilon_hat).
    """
    # Eq. S1: epsilon_bar = tau_syn
    epsilon_bar = tau_syn
    # Eq. S1: epsilon_hat = tau_syn^2 / (2 * (tau_syn + tau_mem))
    epsilon_hat = (tau_syn**2) / (2.0 * (tau_syn + tau_mem))
    return epsilon_bar, epsilon_hat


def _epsilon_numerical(
    tau_mem: float, tau_syn: float, dt: float
) -> Tuple[float, float]:
    """Compute PSP kernel integrals numerically by simulating LIF dynamics.

    Simulates the membrane response to a single spike input through
    an exponential current-based synapse, matching the approach used in
    the stork library (Rossbroich et al. 2022).

    Args:
        tau_mem: Membrane time constant in seconds.
        tau_syn: Synaptic time constant in seconds.
        dt: Simulation timestep in seconds.

    Returns:
        Tuple of (epsilon_bar, epsilon_hat).
    """
    tau_max = max(tau_mem, tau_syn)
    num_steps = int(tau_max * 10 / dt)

    # Simulate LIF dynamics with single spike input (matching stork library)
    kernel = torch.empty(num_steps, dtype=torch.float64)
    current = 1.0  # Current variable for single spike input
    voltage = 0.0  # Membrane potential
    dcy_mem = math.exp(-dt / tau_mem)
    dcy_syn = math.exp(-dt / tau_syn)

    for i in range(num_steps):
        kernel[i] = voltage
        voltage = dcy_mem * voltage + (1.0 - dcy_mem) * current
        current *= dcy_syn

    # Compute integrals
    epsilon_bar = kernel.sum() * dt
    epsilon_hat = (kernel**2).sum() * dt

    return epsilon_bar.item(), epsilon_hat.item()


def compute_psp_kernel_integrals(
    tau_mem: float,
    tau_syn: float,
    dt: float = 1e-3,
    mode: str = "analytical",
) -> Tuple[float, float]:
    """Compute the integrals of the PSP kernel (epsilon_bar and epsilon_hat).

    These integrals are used by the fluctuation-driven initialization to
    compute weight parameters from target membrane statistics.

    Args:
        tau_mem: Membrane time constant in seconds.
        tau_syn: Synaptic time constant in seconds.
        dt: Simulation timestep in seconds. Only used when mode="numerical".
        mode: "analytical" or "numerical". Default: "analytical".

    Returns:
        Tuple of (epsilon_bar, epsilon_hat).

    Raises:
        ValueError: If tau_mem or tau_syn are not positive.
        ValueError: If mode is not "analytical" or "numerical".

    Example:
        >>> from snntorch import weight_init
        >>> ebar, ehat = weight_init.compute_psp_kernel_integrals(
        ...     tau_mem=0.02, tau_syn=0.005
        ... )
        >>> print(f"epsilon_bar={ebar:.4f}, epsilon_hat={ehat:.6f}")
    """
    if tau_mem <= 0:
        raise ValueError(f"tau_mem must be positive, got {tau_mem}")
    if tau_syn <= 0:
        raise ValueError(f"tau_syn must be positive, got {tau_syn}")

    valid_modes = ("analytical", "numerical")
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

    if mode == "analytical":
        return _epsilon_analytical(tau_mem, tau_syn)
    else:
        return _epsilon_numerical(tau_mem, tau_syn, dt)


def fluctuation_driven_normal_(
    tensor: torch.Tensor,
    tau_mem: float,
    tau_syn: float,
    nu: float,
    mu_u: float = 0.0,
    sigma_u: float = 1.0,
    theta: float = 1.0,
    dt: float = 1e-3,
    mode: str = "analytical",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Initialize weights from a normal distribution using
    fluctuation-driven initialization.

    This method targets a membrane potential with mean ``mu_u`` and standard
    deviation ``sigma_u``, placing neurons near threshold for optimal
    surrogate gradient learning.

    Args:
        tensor: Weight tensor to initialize (modified in-place).
        tau_mem: Membrane time constant in seconds.
        tau_syn: Synaptic time constant in seconds.
        nu: Estimated presynaptic firing rate in Hz.
        mu_u: Target mean membrane potential. Default: 0.0 (balanced).
        sigma_u: Target standard deviation of membrane potential. Default: 1.0.
        theta: Spike threshold. Default: 1.0.
        dt: Simulation timestep in seconds. Default: 1e-3.
        mode: "analytical" or "numerical" for PSP kernel integrals.
            Default: "analytical".
        generator: Optional torch.Generator for reproducibility.

    Returns:
        The initialized tensor (same object, modified in-place).

    Raises:
        ValueError: If computed sigma_w is negative.
        ValueError: If tensor has fewer than 2 dimensions.
        ValueError: If mode is not "analytical" or "numerical".
        ValueError: If any parameter is not positive.

    Example:
        >>> import torch.nn as nn
        >>> import snntorch as snn
        >>> fc = nn.Linear(784, 128)
        >>> snn.weight_init.fluctuation_driven_normal_(
        ...     fc.weight, tau_mem=0.02, tau_syn=0.005, nu=10.0, sigma_u=1.0
        ... )
    """
    # Validate parameters
    if tau_mem <= 0:
        raise ValueError(f"tau_mem must be positive, got {tau_mem}")
    if tau_syn <= 0:
        raise ValueError(f"tau_syn must be positive, got {tau_syn}")
    if nu <= 0:
        raise ValueError(f"nu must be positive, got {nu}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if sigma_u <= 0:
        raise ValueError(f"sigma_u must be positive, got {sigma_u}")
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")

    if tensor.dim() < 2:
        raise ValueError(
            f"Weight tensor must have at least 2 dimensions, "
            f"got {tensor.dim()}D tensor."
        )

    if 0 in tensor.shape:
        warnings.warn(
            "Initializing zero-element tensor is a no-op.",
            stacklevel=2,
        )
        return tensor

    valid_modes = ("analytical", "numerical")
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'.")

    # Compute PSP kernel integrals
    epsilon_bar, epsilon_hat = compute_psp_kernel_integrals(
        tau_mem, tau_syn, dt, mode
    )

    # Compute fan-in
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(tensor)

    # Compute weight parameters (Eqs. 6-7 from paper)
    # mu_w = mu_u / (n * nu * epsilon_bar)
    mu_w = mu_u / (fan_in * nu * epsilon_bar)

    # sigma_w^2 = sigma_u^2 / (n * nu * epsilon_hat) - mu_w^2
    weight_var = (sigma_u**2) / (fan_in * nu * epsilon_hat) - mu_w**2

    if weight_var < 0:
        raise ValueError(
            "Computed weight variance is negative "
            "({}). This occurs when the target membrane "
            "statistics are incompatible with the network "
            "parameters. Try increasing sigma_u or nu, "
            "or decreasing mu_u.".format(round(weight_var, 6))
        )

    sigma_w = math.sqrt(weight_var)

    # Fill tensor with normal distribution
    with torch.no_grad():
        tensor.normal_(mu_w, sigma_w, generator=generator)

    return tensor


def fluctuation_driven_normal_centered_(
    tensor: torch.Tensor,
    tau_mem: float,
    tau_syn: float,
    nu: float,
    sigma_u: float = 1.0,
    dt: float = 1e-3,
    mode: str = "analytical",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Initialize weights using centered fluctuation-driven
    initialization (mu_u=0).

    This is a simplified version assuming balanced excitation/inhibition
    where the target mean membrane potential is zero. This is the most
    commonly used variant.

    The weight distribution is N(0, sigma_w) where:
        sigma_w = sigma_u / sqrt(n * nu * epsilon_hat)

    (Eq. 9 from Rossbroich et al. 2022)

    Args:
        tensor: Weight tensor to initialize (modified in-place).
        tau_mem: Membrane time constant in seconds.
        tau_syn: Synaptic time constant in seconds.
        nu: Estimated presynaptic firing rate in Hz.
        sigma_u: Target standard deviation of membrane potential. Default: 1.0.
        dt: Simulation timestep in seconds. Default: 1e-3.
        mode: "analytical" or "numerical" for PSP kernel integrals.
            Default: "analytical".
        generator: Optional torch.Generator for reproducibility.

    Returns:
        The initialized tensor (same object, modified in-place).

    Raises:
        ValueError: If tensor has fewer than 2 dimensions.
        ValueError: If mode is not "analytical" or "numerical".
        ValueError: If any parameter is not positive.

    Example:
        >>> import torch.nn as nn
        >>> import snntorch as snn
        >>> conv = nn.Conv2d(64, 128, 3, padding=1)
        >>> snn.weight_init.fluctuation_driven_normal_centered_(
        ...     conv.weight, tau_mem=0.02, tau_syn=0.005, nu=15.0
        ... )
    """
    # Delegate to general function with mu_u=0
    # When mu_u=0, the formula simplifies to:
    # sigma_w = sigma_u / sqrt(n * nu * epsilon_hat)
    # (theta is not used in this path)
    return fluctuation_driven_normal_(
        tensor,
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        nu=nu,
        mu_u=0.0,
        sigma_u=sigma_u,
        theta=1.0,  # Not used when mu_u=0, but required by general function
        dt=dt,
        mode=mode,
        generator=generator,
    )
