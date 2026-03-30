"""
Finite-difference baseline solver for 1D Fokker-Planck equations.
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class FokkerPlanckTrajectory:
    x_grid: np.ndarray
    t_grid: np.ndarray
    pdf_t: np.ndarray


@dataclass
class FokkerPlanck2DTrajectory:
    x_grid: np.ndarray
    y_grid: np.ndarray
    t_grid: np.ndarray
    pdf_t: np.ndarray  # shape: [time, nx, ny]


class FokkerPlanck1DSolver:
    """
    Solve:
      d_t p = -d_x(f(x,t)p) + 0.5 d_xx(g(x,t)^2 p)
    on a fixed 1D grid with simple no-flux boundary approximation.
    """

    def __init__(self, x_min: float, x_max: float, n_x: int = 201):
        self.x = np.linspace(x_min, x_max, n_x)
        self.dx = self.x[1] - self.x[0]

    def evolve(
        self,
        p0: np.ndarray,
        drift_fn: Callable[[np.ndarray, float], np.ndarray],
        diffusion_fn: Callable[[np.ndarray, float], np.ndarray],
        dt: float,
        n_steps: int,
    ) -> FokkerPlanckTrajectory:
        p = self._normalize(np.maximum(p0.astype(float), 0.0))
        pdf_stack: List[np.ndarray] = [p.copy()]
        t_grid = [0.0]

        for k in range(1, n_steps + 1):
            t = k * dt
            f = drift_fn(self.x, t)
            g = diffusion_fn(self.x, t)
            d2 = g * g

            adv_flux = f * p
            diff_flux = d2 * p

            d_adv = self._ddx(adv_flux)
            d_diff = self._d2dx2(diff_flux)
            p = p + dt * (-d_adv + 0.5 * d_diff)

            # Numerical safety and probability constraints
            p = np.maximum(p, 0.0)
            p = self._normalize(p)

            pdf_stack.append(p.copy())
            t_grid.append(t)

        return FokkerPlanckTrajectory(
            x_grid=self.x.copy(),
            t_grid=np.asarray(t_grid),
            pdf_t=np.vstack(pdf_stack),
        )

    def _ddx(self, y: np.ndarray) -> np.ndarray:
        out = np.zeros_like(y)
        out[1:-1] = (y[2:] - y[:-2]) / (2.0 * self.dx)
        out[0] = (y[1] - y[0]) / self.dx
        out[-1] = (y[-1] - y[-2]) / self.dx
        return out

    def _d2dx2(self, y: np.ndarray) -> np.ndarray:
        out = np.zeros_like(y)
        out[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (self.dx * self.dx)
        out[0] = out[1]
        out[-1] = out[-2]
        return out

    def _normalize(self, p: np.ndarray) -> np.ndarray:
        mass = np.trapezoid(p, self.x)
        if mass <= 0:
            # fallback to narrow positive density around center
            p = np.exp(-0.5 * (self.x / max(self.dx, 1e-6)) ** 2)
            mass = np.trapezoid(p, self.x)
        return p / mass


class FokkerPlanck2DSolver:
    """
    Lightweight 2D finite-difference baseline for phase-4 validation.
    Uses diagonal diffusion approximation:
      d_t p = -(d_x(fx p) + d_y(fy p)) + 0.5*(d_xx(dx2 p) + d_yy(dy2 p))
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        n_x: int = 81,
        n_y: int = 81,
    ):
        self.x = np.linspace(x_min, x_max, n_x)
        self.y = np.linspace(y_min, y_max, n_y)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="ij")

    def evolve(
        self,
        p0: np.ndarray,
        drift_fn: Callable[[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]],
        diffusion_fn: Callable[[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]],
        dt: float,
        n_steps: int,
    ) -> FokkerPlanck2DTrajectory:
        p = self._normalize(np.maximum(p0.astype(float), 0.0))
        stack: List[np.ndarray] = [p.copy()]
        t_grid = [0.0]

        for k in range(1, n_steps + 1):
            t = k * dt
            fx, fy = drift_fn(self.xx, self.yy, t)
            gx, gy = diffusion_fn(self.xx, self.yy, t)
            dx2 = gx * gx
            dy2 = gy * gy

            adv_x = self._ddx(fx * p)
            adv_y = self._ddy(fy * p)
            diff_x = self._d2dx2(dx2 * p)
            diff_y = self._d2dy2(dy2 * p)

            p = p + dt * (-(adv_x + adv_y) + 0.5 * (diff_x + diff_y))
            p = self._normalize(np.maximum(p, 0.0))
            stack.append(p.copy())
            t_grid.append(t)

        return FokkerPlanck2DTrajectory(
            x_grid=self.x.copy(),
            y_grid=self.y.copy(),
            t_grid=np.asarray(t_grid),
            pdf_t=np.stack(stack, axis=0),
        )

    def _ddx(self, z: np.ndarray) -> np.ndarray:
        out = np.zeros_like(z)
        out[1:-1, :] = (z[2:, :] - z[:-2, :]) / (2.0 * self.dx)
        out[0, :] = (z[1, :] - z[0, :]) / self.dx
        out[-1, :] = (z[-1, :] - z[-2, :]) / self.dx
        return out

    def _ddy(self, z: np.ndarray) -> np.ndarray:
        out = np.zeros_like(z)
        out[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / (2.0 * self.dy)
        out[:, 0] = (z[:, 1] - z[:, 0]) / self.dy
        out[:, -1] = (z[:, -1] - z[:, -2]) / self.dy
        return out

    def _d2dx2(self, z: np.ndarray) -> np.ndarray:
        out = np.zeros_like(z)
        out[1:-1, :] = (z[2:, :] - 2.0 * z[1:-1, :] + z[:-2, :]) / (self.dx * self.dx)
        out[0, :] = out[1, :]
        out[-1, :] = out[-2, :]
        return out

    def _d2dy2(self, z: np.ndarray) -> np.ndarray:
        out = np.zeros_like(z)
        out[:, 1:-1] = (z[:, 2:] - 2.0 * z[:, 1:-1] + z[:, :-2]) / (self.dy * self.dy)
        out[:, 0] = out[:, 1]
        out[:, -1] = out[:, -2]
        return out

    def _normalize(self, p: np.ndarray) -> np.ndarray:
        mass = np.trapezoid(np.trapezoid(p, self.y, axis=1), self.x)
        if mass <= 0:
            p = np.exp(-0.5 * (self.xx**2 + self.yy**2))
            mass = np.trapezoid(np.trapezoid(p, self.y, axis=1), self.x)
        return p / mass


def ou_drift(theta: float = 1.0) -> Callable[[np.ndarray, float], np.ndarray]:
    return lambda x, _t: -theta * x


def const_diffusion(sigma: float = 1.0) -> Callable[[np.ndarray, float], np.ndarray]:
    return lambda x, _t: sigma * np.ones_like(x)


def gaussian_pdf(x: np.ndarray, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    std = max(std, 1e-9)
    p = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))
    return p / np.trapezoid(p, x)


def monte_carlo_histogram(
    n_samples: int,
    n_steps: int,
    dt: float,
    x0_std: float,
    drift_fn: Callable[[np.ndarray, float], np.ndarray],
    diffusion_fn: Callable[[np.ndarray, float], np.ndarray],
    bins: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, x0_std, size=n_samples)
    for k in range(n_steps):
        t = k * dt
        f = drift_fn(x, t)
        g = diffusion_fn(x, t)
        x = x + f * dt + g * np.sqrt(dt) * rng.normal(size=n_samples)
    hist, edges = np.histogram(x, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist
