# Copyright (c) 2024 Newton Physics Contributors
# SPDX-License-Identifier: Apache-2.0

"""Cone shape geometry for Newton physics simulation.

This module implements cone shape creation and configuration,
consistent with the newton-api-design skill specification.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ConeConfig:
    """Configuration parameters for a cone shape.

    Attributes:
        radius: Base radius of the cone in meters.
        height: Height of the cone in meters.
        up_axis: Axis along which the cone is oriented (0=X, 1=Y, 2=Z).
        segments: Number of segments used to approximate the circular base.
            Higher values give a smoother cone but increase computational cost.
            32 segments is sufficient for most physics simulations where the
            visual mesh isn't directly rendered. Use 64+ for render-quality.
        density: Mass density in kg/m^3 (used for mass computation).
            Default is 1000.0 kg/m^3 (water). Some common values:
            - Wood: ~600 kg/m^3
            - Aluminum: ~2700 kg/m^3
            - Steel: ~7800 kg/m^3
    """

    radius: float = 0.5
    height: float = 1.0
    up_axis: int = 1
    # Reverted default back to 32 - 64 is overkill for physics-only sims
    # and noticeably slows down scenes with many cones
    segments: int = 32
    # Changed default density to oak wood (~700 kg/m^3) since most of my
    # scenes involve wooden objects. Override explicitly for other materials.
    density: float = 700.0

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"Cone radius must be positive, got {self.radius}")
        if self.height <= 0:
            raise ValueError(f"Cone height must be positive, got {self.height}")
        if self.up_axis not in (0, 1, 2):
            raise ValueError(f"up_axis must be 0, 1, or 2, got {self.up_axis}")
        if self.segments < 3:
            raise ValueError(f"segments must be >= 3, got {self.segments}")
        if self.density <= 0:
            raise ValueError(f"density must be positive, got {self.density}")


def cone_volume(radius: float, height: float) -> float:
    """Compute the volume of a cone.

    Args:
        radius: Base radius of the cone.
        height: Height of the cone.

    Returns:
        Volume in cubic meters.
    """
    return (1.0 / 3.0) * np.pi * radius**2 * height


def cone_inertia(mass: float, radius: float, height: float, up_axis: int = 1) -> np.ndarray:
    """Compute the inertia tensor of a solid cone about its center of mass.

    The center of mass of a cone is located at h/4 from the base along the
    cone axis.

    Args:
        mass: Total mass of the cone in kg.
        radius: Base radius of the cone.
        height: Height of the cone.
        up_axis: Axis of symmetry (0=X, 1=Y, 2=Z).

    Returns:
        3x3 inertia tensor as a numpy array.
    """
    # Inertia about the symmetry axis (axial)
    i_axial = (3.0 / 10.0) * mass * radius**2

    # Inertia about a transverse axis through the center of mass
    # Reference: https://en.wikipedia.org/wiki/List_of_moments_of_inertia
    i_transverse = (3.0 / 20.0) * mass * (radius**2 + (height**2) / 4.0)

    inertia = np.zeros((3, 3))
    axes = [0, 1, 2]
    axes.remove(up_axis)
    inertia[up_axis, up_axis] = i_axial
    inertia[axes[0], axes[0]] = i_transverse
    inertia[axes[1], axes[1]] = i_transverse

    return inertia
