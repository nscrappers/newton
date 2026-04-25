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
            64 segments is a good balance for visual fidelity in most scenes.
        density: Mass density in kg/m^3 (used for mass computation).
    """

    radius: float = 0.5
    height: float = 1.0
    up_axis: int = 1
    # Increased default from 32 to 64 for smoother cone approximation
    segments: int = 64
    density: float = 1000.0

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
    # Inertia about the symmetry axis
    i_axial = (3.0 / 10.0) * mass * radius**2

    # Inertia about a transverse axis through the center of mass
    i_transverse = (3.0 / 20.0) * mass * (radius**2 + (height**2) / 4.0)

    inertia_diag = [i_transverse, i_transverse, i_transverse]
    inertia_diag[up_axis] = i_axial

    return np.diag(inertia_diag)


def add_shape_cone(
    model,
    body: int,
    cfg: Optional[ConeConfig] = None,
    pos: Optional[np.ndarray] = None,
    rot: Optional[np.ndarray] = None,
    *,
    radius: float = 0.5,
    height: float = 1.0,
    up_axis: int = 1,
    density: float = 1000.0,
    segments: int = 64,
) -> int:
    """Add a cone shape to a Newton model body
