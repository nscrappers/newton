# Copyright (c) 2024 Newton Physics Contributors
# SPDX-License-Identifier: Apache-2.0

"""Cylinder geometry primitive for Newton physics.

Provides configuration, volume, and inertia tensor calculations
for cylinder shapes, consistent with the cone primitive module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CylinderConfig:
    """Configuration for a cylinder shape.

    Attributes:
        radius: Radius of the cylinder (meters). Must be positive.
        height: Height of the cylinder along its axis (meters). Must be positive.
        density: Mass density in kg/m³. Defaults to 1000 (water-like).
        axis: Primary axis of the cylinder. One of 'x', 'y', 'z'. Defaults to 'y'.
    """

    radius: float = 0.5
    height: float = 1.0
    density: float = 1000.0
    axis: str = "y"

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"Cylinder radius must be positive, got {self.radius}")
        if self.height <= 0.0:
            raise ValueError(f"Cylinder height must be positive, got {self.height}")
        if self.density <= 0.0:
            raise ValueError(f"Cylinder density must be positive, got {self.density}")
        if self.axis not in ("x", "y", "z"):
            raise ValueError(f"Cylinder axis must be 'x', 'y', or 'z', got '{self.axis}'")


def cylinder_volume(config: CylinderConfig) -> float:
    """Compute the volume of a cylinder.

    Args:
        config: Cylinder configuration.

    Returns:
        Volume in cubic meters: π * r² * h
    """
    return math.pi * config.radius ** 2 * config.height


def cylinder_mass(config: CylinderConfig) -> float:
    """Compute the mass of a solid cylinder.

    Args:
        config: Cylinder configuration.

    Returns:
        Mass in kilograms.
    """
    return config.density * cylinder_volume(config)


def cylinder_inertia(config: CylinderConfig) -> Tuple[float, float, float]:
    """Compute the principal moments of inertia for a solid cylinder.

    The moments are computed about axes through the center of mass.
    For a cylinder aligned along the *axis* direction:

        I_axial   = (1/2) * m * r²
        I_lateral = (1/12) * m * (3r² + h²)

    Args:
        config: Cylinder configuration.

    Returns:
        Tuple (Ixx, Iyy, Izz) in kg·m². The axial component is placed
        on the configured axis; the other two components are lateral.
    """
    m = cylinder_mass(config)
    r = config.radius
    h = config.height

    i_axial = 0.5 * m * r ** 2
    i_lateral = (1.0 / 12.0) * m * (3.0 * r ** 2 + h ** 2)

    if config.axis == "x":
        return (i_axial, i_lateral, i_lateral)
    elif config.axis == "y":
        return (i_lateral, i_axial, i_lateral)
    else:  # z
        return (i_lateral, i_lateral, i_axial)
