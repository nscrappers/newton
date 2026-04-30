# Copyright (c) 2024 Newton Physics Contributors
# SPDX-License-Identifier: Apache-2.0

"""Capsule geometry primitive for Newton physics.

A capsule is a cylinder with hemispherical end caps. It is defined by a radius
and a half-height (the half-length of the cylindrical section, not including the caps).
"""

from dataclasses import dataclass, field
import math


@dataclass
class CapsuleConfig:
    """Configuration for a capsule shape.

    A capsule consists of a cylindrical body of length ``2 * half_height`` capped
    by two hemispheres of the given ``radius``.  The total length along the
    principal axis is therefore ``2 * (half_height + radius)``.

    Args:
        radius: Radius of the cylindrical shaft and the hemispherical end-caps.
        half_height: Half the length of the cylindrical section (excluding caps).
        density: Uniform mass density (kg/m³).  Defaults to 1000 kg/m³ (water).
    """

    radius: float = 0.5
    half_height: float = 0.5
    density: float = 1000.0

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if self.half_height < 0.0:
            raise ValueError(
                f"half_height must be non-negative, got {self.half_height}"
            )
        if self.density <= 0.0:
            raise ValueError(f"density must be positive, got {self.density}")


def capsule_volume(radius: float, half_height: float) -> float:
    """Compute the volume of a capsule.

    V = π r² (4r/3 + 2h)  where h = half_height.

    Args:
        radius: Radius of the capsule.
        half_height: Half-length of the cylindrical section.

    Returns:
        Volume in cubic metres (assuming inputs are in metres).
    """
    sphere_volume = (4.0 / 3.0) * math.pi * radius**3
    cylinder_volume = math.pi * radius**2 * 2.0 * half_height
    return sphere_volume + cylinder_volume


def capsule_mass(radius: float, half_height: float, density: float) -> float:
    """Compute the mass of a capsule.

    Args:
        radius: Radius of the capsule.
        half_height: Half-length of the cylindrical section.
        density: Uniform density (kg/m³).

    Returns:
        Mass in kilograms.
    """
    return density * capsule_volume(radius, half_height)


def capsule_inertia(
    radius: float, half_height: float, density: float
) -> tuple[float, float, float]:
    """Compute the principal moments of inertia for a capsule.

    The capsule is assumed to be aligned along the Y-axis.  Inertia is
    computed about the centre of mass.

    Reference:
        https://www.gamedev.net/tutorials/programming/math-and-physics/capsule-inertia-tensor-r3856/

    Args:
        radius: Radius of the capsule.
        half_height: Half-length of the cylindrical section.
        density: Uniform density (kg/m³).

    Returns:
        Tuple ``(Ixx, Iyy, Izz)`` — principal moments of inertia (kg·m²).
        By symmetry ``Ixx == Izz``.
    """
    r = radius
    h = half_height  # half-height of cylinder
    r2 = r * r
    h2 = h * h

    m_cyl = density * math.pi * r2 * 2.0 * h
    m_hemi = density * (2.0 / 3.0) * math.pi * r2 * r  # mass of one hemisphere

    # --- Cylinder contribution ---
    # Ixx_cyl = Izz_cyl = m(3r² + 4h²) / 12  (full cylinder height = 2h)
    ixx_cyl = m_cyl * (3.0 * r2 + 4.0 * h2) / 12.0
    iyy_cyl = m_cyl * r2 / 2.0

    # --- Hemisphere contribution (two caps) ---
    # Iyy for a single hemisphere about its flat face centre: (2/5) m r²
    # Ixx for a single hemisphere about its flat face centre: (83/320) m r²  (approx)
    # Parallel-axis shift to capsule centre: d = h + 3r/8
    d = h + 3.0 * r / 8.0  # distance from hemisphere CoM to capsule CoM

    iyy_hemi_one = (2.0 / 5.0) * m_hemi * r2
    ixx_hemi_one = (83.0 / 320.0) * m_hemi * r2

    # Two caps, applying parallel-axis theorem to Ixx (axis perpendicular to Y)
    iyy_caps = 2.0 * iyy_hemi_one
    ixx_caps = 2.0 * (ixx_hemi_one + m_hemi * d * d)

    ixx = ixx_cyl + ixx_caps
    iyy = iyy_cyl + iyy_caps
    izz = ixx  # symmetry about Y-axis

    return ixx, iyy, izz
