import numpy as np

def spherical_to_cartesian(r, theta_rad, phi_rad):
    """
    Transform a single point from spherical coordinates to Cartesian coordinates.

    Parameters:
    r (float): Radius.
    theta_deg (float): Polar angle in degrees.
    phi_deg (float): Azimuthal angle in degrees.

    Returns:
    tuple: The Cartesian coordinates (x, y, z).
    """

    x = r * np.sin(theta_rad) * np.cos(phi_rad)
    y = r * np.sin(theta_rad) * np.sin(phi_rad)
    z = r * np.cos(theta_rad)

    return (x, y, z)

def spherical_vector_to_cartesian(V_r, V_theta, V_phi, r, theta, phi, degrees=False):
    """
    Transform a vector field from spherical (V_r, V_theta, V_phi) to Cartesian (V_x, V_y, V_z).

    Parameters
    ----------
    V_r : array_like
        Radial component of the vector field.
    V_theta : array_like
        Polar (colatitude) component of the vector field.
    V_phi : array_like
        Azimuthal component of the vector field.
    r : array_like
        Radius coordinate.
    theta : array_like
        Polar angle (colatitude) in radians (or degrees if degrees=True).
    phi : array_like
        Azimuthal angle in radians (or degrees if degrees=True).
    degrees : bool, optional
        If True, input angles are given in degrees.

    Returns
    -------
    V_x, V_y, V_z : ndarray
        Cartesian components of the vector field.
    """
    # Convert to radians if needed

    # Transformation formulas
    V_x = (
        V_r * np.sin(theta) * np.cos(phi)
        + V_theta * np.cos(theta) * np.cos(phi)
        - V_phi * np.sin(phi)
    )
    V_y = (
        V_r * np.sin(theta) * np.sin(phi)
        + V_theta * np.cos(theta) * np.sin(phi)
        + V_phi * np.cos(phi)
    )
    V_z = V_r * np.cos(theta) - V_theta * np.sin(theta)

    return V_x, V_y, V_z

