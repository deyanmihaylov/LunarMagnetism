import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import root

def geographic_to_Mollweide_point(
    points_geographic: np.ndarray
) -> np.ndarray:
    """
    Transform points on the unit sphere from geographic coordinates (ra,dec)
    to Mollweide projection coordiantes (x,y).
    
    INPUTS
    ------
    points_geographic: numpy array
        The geographic coords (ra,dec).
        Either a single point [shape=(2,)], or
        a list of points [shape=(Npoints,2)].
    
    RETURNS
    -------
    points_Mollweide: numpy array
        The Mollweide projection coords (x,y).
        Either a single point [shape=(2,)], or
        a list of points [shape=(Npoints,2)].
    """
    final_shape_Mollweide = list(points_geographic.shape)

    points_geographic = points_geographic.reshape(-1, points_geographic.shape[-1])
        
    points_Mollweide = np.zeros(shape=points_geographic.shape,
                                dtype=points_geographic.dtype)

    alpha_tol = 1.e-6

    def alpha_eq(x):
        return np.where(np.pi/2 - np.abs(points_geographic[...,1]) < alpha_tol, points_geographic[...,1], 2 * x + np.sin(2 * x) - np.pi * np.sin(points_geographic[...,1]))

    alpha = root(fun=alpha_eq, x0=points_geographic[...,1], method='krylov', tol=1.e-10)

    points_Mollweide[...,0] = 2 * np.sqrt(2) * (points_geographic[...,0] - np.pi) * np.cos(alpha.x) / np.pi
    points_Mollweide[...,1] = np.sqrt(2) * np.sin(alpha.x)

    points_Mollweide = points_Mollweide.reshape(final_shape_Mollweide)

    return points_Mollweide

def geographic_to_Cartesian_vector(points, dpoints):
    """
    Transform vectors in the tangent plane of the unit sphere from
    geographic coords (d_ra,d_dec) to Cartesian coords (d_x,d_y,d_z).
    
    INPUTS
    ------
    points: numpy array
        The geographic coords (r, theta, phi).
        Either a single point [shape=(3,)], or
        a list of points [shape=(Npoints, 3)].
    dpoints: numpy array
        The geographic coords (dr, d_ra, d_dec).
        Either a single point or many with shape 
        matching points.
    
    RETURNS
    -------
    tangent_vector: numpy array
        The coords (d_x,d_y,d_z).
        Either a single point [shape=(3,)], or
        a list of points [shape=(Npoints,3)].
    """
    if points.ndim == 1:
        tangent_vector = np.zeros((3), dtype=dpoints.dtype)
    else:
        tangent_vector = np.zeros((len(points), 3), dtype=dpoints.dtype)
    
    r = points[..., 0]
    theta = np.pi / 2 - points[..., 1]
    phi = points[..., 2]
    
    dr = dpoints[..., 0]
    dtheta = - dpoints[..., 1]
    dphi = dpoints[..., 2]
    
    tangent_vector[..., 0] = (
        dr * np.sin(theta) * np.cos(phi)
        + r * np.cos(theta) * np.cos(phi) * dtheta
        - r * np.sin(theta) * np.sin(phi) * dphi
    )

    tangent_vector[..., 1] = (
        dr * np.sin(theta) * np.sin(phi)
        + r * np.cos(theta) * np.sin(phi) * dtheta
        + r * np.sin(theta) * np.cos(phi) * dphi
    )

    tangent_vector[..., 2] = (
        dr * np.cos(theta) - r * np.sin(theta) * dtheta
    )
    
    return tangent_vector

def deg_to_rad(X):
    return np.deg2rad(X)


if __name__ == "__main__":
    lunar_data = np.genfromtxt("./Moon_Mag_100km.txt", delimiter=' ', skip_header=True)

    latitude = deg_to_rad(lunar_data[:, 0])
    longitude = deg_to_rad(lunar_data[:, 1]) + np.pi

    lunar_coords = geographic_to_Mollweide_point(np.vstack((longitude, latitude)).T)

    lunar_pts = np.vstack((np.ones_like(latitude) * 1837e3, latitude, deg_to_rad(lunar_data[:, 1]))).T

    B_r = lunar_data[:, 4]
    B_theta = lunar_data[:, 3]
    B_phi = lunar_data[:, 2]

    B_vec = np.vstack((B_r, B_theta, B_phi)).T

    B_vec_cart = geographic_to_Cartesian_vector(lunar_pts, B_vec)

    B_field = np.linalg.norm(B_vec_cart, axis=1)
    B_field = (B_field - B_field.min()) / (B_field.max() - B_field.min())

    plt.figure(figsize=(16, 10))

    plt.scatter(lunar_coords[:, 0], lunar_coords[:, 1], edgecolor=None, c=B_field, cmap='viridis')

    plt.axis('equal')
    plt.axis('off')

    plt.savefig("mollweide.png")
