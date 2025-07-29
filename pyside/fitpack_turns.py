import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks


#
#def plot_polyline_and_closest_points(x, y, curvature_threshold=0.0, num_points=1000, s=0, k=3, plot=True):
#    """
#    Fit a B-spline to a polyline, find sharp turn points with curvature above a threshold,
#    identify the closest original input points to those sharp turns, and plot the results.
#
#    Parameters:
#    x, y : array-like, input polyline coordinates (must be same length)
#    curvature_threshold : float, minimum curvature for a point to be considered a sharp turn
#    num_points : int, number of points to evaluate the spline
#    s : float, smoothing factor for splprep (0 for interpolation)
#    k : int, degree of the spline (default 3 for cubic)
#
#    Returns:
#    sharp_turn_t : array, t values of sharp turns on the spline
#    closest_indices : array, indices of original points closest to sharp turns
#    x_closest, y_closest : arrays, coordinates of closest original points
#    """
#    # Ensure x and y are numpy arrays and have the same length
#    x = np.asarray(x)
#    y = np.asarray(y)
#    if len(x) != len(y):
#        raise ValueError("x and y must have the same length")
#
#    # Fit a B-spline to the polyline
#    tck, u = splprep([x, y], s=s, k=k)
#
#    # Evaluate the spline at dense t values
#    t = np.linspace(0, 1, num_points)
#    x_spline, y_spline = splev(t, tck, der=0)  # Spline coordinates
#    dx, dy = splev(t, tck, der=1)  # First derivatives
#    ddx, ddy = splev(t, tck, der=2)  # Second derivatives
#
#    # Compute curvature
#    numerator = np.abs(dx * ddy - dy * ddx)
#    denominator = (dx**2 + dy**2)**1.5
#    curvature = np.where(denominator > 1e-10, numerator / denominator, 0)  # Avoid division by zero
#
#    # Find local maxima in curvature with threshold
#    peaks, _ = find_peaks(curvature, distance=7, height=curvature_threshold)
#    sharp_turn_t = t[peaks]
#    x_sharp, y_sharp = splev(sharp_turn_t, tck, der=0)  # Coordinates of sharp turns
#
#    # Find the closest original points to each sharp turn
#    closest_indices = []
#    x_closest = []
#    y_closest = []
#    for xs, ys in zip(x_sharp, y_sharp):
#        # Compute Euclidean distances from this sharp turn point to all original points
#        distances = np.sqrt((x - xs)**2 + (y - ys)**2)
#        closest_idx = np.argmin(distances)
#        closest_indices.append(closest_idx)
#        x_closest.append(x[closest_idx])
#        y_closest.append(y[closest_idx])
#
#    closest_indices = np.array(closest_indices)
#    x_closest = np.array(x_closest)
#    y_closest = np.array(y_closest)
#
#    if plot:
#        # Plotting
#        plt.figure(figsize=(10, 6))
#        # Plot original polyline
#        plt.plot(x, y, 'bo-', label='Original Polyline', markersize=8, linewidth=1)
#        # Plot fitted B-spline
#        plt.plot(x_spline, y_spline, 'r-', label='Fitted B-spline', linewidth=2)
#        # Plot closest original points to sharp turns
#        plt.plot(x_closest, y_closest, 'g*', label='Closest Points to Sharp Turns', markersize=2)
#    
#        plt.title(f'Polyline, B-spline Fit, and Closest Points to Sharp Turns (Curvature > {curvature_threshold})')
#        plt.xlabel('x')
#        plt.ylabel('y')
#        plt.legend()
#        plt.grid(True)
#        plt.axis('equal')  # Equal scaling on axes for better visualization
#        plt.show()
#
#    #return sharp_turn_t, closest_indices, x_closest, y_closest
#    return sharp_turn_t, closest_indices, x_sharp, y_sharp, x_spline, y_spline

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks

def plot_polyline_and_sharp_turns(x, y, curvature_threshold=0.0, s=0, k=3, plot=True):
    """
    Fit a B-spline to a polyline, find sharp turn points with curvature above a threshold
    by evaluating at the u values from splprep, and plot the results.

    Parameters:
    x, y : array-like, input polyline coordinates (must be same length)
    curvature_threshold : float, minimum curvature for a point to be considered a sharp turn
    s : float, smoothing factor for splprep (0 for interpolation)
    k : int, degree of the spline (default 3 for cubic)

    Returns:
    sharp_turn_u : array, u values of sharp turns on the spline
    sharp_indices : array, indices of original points at sharp turns
    x_sharp, y_sharp : arrays, coordinates of original points at sharp turns
    """
    # Ensure x and y are numpy arrays and have the same length
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    # Fit a B-spline to the polyline
    tck, u = splprep([x, y], s=s, k=k)

    # Evaluate the spline at the u values returned by splprep
    x_spline, y_spline = splev(u, tck, der=0)  # Spline coordinates at u
    dx, dy = splev(u, tck, der=1)  # First derivatives at u
    ddx, ddy = splev(u, tck, der=2)  # Second derivatives at u

    # Compute curvature at u values
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**1.5
    curvature = np.where(denominator > 1e-10, numerator / denominator, 0)  # Avoid division by zero

    # Find local maxima in curvature with threshold
    sharp_indices, _ = find_peaks(curvature, height=curvature_threshold, distance=15)
    sharp_turn_u = u[sharp_indices]
    x_sharp = x[sharp_indices]  # Use original points directly
    y_sharp = y[sharp_indices]

    if plot:
        # Plotting
        plt.figure(figsize=(10, 6))
        # Plot original polyline
        plt.plot(x, y, 'bo-', label='Original Polyline', markersize=8, linewidth=1)
        # Plot fitted B-spline (using a dense t for smooth visualization)
        t_dense = np.linspace(0, 1, 200)  # Dense t for plotting the spline
        x_dense, y_dense = splev(t_dense, tck, der=0)
        plt.plot(x_dense, y_dense, 'r-', label='Fitted B-spline', linewidth=2)
        # Plot original points at sharp turns
        plt.plot(x_sharp, y_sharp, 'g*', label='Sharp Turn Points', markersize=15)
    
        plt.title(f'Polyline, B-spline Fit, and Sharp Turn Points (Curvature > {curvature_threshold})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')  # Equal scaling on axes for better visualization
        plt.show()

    return sharp_turn_u, sharp_indices, x_sharp, y_sharp, x_spline, y_spline

# Example usage
if __name__ == "__main__":
    # Example polyline with sharp turns
    x = [0, 1, 2, 1, 0, -1, -2]
    y = [0, 1, 0, -1, 0, 1, 0]
    t = np.linspace(0, 4*np.pi, 100)
    x= t + 0.5 * np.sin(3*t)
    y= 0.5 * np.cos(2*t) + 0.3 * np.sin(5*t)

    # Call the function with a curvature threshold
    curvature_threshold = 5.0  # Adjust this value based on your data
    u_sharp, sharp_indices, x_sharp, y_sharp = plot_polyline_and_sharp_turns(
        x, y, curvature_threshold=curvature_threshold, s=0)

    print("u values at sharp turns:", u_sharp)
    print("Indices of sharp turn points:", sharp_indices)
    print("Sharp turn points coordinates (x, y):")
    for idx, xu, yu in zip(sharp_indices, x_sharp, y_sharp):
        print(f"Index {idx}: ({xu:.3f}, {yu:.3f})")

# Example usage
if __name__ == "__main__jopa":
    # Example polyline with sharp turns
    x = [0, 1, 2, 1, 0, -1, -2]
    y = [0, 1, 0, -1, 0, 1, 0]
    t = np.linspace(0, 4*np.pi, 100)
    x_test = t + 0.5 * np.sin(3*t)
    y_test = 0.5 * np.cos(2*t) + 0.3 * np.sin(5*t)
    x=x_test
    y=y_test
    
    # Call the function
    t_sharp, x_sharp, y_sharp = plot_polyline_and_sharp_turns(x, y, curvature_threshold=20, s=0)
#    print("t values at sharp turns:", t_sharp)
#   print("Sharp turn coordinates (x, y):")
#    for xt, yt in zip(x_sharp, y_sharp):
#        print(f"({xt:.3f}, {yt:.3f})")
