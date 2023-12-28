import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
def generate_random_longitude_lines_around_sphere(radius, num_lines, num_points_per_line):
    """
    Generate random lines around a sphere, similar to longitude lines but with random start and end points.

    :param radius: Radius of the sphere
    :param num_lines: Number of lines to generate
    :param num_points_per_line: Number of points in each line
    :return: Arrays of x, y, z coordinates for each line
    """
    all_x, all_y, all_z = [], [], []
    last_end_point = None

    for i in range(num_lines):
        # Random start and end z (latitude)
        start_z = np.random.uniform(-radius, radius) if last_end_point is None else last_end_point
        end_z = np.random.uniform(-radius, radius)

        # Generate points along the line
        z = np.linspace(start_z, end_z, num_points_per_line)
        r = np.sqrt(radius**2 - z**2)
        theta = np.random.uniform(0, 2 * np.pi)  # Random longitude
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        all_x.append(x)
        all_y.append(y)
        all_z.append(z)

        last_end_point = end_z

    return all_x, all_y, all_z

# Generate random longitude-like lines
random_lines_x, random_lines_y, random_lines_z = generate_random_longitude_lines_around_sphere(radius, num_lines, num_points)

# Plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each random line
for i in range(num_lines):
    ax.plot(random_lines_x[i], random_lines_y[i], random_lines_z[i], color='b')

ax.set_xlim([-radius, radius])
ax.set_ylim([-radius, radius])
ax.set_zlim([-radius, radius])
ax.set_title("Random Lines around a Sphere")
plt.show()

def generate_random_direction_lines_around_sphere(radius, num_lines, num_points_per_line, line_width):
    """
    Generate lines around a sphere with random direction changes and considering line width.

    :param radius: Radius of the sphere
    :param num_lines: Number of lines to generate
    :param num_points_per_line: Number of points in each line
    :param line_width: Width of each line (to simulate the yarn thickness)
    :return: Arrays of x, y, z coordinates for each line
    """
    all_x, all_y, all_z = [], [], []
    last_end_point = [0, 0, -radius]  # Starting from the bottom of the sphere

    for _ in range(num_lines):
        points_x, points_y, points_z = [], [], []

        # Start from the last end point
        start_x, start_y, start_z = last_end_point

        for _ in range(num_points_per_line):
            # Generate a random direction change
            delta_theta = np.random.uniform(-np.pi, np.pi) * line_width / radius
            delta_phi = np.random.uniform(-np.pi, np.pi) * line_width / radius

            # Convert to Cartesian coordinates
            r = np.sqrt(start_x**2 + start_y**2 + start_z**2)
            theta = np.arctan2(start_y, start_x) + delta_theta
            phi = np.arccos(start_z / r) + delta_phi

            # Update point
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            # Append to points
            points_x.append(x)
            points_y.append(y)
            points_z.append(z)

            # Update start point for next iteration
            start_x, start_y, start_z = x, y, z

        all_x.append(points_x)
        all_y.append(points_y)
        all_z.append(points_z)

        # Update last end point
        last_end_point = [points_x[-1], points_y[-1], points_z[-1]]

    return all_x, all_y, all_z

# Parameters for the yarn-like lines
num_lines = 100  # More lines for a more detailed simulation
line_width = 0.1  # Simulate the thickness of the yarn

# Generate lines with random direction changes
directional_lines_x, directional_lines_y, directional_lines_z = generate_random_direction_lines_around_sphere(
    radius, num_lines, num_points, line_width)

# Plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each line with directional changes
for i in range(num_lines):
    ax.plot(directional_lines_x[i], directional_lines_y[i], directional_lines_z[i], color='b')

ax.set_xlim([-radius, radius])
ax.set_ylim([-radius, radius])
ax.set_zlim([-radius, radius])
ax.set_title("Random Directional Lines around a Sphere")
plt.show()


def plot_with_plotly(x_lines, y_lines, z_lines, title):
    """
    Create a 3D plot using plotly with multiple lines.

    :param x_lines: List of x coordinates for each line
    :param y_lines: List of y coordinates for each line
    :param z_lines: List of z coordinates for each line
    :param title: Title of the plot
    """
    fig = go.Figure()

    # Add each line to the plot
    for x, y, z in zip(x_lines, y_lines, z_lines):
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines'))

    # Update layout
    fig.update_layout(title=title, scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        xaxis=dict(range=[-radius, radius]),
        yaxis=dict(range=[-radius, radius]),
        zaxis=dict(range=[-radius, radius])))
    
    fig.show()

# Plot using plotly
plot_with_plotly(directional_lines_x, directional_lines_y, directional_lines_z, "Random Directional Lines around a Sphere with Plotly")
