from flask import Flask, render_template, jsonify
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Flask(__name__)

# ----------------------------------------------------------------------------
# Helper functions for generating data and plots
# ----------------------------------------------------------------------------

def generate_data(view_type):
    """
    Generate data for the interactive 3D plot based on the selected view type.
    """
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    x, y = np.meshgrid(x, y)

    if view_type == "Prime Gaps":
        z = np.log(np.abs(np.sin(x**2 + y**2) + 1))
    elif view_type == "Zeta Function":
        z = np.abs(np.sin(x) * np.cos(y))
    elif view_type == "Curvature Map":
        z = np.sqrt(x**2 + y**2)
    elif view_type == "Fourier Amplitudes":
        z = np.abs(np.sin(2 * np.pi * x) + np.cos(2 * np.pi * y))
    elif view_type == "GMM Density":
        z = np.exp(-(x**2 + y**2) / 20)
    elif view_type == "Frame Shifts":
        z = np.abs(np.tan(x) + np.tan(y))
    elif view_type == "Mersenne Growth":
        z = np.log(np.abs(np.sin(x) * np.cos(y) + 1))
    elif view_type == "Prime Distribution":
        z = np.abs(np.sin(x * y))
    elif view_type == "Z-Topology":
        z = np.cos(np.sqrt(x**2 + y**2))
    elif view_type == "Relativistic Density":
        z = np.log(np.abs(np.sin(x) + np.cos(y)))
    else:
        z = np.zeros_like(x)

    return x, y, z

def create_3d_plot(view_type):
    """
    Create a 3D plot using Plotly based on the selected view type.
    """
    x, y, z = generate_data(view_type)

    fig = go.Figure()
    fig.add_trace(go.Surface(z=z, x=x, y=y, colorscale="Viridis"))

    fig.update_layout(
        title=f"3D Plot: {view_type}",
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis"
        )
    )

    return fig

# ----------------------------------------------------------------------------
# Routes for the Flask web app
# ----------------------------------------------------------------------------

@app.route('/')
def index():
    """
    Render the main page with the dropdown menu and initial plot.
    """
    views = [
        "Prime Gaps",
        "Zeta Function",
        "Curvature Map",
        "Fourier Amplitudes",
        "GMM Density",
        "Frame Shifts",
        "Mersenne Growth",
        "Prime Distribution",
        "Z-Topology",
        "Relativistic Density"
    ]
    return render_template("index.html", views=views)

@app.route('/plot/<view_type>')
def get_plot(view_type):
    """
    Return the plot as a JSON response for dynamic rendering.
    """
    fig = create_3d_plot(view_type)
    return jsonify(fig.to_dict())

# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)
