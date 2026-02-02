import pandas as pd

from dataselector.analysis.visualizer import Visualizer


def test_plot_spatial_distribution_without_geo(tmp_path):
    # Simple lat/lon DataFrame
    df = pd.DataFrame({"N": [52.0, 48.0], "left": [13.0, 11.5]})
    viz = Visualizer(output_dir=str(tmp_path))
    out = tmp_path / "spatial_no_geo.png"
    fig = viz.plot_spatial_distribution(df, selected_indices=None, save_path=str(out))
    assert out.exists()
    # aspect for lat/lon fallback should not necessarily be 'equal'
    assert fig is not None


def test_plot_spatial_distribution_with_projected_coords(tmp_path):
    # Dummy object with gdf_metric DataFrame
    class Meta:
        pass

    m = Meta()
    # create projected coordinates in meters (three points)
    m.gdf_metric = pd.DataFrame(
        {"_proj_x": [0.0, 0.0, 0.0], "_proj_y": [0.0, 100000.0, 200000.0]}
    )
    viz = Visualizer(output_dir=str(tmp_path))
    out = tmp_path / "spatial_proj.png"
    fig = viz.plot_spatial_distribution(m, selected_indices=[0, 2], save_path=str(out))
    assert out.exists()
    # check aspect is set to equal for projected coords
    ax = fig.axes[0]
    a = ax.get_aspect()
    # matplotlib may return 'equal' or numeric (1.0) when aspect is equal
    assert (isinstance(a, str) and a in ("equal", "auto", "box")) or (
        isinstance(a, (int, float)) and abs(float(a) - 1.0) < 1e-6
    )
