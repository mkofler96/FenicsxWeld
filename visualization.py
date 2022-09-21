import pyvista
import time
from dolfinx import plot

def plot_timeseries(V, timesteps, uh_t):
    u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_plotter = pyvista.Plotter()
    current_mesh = u_plotter.add_mesh(u_grid, show_edges=True)
    u_plotter.view_xy()
    u_plotter.show(interactive_update=True)
    i = 0
    for currnt_uh, dt in zip(uh_t, timesteps):
        i = i+1
        u_grid.point_data["u"] = currnt_uh
        u_grid.set_active_scalars("u")
        #u_plotter.remove_actor(current_mesh)
        current_mesh = u_plotter.add_mesh(u_grid, show_edges=True)
        #u_plotter.update_scalar_bar_range([0,8])
        time.sleep(dt)
        # Close movie and delete object
    u_plotter.close()




