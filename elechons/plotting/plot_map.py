import matplotlib.pyplot as plt
import matplotlib.animation as animate
from matplotlib.colors import ListedColormap
import numpy as np

def get_transparent_cmap(cmap, levels=256):
    base_cmap = plt.get_cmap(cmap, levels)
    colors = base_cmap(np.linspace(0, 1, 256))

    center = int(levels/2)
    fade_width = int(levels/10)
    for i in range(center - fade_width, center + fade_width + 1):
        colors[i, 3] = abs(i - center) / fade_width
    
    return ListedColormap(colors)

# x and y are 2d coordinates
# v are values
# e are edges from row->col
# v_animate allows v colour to be animated w/ columns as frames
# e_animate allows e colour to be animated w/ 3rd dimension as frames
def plot_data(x, y, v=None, e=None, v_animate=False, e_animate=False):
    N = len(x)

    fig, ax = plt.subplots()

    if e is not None:
        i, j = np.meshgrid(np.arange(N), np.arange(N))
        i = i.flatten()
        j = j.flatten()
        mask = (i != j)
        i = i[mask]
        j = j[mask]

        if not e_animate:
            e_flat = e.flatten()[mask]
            max_e = np.max(np.abs(e_flat))

            q = ax.quiver(x[i], y[i], x[j] - x[i], y[j] - y[i], e_flat, cmap=get_transparent_cmap('plasma'), zorder=0,
              scale=1,
              scale_units='xy',
              angles='xy')
            q.set_clim(vmin=-max_e, vmax=max_e)
        else:
            e_flat_full = e.reshape(-1, e.shape[2])
            e_flat = np.array([e_flat_full[:, i][mask] for i in range(e.shape[2])]).T
            max_e = np.max(np.abs(e_flat))
            
            q = ax.quiver(x[i], y[i], x[j] - x[i], y[j] - y[i], e_flat[:, 0], cmap=get_transparent_cmap('plasma'), zorder=0,
              scale=1,
              scale_units='xy',
              angles='xy')
            q.set_clim(vmin=-max_e, vmax=max_e)

            def e_update(frame):
                q.set_array(e_flat[:, frame])
                return q

        fig.colorbar(q, ax=ax)
        
        if v is None:
            v = np.diagonal(e, axis1=0, axis2=1).T
            v_animate = e_animate
    
    if v is None:
        s = ax.scatter(x, y)
    elif not v_animate:
        s = ax.scatter(x, y, c=v, cmap='plasma', zorder=5)
    else:
        s = ax.scatter(x, y, c=v[:, 0], cmap='plasma', zorder=5)

        def v_update(frame):
            s.set_array(v[:, frame])
            return s
    fig.colorbar(s, ax=ax)
    
    if v_animate and e_animate:
        def update(frame):
            q = e_update(frame)
            s = v_update(frame)
            return q, s
        ani = animate.FuncAnimation(fig, update, frames=N, interval=50, blit=True)
        return fig, ax, ani
    
    if v_animate:
        ani = animate.FuncAnimation(fig, v_update, frames=N, interval=50, blit=True)
        return fig, ax, ani
    
    if e_animate:
        ani = animate.FuncAnimation(fig, e_update, frames=N, interval=50, blit=True)
        return fig, ax, ani
    
    return fig, ax

if __name__=='__main__':
    import elechons.regress_temp as r
    r.init('mean')
    import elechons.models.linear_regression as l
    LES, mu_t, var_t, history = l.linear_exp_SGD(r.temps_mean_sin_adj, transition_learning_coefficient=0.1)
    fig, ax, ani = plot_data(r.long, r.lat, e=history, e_animate=True)
    plt.show()