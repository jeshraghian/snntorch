import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from celluloid import Camera
from IPython.display import HTML

# import matplotlib.animation as animation
from matplotlib.figure import Figure

# this function is redundant at the moment
# def spike_animator(data, x, y, T=100, interval=40, cmap='plasma'):
#     """Makes an animation by looping through the first dimension of a sample of spiking data.
#
#                Parameters
#                ----------
#                data : torch tensor
#                     Sample of spiking data across numerous time steps.
#                x : int
#                     width of data.
#                y : int
#                     height of data.
#                T : int
#                     Number of time steps to display (default: ``100``).
#                interval : int, optional
#                     Delay between frames in milliseconds (default: ``40``).
#                cmap : string, optional
#                     color map (default: ``plasma``).
#
#                 Returns
#                 -------
#                 FuncAnimation
#                     Contains animation to be displayed by using plt.show().
#                """
#
#     data = data.cpu()
#
#     fig, ax = plt.subplots()
#
#     ax.set_xlim((0, (x-1)))
#     ax.set_ylim((0, (y-1)))
#     plt.axis('off')
#     plt.gca().invert_yaxis()
#
#     # cmap: plasma has nice UoM colors. grayscale is safe. brg looks like a DVS.
#     im = ax.imshow(data[0, :, :], cmap=cmap)
#
#     def init():
#         im.set_data(data[0, :, :])
#         return (im,)
#
#     # animation function. This is called sequentially
#     def animate(i):
#         data_slice = data[i, :, :]
#         im.set_data(data_slice)
#         return (im,)
#
#     # call the animator. blit=True means only re-draw the parts that have changed.
#     anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                    frames=T, interval=interval, blit=True)
#
#     #HTML(anim.to_html5_video())
#
#     return anim


def raster(data, ax, **kwargs):
    """Generate a raster plot using plt.scatter."""
    if len(data.size()) == 1:
        return ax.scatter(*torch.where(data.unsqueeze(1).cpu()), **kwargs)
    return ax.scatter(*torch.where(data.cpu()), **kwargs)


def spike_count(data, fig, ax, labels,  num_steps=False, animate=False, interpolate=1, gridshader=True, interval=25,
                time_step=False):
    """Autogenerate horizontal bar plot for a single forward pass. Options to animate also exist.

                  Parameters
                  ----------
                  data : torch tensor
                       Sample of spiking data across numerous time steps [T x N].
                  fig : matplotlib.figure.Figure
                       Top level container for all plot elements.
                  ax : matplotlib.axes._subplots.AxesSubplot
                       Contains additional figure elements and sets the coordinate system.
                       E.g., fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
                  labels : list
                       List of strings of the names of the output labels.
                       For MNIST, e.g., ['0', '1', '2', ... , '9']
                  num_steps : int, optional
                        Number of time steps to plot. If not specified, the number of entries in the first dimension
                        of ``data`` will automatically be used (default: ``False``).
                  animate : Bool, optional
                        If ``True``, return type matplotlib.animation.ArtistAnimation sequentially scanning across the
                        range of time steps available in ``data``.
                        If ``False``, display plot of the final step once all spikes have been counted
                        (default: ``False``).
                  interpolate : int, optional
                        Can be increased to smooth the animation of the vertical time bar. The value passed is the
                        interpolation factor:
                        e.g., interpolate=1 results in no additional; interpolation.
                        e.g., interpolate=5 results in 4 additional frames for each time step (default : ``1``).
                  gridshader : Bool, optional
                        Applies shading to figure (default : ``True``).
                  interval : int, optional
                       Delay between frames in milliseconds (default: ``25``).
                  time_step : Bool, optional
                       Duration of each time step in seconds.
                       If ``False``, time-axis will be in terms of number of steps. If ``True``, time-axis is scaled
                       by the argument passed. (default : ``False``).

                   Returns
                   -------
                   FuncAnimation (if animate is ``True``)
                       Contains animation to be displayed by using plt.show().
                  """
    if num_steps:
        xrange = num_steps
    else:
        xrange = data.size()[0]

    # set the style of the axes and the text color
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333F4B'
    plt.rcParams['ytick.color'] = '#333F4B'
    plt.rcParams['text.color'] = '#333F4B'

    # white bg
    plt.rcParams['figure.facecolor'] = 'white'

    ax.set_xlim([0, xrange])

    if animate:
        camera = Camera(fig)
        for i in range(xrange * interpolate):
            idx = math.floor(i / interpolate)
            _plt_style(data, labels, ax, idx, time_step)

            if gridshader:
                gs = _GridShader(ax, facecolor="lightgrey", first=False, alpha=0.7)

            # time scanner
            plt.axvline(x=(i / interpolate), color='tab:orange', linewidth=3)

            # animate
            camera.snap()

        anim = camera.animate(interval=interval)
        return anim

    else:
        _plt_style(data, labels, ax, idx=xrange, time_step=time_step)
        if gridshader:
            gs = _GridShader(ax, facecolor="lightgrey", first=False, alpha=0.7)
        # plt.savefig('hist2.png', dpi=300, bbox_inches='tight')


def _plt_style(data, labels, ax, idx, time_step=False):
    """Called by spike_count to modify style of plot."""

    # spike data
    time = pd.Series(data[:idx].sum(dim=0), index=labels)
    df = pd.DataFrame({'time': time})

    # numeric placeholder for the y axis
    my_range = list(range(1, len(df.index) + 1))

    # create horizontal line for each labels that starts at x = 0 with the length represented by the spike count
    plt.hlines(y=my_range, xmin=0, xmax=df['time'], color='#007ACC', alpha=0.5, linewidth=8)

    # create dot for each label
    plt.plot(df['time'], my_range, "o", markersize=8, color='#007ACC', alpha=0.6)

    # set labels
    ax.set_xlabel('Time Step', fontsize=15, fontweight='black', color='#333F4B')
    ax.set_ylabel('Labels', fontsize=15, fontweight='black', color='#333F4B')

    # set axis
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.yticks(my_range, df.index)

    # change the style of the axis spines
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    # set the spines position
    ax.spines['bottom'].set_position(('axes', -0.04))
    ax.spines['left'].set_position(('axes', 0.0))

    if time_step:
        ax.set_xlabel('Time [s]', fontsize=15, fontweight='black', color='#333F4B')
        locs, steps = plt.xticks()
        steps = [float(item) * time_step for item in locs]
        plt.xticks(locs, steps)


class _GridShader():
    """Called by spike_count to apply shade to background of plot/animation."""
    def __init__(self, ax, first=True, **kwargs):
        self.spans = []
        self.sf = first
        self.ax = ax
        self.kw = kwargs
        self.ax.autoscale(False, axis="x")
        self.cid = self.ax.callbacks.connect('xlim_changed', self.shade)
        self.shade()

    def clear(self):
        for span in self.spans:
            try:
                span.remove()
            except:
                pass

    def shade(self, evt=None):
        self.clear()
        xticks = self.ax.get_xticks()
        xlim = self.ax.get_xlim()
        xticks = xticks[(xticks > xlim[0]) & (xticks < xlim[-1])]
        locs = np.concatenate(([[xlim[0]], xticks, [xlim[-1]]]))

        start = locs[1 - int(self.sf)::2]
        end = locs[2 - int(self.sf)::2]

        for s, e in zip(start, end):
            self.spans.append(self.ax.axvspan(s, e, zorder=0, **self.kw))
