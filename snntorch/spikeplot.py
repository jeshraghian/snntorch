import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import math
from celluloid import Camera


def animator(data, fig, ax, num_steps=False, interval=40, cmap="plasma"):
    """Generate an animation by looping through the first dimension of a sample of spiking data.
    Time must be the first dimension of ``data``.

    Example::

        import snntorch.spikeplot as splt
        import matplotlib.pyplot as plt

        #  spike_data contains 128 samples, each of 100 time steps in duration
        print(spike_data.size())
        >>> torch.Size([100, 128, 1, 28, 28])

        #  Index into a single sample from a minibatch
        spike_data_sample = spike_data[:, 0, 0]
        print(spike_data_sample.size())
        >>> torch.Size([100, 28, 28])

        #  Plot
        fig, ax = plt.subplots()
        anim = splt.animator(spike_data_sample, fig, ax)
        HTML(anim.to_html5_video())

        #  Save as a gif
        anim.save("spike_mnist.gif")

    :param data: Data tensor for a single sample across time steps of shape [num_steps x input_size]
    :type data: torch.Tensor

    :param fig: Top level container for all plot elements
    :type fig: matplotlib.figure.Figure

    :param ax: Contains additional figure elements and sets the coordinate system. E.g.:
        fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    :type ax: matplotlib.axes._subplots.AxesSubplot

    :param num_steps: Number of time steps to plot. If not specified, the number of entries in the first dimension
        of ``data`` will automatically be used, defaults to ``False``
    :type num_steps: int, optional

    :param interval: Delay between frames in milliseconds, defaults to ``40``
    :type interval: int, optional

    :param cmap: color map, defaults to ``plasma``
    :type cmap: string, optional

    :return: animation to be displayed using ``matplotlib.pyplot.show()``
    :rtype: FuncAnimation

    """

    if not num_steps:
        num_steps = data.size()[0]

    data = data.cpu()
    camera = Camera(fig)
    plt.axis("off")

    # iterate over time and take a snapshot with celluloid
    for step in range(num_steps):  # im appears unused but is required by camera.snap()
        im = ax.imshow(data[step], cmap=cmap)  # noqa: F841
        camera.snap()
    anim = camera.animate(interval=interval)

    return anim


def raster(data, ax, **kwargs):
    """Generate a raster plot using ``matplotlib.pyplot.scatter``.

    Example::

        import snntorch.spikeplot as splt
        import matplotlib.pyplot as plt

        #  spike_data contains 128 samples, each of 100 time steps in duration
        print(spike_data.size())
        >>> torch.Size([100, 128, 1, 28, 28])

        #  Index into a single sample from a minibatch
        spike_data_sample = spike_data[:, 0, 0]
        print(spike_data_sample.size())
        >>> torch.Size([100, 28, 28])

        fig = plt.figure(facecolor="w", figsize=(10, 5))
        ax = fig.add_subplot(111)

        #  s: size of scatter points; c: color of scatter points
        splt.raster(spike_data_sample, ax, s=1.5, c="black")
        plt.title("Input Layer")
        plt.xlabel("Time step")
        plt.ylabel("Neuron Number")
        plt.show()

    """
    if len(data.size()) == 1:
        return ax.scatter(*torch.where(data.unsqueeze(1).cpu()), **kwargs)
    return ax.scatter(*torch.where(data.cpu()), **kwargs)


def spike_count(
    data,
    fig,
    ax,
    labels,
    num_steps=False,
    animate=False,
    interpolate=1,
    gridshader=True,
    interval=25,
    time_step=False,
):
    """Generate horizontal bar plot for a single forward pass. Options to animate also available.

    Example::

        import snntorch.spikeplot as splt
        import matplotlib.pyplot as plt
        from IPython.display import HTML

        num_steps = 25

        #  Use splt.spike_count to display behavior of output neurons for a single sample during feedforward

        #  spk_rec is a recording of output spikes across 25 time steps, using ``batch_size=128``
        print(spk_rec.size())
        >>> torch.Size([25, 128, 10])

        #  We only need a single data sample
        spk_results = torch.stack(spk_rec, dim=0)[:, 0, :].to('cpu')
        print(spk_results.size())
        >>> torch.Size([25, 10])

        fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
        labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']

        #  Plot and save spike count histogram
        splt.spike_count(spk_results, fig, ax, labels, num_steps = num_steps, time_step=1e-3)
        plt.show()
        plt.savefig('hist2.png', dpi=300, bbox_inches='tight')

        # Animate and save spike count histogram
        anim = splt.spike_count(spk_results, fig, ax, labels, animate=True, interpolate=5, num_steps = num_steps, time_step=1e-3)
        HTML(anim.to_html5_video())
        anim.save("spike_bar.gif")

    :param data: Sample of spiking data across numerous time steps [num_steps x num_outputs]
    :type data: torch.Tensor

    :param fig: Top level container for all plot elements
    :type fig: matplotlib.figure.Figures

    :param ax: Contains additional figure elements and sets the coordinate system.
        E.g., fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    :type ax: matplotlib.axes._subplots.AxesSubplot

    :param labels: List of strings of the names of the output labels.
         E.g., for MNIST, ``labels = ['0', '1', '2', ... , '9']``
    :type labels: list

    :param num_steps: Number of time steps to plot. If not specified, the number of entries in the first dimension
        of ``data`` will automatically be used, defaults to ``False``
    :type num_steps: int, optional

    :param animate: If ``True``, return type matplotlib.animation.ArtistAnimation sequentially scanning across the
        range of time steps available in ``data``.
        If ``False``, display plot of the final step once all spikes have been counted, defaults to ``False``
    :type animate: Bool, optional

    :param interpolate: Can be increased to smooth the animation of the vertical time bar. The value passed is the
        interpolation factor:
        e.g., ``interpolate=1`` results in no additional interpolation.
        e.g., ``interpolate=5`` results in 4 additional frames for each time step, defaults to ``1``
    :type interpolate: int, optional

    :param gridshader: Applies shading to figure background to distinguish output classes, defaults to ``True``
    :type gridshader: Bool, optional

    :param interval: Delay between frames in milliseconds, defaults to ``25``
    :type interval: int, optional

    :param time_step: Duration of each time step in seconds.
        If ``False``, time-axis will be in terms of ``num_steps``. Else, time-axis is scaled
        by the argument passed, defaults to ``False``
    :type time_step: int, optional

    :return: animation to be displayed using ``matplotlib.pyplot.show()``
    :rtype: FuncAnimation (if animate is ``True``)

    """
    if num_steps:
        xrange = num_steps
    else:
        xrange = data.size()[0]

    # set the style of the axes and the text color
    plt.rcParams["axes.edgecolor"] = "#333F4B"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.color"] = "#333F4B"
    plt.rcParams["ytick.color"] = "#333F4B"
    plt.rcParams["text.color"] = "#333F4B"

    # white bg
    plt.rcParams["figure.facecolor"] = "white"

    ax.set_xlim([0, xrange])

    if animate:
        camera = Camera(fig)
        for i in range(xrange * interpolate):
            idx = math.floor(i / interpolate)
            _plt_style(data, labels, ax, idx, time_step)

            if gridshader:  # gs appears unused but is needed by plt
                gs = _GridShader(  # noqa: F841
                    ax, facecolor="lightgrey", first=False, alpha=0.7
                )

            # time scanner
            plt.axvline(x=(i / interpolate), color="tab:orange", linewidth=3)

            # animate
            camera.snap()

        anim = camera.animate(interval=interval)
        return anim

    else:
        _plt_style(data, labels, ax, idx=xrange, time_step=time_step)
        if gridshader:
            gs = _GridShader(  # noqa: F841
                ax, facecolor="lightgrey", first=False, alpha=0.7
            )
        # plt.savefig('hist2.png', dpi=300, bbox_inches='tight')


def traces(data, spk=None, dim=(3, 3), spk_height=5, titles=None):
    """Plot an array of neuron traces (e.g., membrane potential or synaptic current).
    Optionally apply spikes to ride on the traces.
    `traces` was originally written by Friedemann Zenke.

    Example::

        import snntorch.spikeplot as splt

        #  mem_rec contains the traces of 9 neuron membrane potentials across 100 time steps in duration
        print(mem_rec.size())
        >>> torch.Size([100, 9])

        #  Plot
        traces(mem_rec, dim=(3,3))


    :param data: Data tensor for neuron traces across time steps of shape [num_steps x num_neurons]
    :type data: torch.Tensor

    :param spk: Data tensor for neuron traces across time steps of shape [num_steps x num_neurons], defaults to ``None``
    :type spk: torch.Tensor, optional

    :param dim: Dimensions of figure, defaults to ``(3, 3)``
    :type dim: tuple, optional

    :param spk_height: height of spike to plot, defaults to ``5``
    :type spk_height: float, optional

    :param titles: Adds subplot titles, defaults to ``None``
    :type titles: list of strings, optional

    """

    gs = GridSpec(*dim)

    if spk is not None:
        data = (data + spk_height * spk).detach().cpu().numpy()

    else:
        data = data.detach().cpu().numpy()

    for i in range(np.prod(dim)):
        if i == 0:
            a0 = ax = plt.subplot(gs[i])
            if titles is not None:
                ax.set_title(titles[i])

        else:
            ax = plt.subplot(gs[i], sharey=a0)
            if titles is not None and i < len(titles):
                ax.set_title(titles[i])

        ax.plot(data[:, i])
        ax.axis("off")


def _plt_style(data, labels, ax, idx, time_step=False):
    """Called by spike_count to modify style of plot."""

    # spike data
    time = pd.Series(data[:idx].sum(dim=0), index=labels)
    df = pd.DataFrame({"time": time})

    # numeric placeholder for the y axis
    my_range = list(range(1, len(df.index) + 1))

    # create horizontal line for each labels that starts at x = 0 with the length represented by the spike count
    plt.hlines(
        y=my_range, xmin=0, xmax=df["time"], color="#007ACC", alpha=0.5, linewidth=8
    )

    # create dot for each label
    plt.plot(df["time"], my_range, "o", markersize=8, color="#007ACC", alpha=0.6)

    # set labels
    ax.set_xlabel("Time Step", fontsize=15, fontweight="black", color="#333F4B")
    ax.set_ylabel("Labels", fontsize=15, fontweight="black", color="#333F4B")

    # set axis
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.yticks(my_range, df.index)

    # change the style of the axis spines
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")

    # set the spines position
    ax.spines["bottom"].set_position(("axes", -0.04))
    ax.spines["left"].set_position(("axes", 0.0))

    if time_step:
        ax.set_xlabel("Time [s]", fontsize=15, fontweight="black", color="#333F4B")
        locs, steps = plt.xticks()
        steps = [float(item) * time_step for item in locs]
        plt.xticks(locs, steps)


class _GridShader:
    """Called by spike_count to apply shade to background of plot/animation."""

    def __init__(self, ax, first=True, **kwargs):
        self.spans = []
        self.sf = first
        self.ax = ax
        self.kw = kwargs
        self.ax.autoscale(False, axis="x")
        self.cid = self.ax.callbacks.connect("xlim_changed", self.shade)
        self.shade()

    def clear(self):
        for span in self.spans:
            try:
                span.remove()
            except Exception:
                pass

    def shade(self, evt=None):
        self.clear()
        xticks = self.ax.get_xticks()
        xlim = self.ax.get_xlim()
        xticks = xticks[(xticks > xlim[0]) & (xticks < xlim[-1])]
        locs = np.concatenate(([[xlim[0]], xticks, [xlim[-1]]]))

        start = locs[1 - int(self.sf) :: 2]
        end = locs[2 - int(self.sf) :: 2]

        for s, e in zip(start, end):
            self.spans.append(self.ax.axvspan(s, e, zorder=0, **self.kw))
