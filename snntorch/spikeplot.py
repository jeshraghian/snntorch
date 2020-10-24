import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def spike_animator(data, x, y, T=100, interval=40, cmap='plasma'):
    """Makes an animation by looping through the first dimension of a sample of spiking data.

               Parameters
               ----------
               data : torch tensor
                    Sample of spiking data across numerous time steps.
               x : int
                    width of data.
               y : int
                    height of data.
               T : int
                    Number of time steps to display (default: ``100``).
               interval : int, optional
                    Delay between frames in milliseconds (default: ``40``).
               cmap : string, optional
                    color map (default: ``plasma``).

                Returns
                -------
                FuncAnimation
                    Contains animation to be displayed by using plt.show().
               """

    data.cpu()

    fig, ax = plt.subplots()

    ax.set_xlim((0, (x-1)))
    ax.set_ylim((0, (y-1)))
    plt.axis('off')
    plt.gca().invert_yaxis()

    # cmap: plasma has nice UoM colors. grayscale is safe. brg looks like a DVS.
    im = ax.imshow(data[0,:,:], cmap=cmap)

    def init():
        im.set_data(data[0,:,:])
        return (im,)

    # animation function. This is called sequentially
    def animate(i):
        data_slice = data[i,:,:]
        im.set_data(data_slice)
        return (im,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=T, interval=interval, blit=True)

    #HTML(anim.to_html5_video())

    return anim
