==================================================================
Animator
==================================================================

Generate an animation by looping through the first dimension of a sample of spiking data.
Time must be the first dimension of `data` below.

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
