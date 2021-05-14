==================================================================
Raster
==================================================================

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
