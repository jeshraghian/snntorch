==================================================================
Spike Count
==================================================================

Generate horizontal bar plot for a single forward pass. Options to animate are also available.

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
        