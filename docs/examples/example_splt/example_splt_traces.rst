==================================================================
Traces
==================================================================

Plot an array of neuron traces (e.g., membrane potential or synaptic current). 
Example::

        import snntorch.spikeplot as splt

        #  mem_rec contains the traces of 9 neuron membrane potentials across 100 time steps in duration
        print(mem_rec.size())
        >>> torch.Size([100, 9])

        #  Plot
        splt.traces(mem_rec, dim=(3,3))