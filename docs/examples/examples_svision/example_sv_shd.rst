==================================================================
Spikevision Datasets: SHD Dataset
==================================================================

Spiking Heidelberg Digits Dataset DataLoader

Example::

        import snntorch as snn
        from snntorch.spikevision import spikedata

        from torch.utils.data import DataLoader

        # create datasets
        train_ds = spikedata.SHD("dataset/shd", train=True)
        test_ds = spikedata.SHD("dataset/shd", train=False)

        # create dataloaders   
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=64)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=64)

Visualizing the data::

        import matplotlib.pyplot as plt
        import snntorch.spikeplot as splt

        # choose a random sample
        n = 6295

        # initialize figure and axes
        fig = plt.figure(facecolor="w", figsize=(10, 5))
        ax = fig.add_subplot(111)
        
        # use spikeplot to generate a raster
        splt.raster(train_dl.dataset[n][0], ax, s=1.5, c="black")
