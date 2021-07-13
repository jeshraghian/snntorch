==================================================================
Spikevision Datasets: DVS Gesture
==================================================================

DVS Gesture DataLoader

Example::

        import snntorch as snn
        from snntorch.spikevision import spikedata

        from torch.utils.data import DataLoader

        # create datasets
        train_ds = spikedata.DVSGesture("dataset/dvsgesture", train=True)
        test_ds = spikedata.DVSGesture("dataset/dvs_gesture", train=False)

        # create dataloaders   
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=64)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=64)

Visualizing the data::

        import matplotlib.pyplot as plt
        import snntorch.spikeplot as splt
        from IPython.display import HTML

        # choose a random sample
        n = 125

        # index into a single sample and sum the on/off channels
        a = (train_dl.dataset[n][0][:, 0] + train_dl.dataset[n][0][:, 1])

        #  Plot
        fig, ax = plt.subplots()
        anim = splt.animator(a, fig, ax, interval=10)
        HTML(anim.to_html5_video())

        anim.save('dvsgesture_animator.mp4', writer = 'ffmpeg', fps=50) 

        