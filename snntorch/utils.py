# class Configuration:
#     """A class containing data about the model and dataset.
#
#     Parameters
#     ----------
#     input_size : list
#         Input data dimensions [W x H]. **Extend this out to [C x W x H] or [T x C x W x H].
#     channels : int
#         Number of channels (default: ``1``).
#     batch_size : int, optional
#         Batch size used for training (default: ``100``).
#     split : int, optional
#         Percentage split of the train set used for the validation set (default: ``0``).
#     subset : int, optional
#         Divide the size of train and test sets by the subset factor (default: ``1``).
#     num_classes : int,optional
#         Number of output classes. Automatically calculated if not provided (default: ``False``).
#     T : int, optional
#         Number of time steps (default: ``500``).
#     data_path : string, optional
#         Root directory of dataset (default : ``\data``).
#     """
#
#     def __init__(self, input_size, channels=1, batch_size=100, split=0, subset=1, num_classes=False, T=500,
#                  data_path='/data'):
#
#         self.input_size = input_size
#         self.channels = channels
#         self.batch_size = batch_size
#         self.split = split
#         self.subset = subset
#         self.num_classes = num_classes
#         self.T = T
#         self.data_path = data_path
#
#     #def _calc_input_size(self, ds_train):
#     #     """Measure input dimension size and update config file."""
#     #    self.input_size = list(np.array(ds_train[0][0]).shape)