import snntorch as snn

from torch.utils.data.dataloader import DataLoader, TensorDataset

# define a configuration file below. Split: train/validation split. Subset: factor by which to reduce train and
# test set.

# define configuration file.
config = snn.Configuration([28, 28], batch_size=100, split=5000, subset=50, num_classes=10, T=500,
                           data_path='/data/mnist')

# download mnist_dataset and automatically split it into train/valid/test sets
ds_train, ds_valid, ds_test = snn.mnist_dataset(config)

# snn.mnist_dataset automatically updates the config file with ds_train and ds_valid.
print("The size of the validation set is: {}".format(config.valid_size))
print("The size of the test set is: {}".format(config.test_size))

# snn.mnist_dataset automatically updates the config file with num_classes if unspecified
# On MNIST, runtime was 10x when num_classes was not specified. 0.4s vs 4s. Not a big deal.
print("\nThe number of classes is: {}".format(config.num_classes))

### Testing sizes and shapes
print("\nLength of ds_train: {}".format(ds_train.__len__()))
print("Testing indexing into ds_train's target: {}".format(ds_train[0][1]))
print("Dimensions of one sample from ds_train: {}".format(ds_train[0][0].size()))

# convert input data into time series. Extract targets as well.
train_time, targets = snn.time_series_data(ds_train, config)
print("\nDimensions of time series training set: {}".format(train_time.shape))

# convert targets into one_hot and time series
st_targets = snn.time_series_targets(targets, config)
print("Dimensions of time series targets: {}".format(st_targets.shape))

# generate poisson spike train for test set
st_train = snn.spiketrain(train_time)
print("Dimensions of spike train: {}".format(st_train.shape))

# now generate validation and test set spike trains
valid_time, valid_targets = snn.time_series_data(ds_valid, config)
test_time, test_targets = snn.time_series_data(ds_test, config)

# convert targets for validation and test sets
st_valid_targets = snn.time_series_targets(valid_targets, config)
st_test_targets = snn.time_series_targets(test_targets, config)

# generate poisson spike trains for validation and test sets
st_valid = snn.spiketrain(valid_time)
st_test = snn.spiketrain(test_time)

# rebuild dataset with spikes and dataloader in batches. shuffle set to False as it was already shuffled at the start.
st_trainset = TensorDataset(st_train, st_targets)
st_trainloader = DataLoader(st_trainset, batch_size=config.batch_size, shuffle=False)

print("Rest in peace RAM.")

# TO-DO: Verify this all worked correctly.
# Then move onto building a LIF neuron class + network to train with.