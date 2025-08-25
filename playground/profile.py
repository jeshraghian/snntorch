import torch
import time
from snntorch._neurons.stateleaky import StateLeaky


timesteps = 2000
batch = 256
channels = 1000

input_ = (
    torch.arange(1, timesteps * batch * channels + 1)
    .float()
    .view(timesteps, batch, channels)
    .to("cuda:1")
)
print(input_.shape)

layer = StateLeaky(beta=0.9, channels=channels)
layer.forward(input_)

input_ = (
    torch.arange(1, timesteps * batch * channels + 1)
    .float()
    .view(timesteps, batch, channels)
    .to("cuda:1")
)

start_time = time.time()
layer.forward(input_)
end_time = time.time()

print(f"{end_time-start_time}")
