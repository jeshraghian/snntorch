import snntorch as snn
import matplotlib.pyplot as plt
import torch

time_step = 1e-3
R = 5
C = 1e-3

num_steps = 500

# lif1 = snn.Lapicque(R=R, C=C, time_step=time_step)
lif1 = snn.Izhikevich()

mem = torch.ones(1) * -70  # U=0.9 at t=0
cur_in = torch.zeros(num_steps)  # I=0 for all t
spk_out = torch.zeros(1)  # initialize output spikes

cur_in[:] = 10

# A list to store a recording of membrane potential
mem_rec = [mem]
spk_rec = [spk_out]
u = torch.ones(1)

# pass updated value of mem and cur_in[step]=0 at every time step
for step in range(num_steps):
    spk_out, mem, u = lif1(cur_in[step], mem)

    if spk_out:
        print(step, mem)
    # Store recordings of membrane potential
    mem_rec.append(mem)
    spk_rec.append(spk_out)

# mem_rec = torch.stack(mem_rec)
# print(mem_rec)

fig = plt.figure()
plt.plot(mem_rec)
plt.plot(cur_in)
plt.show()
plt.close()

fig = plt.figure()
plt.plot(spk_rec)
plt.show()
plt.close()