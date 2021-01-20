import snntorch as snn
import torch


def BPTT(net, data, target, num_steps, batch_size, optimizer, criterion):
    #  Net requires hidden instance variables rather than global instance variables for TBPTT
    return TBPTT(
        net, data, target, num_steps, batch_size, optimizer, criterion, K=num_steps
    )


def TBPTT(net, data, target, num_steps, batch_size, optimizer, criterion, K=1):
    #  Net requires hidden instance variables rather than global instance variables for TBPTT
    is_stein = False
    is_srm0 = False

    for idx in range(len(list(net._modules.values()))):
        if isinstance(list(net._modules.values())[idx], snn.Stein):
            is_stein = True
        if isinstance(list(net._modules.values())[idx], snn.SRM0):
            is_srm0 = True

    if is_stein:
        snn.Stein.zeros_hidden()  # reset hidden state to 0's
        snn.Stein.detach_hidden()
    if is_srm0:
        snn.SRM0.zeros_hidden()  # reset hidden state to 0's
        snn.SRM0.detach_hidden()

    t = 0
    loss_trunc = 0
    loss_avg = 0
    for step in range(num_steps):
        spk_out, mem_out = net(data.view(batch_size, -1))
        loss = criterion(mem_out, target)
        loss_trunc += loss
        loss_avg += loss
        t += 1
        if t == K:
            optimizer.zero_grad()
            loss_trunc.backward()
            optimizer.step()
            if is_stein:
                snn.Stein.detach_hidden()
            if is_srm0:
                snn.SRM0.detach_hidden()
            t = 0
            loss_trunc = 0
    if (step == num_steps - 1) and (num_steps % K):
        optimizer.zero_grad()
        loss_trunc.backward()
        optimizer.step()
        if is_stein:
            snn.Stein.detach_hidden()
        if is_srm0:
            snn.SRM0.detach_hidden()

    return loss_avg
