import snntorch as snn
import torch


def TBPTT(
    net,
    data,
    target,
    num_steps,
    batch_size,
    optimizer,
    criterion,
    time_varying_data=True,
    return_spk=False,
    return_mem=False,
    K=1,
):
    """Truncated backpropagation through time. LIF layers require parameter ``hidden_init = True``.
    Forward and backward passes are performed at every time step. Weight updates are performed every ``K`` time steps.

    :param net: Network model
    :type net: nn.Module

    :param data: Data tensor for a single batch
    :type data: torch.Tensor

    :param target: Target tensor for a single batch
    :type target: torch.Tensor

    :param num_steps: Number of time steps
    :type num_steps: int

    :param batch_size: Number of samples in a single batch
    :type batch_size: int

    :param optimizer: Optimizer used, e.g., torch.optim.adam.Adam
    :type optimizer: torch.optim

    :param criterion: Loss criterion, e.g., torch.nn.modules.loss.CrossEntropyLoss
    :type criterion: torch.nn.modules.loss

    :param time_varying_data: True if input data is time-varying [T x B x dims], defaults to ``True``
    :type time_varying_data: Bool, optional

    :param return_spk: Option to return output spikes, defaults to ``False``
    :type return_spk: Bool, optional

    :param return_mem: Option to return output membrane potential traces, defaults to ``False``
    :type return_mem: Bool, optional

    :param K: Number of time steps to process per weight update, defaults to ``1``
    :type K: int, optional

    :return: average loss for a single minibatch
    :rtype: torch.Tensor

    :return: optionally return output spikes over time
    :rtype: torch.Tensor

    :return: optionally return output membrane potential trace over time
    :rtype: torch.Tensor

    """

    if K > num_steps:
        raise ValueError("K must be less than or equal to num_steps.")

    _layer_init(net=net)  # Check which LIF neurons are in net. Reset and detach them.

    t = 0
    loss_trunc = 0  # reset every K time steps
    loss_avg = 0

    if return_spk:
        spk_rec = []

    if return_mem:
        mem_rec = []

    for step in range(num_steps):
        if time_varying_data:
            spk_out, mem_out = net(data[step].view(batch_size, -1))
        else:
            spk_out, mem_out = net(data.view(batch_size, -1))

        if return_spk:
            spk_rec.append(spk_out)

        if return_mem:
            mem_rec.append(mem_out)

        loss = criterion(mem_out, target)
        loss_trunc += loss
        loss_avg += loss
        t += 1
        if t == K:
            optimizer.zero_grad()
            loss_trunc.backward()
            optimizer.step()
            if is_lapicque:
                snn.Lapicque.detach_hidden()
            if is_leaky:
                snn.Leaky.detach_hidden()
            if is_synaptic:
                snn.Synaptic.detach_hidden()
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
        if is_lapicque:
            snn.Lapicque.detach_hidden()
        if is_leaky:
            snn.Leaky.detach_hidden()
        if is_synaptic:
            snn.Synaptic.detach_hidden()
        if is_stein:
            snn.Stein.detach_hidden()
        if is_srm0:
            snn.SRM0.detach_hidden()

    if return_spk:
        if return_mem:
            return loss_avg, torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
        else:
            return loss_avg, torch.stack(spk_rec, dim=0)

    if return_mem:
        return loss_avg, torch.stack(mem_rec, dim=0)

    else:
        return loss_avg


def BPTT(
    net,
    data,
    target,
    num_steps,
    batch_size,
    optimizer,
    criterion,
    time_varying_data=True,
    return_spk=False,
    return_mem=False,
):
    """Backpropagation through time. LIF layers require parameter ``hidden_init = True``.
    A forward pass is applied for each time step while the loss accumulates. The backward pass and parameter update is only applied at the end of each time step sequence.
    BPTT is equivalent to TBPTT for the case where ``num_steps = K``.

    :param net: Network model
    :type net: nn.Module

    :param data: Data tensor for a single batch
    :type data: torch.Tensor

    :param target: Target tensor for a single batch
    :type target: torch.Tensor

    :param num_steps: Number of time steps
    :type num_steps: int

    :param batch_size: Number of samples in a single batch
    :type batch_size: int

    :param optimizer: Optimizer used, e.g., torch.optim.adam.Adam
    :type optimizer: torch.optim

    :param criterion: Loss criterion, e.g., torch.nn.modules.loss.CrossEntropyLoss
    :type criterion: torch.nn.modules.loss

    :param time_varying_data: True if input data is time-varying [T x B x dims], defaults to ``True``
    :type time_varying_data: Bool, optional

    :param return_spk: Option to return output spikes, defaults to ``False``
    :type return_spk: Bool, optional

    :param return_mem: Option to return output membrane potential traces, defaults to ``False``
    :type return_mem: Bool, optional

    :return: average loss for a single minibatch
    :rtype: torch.Tensor

    :return: optionally return output spikes over time
    :rtype: torch.Tensor

    :return: optionally return output membrane potential trace over time
    :rtype: torch.Tensor
    """
    #  Net requires hidden instance variables rather than global instance variables for TBPTT
    return TBPTT(
        net,
        data,
        target,
        num_steps,
        batch_size,
        optimizer,
        criterion,
        time_varying_data,
        return_spk,
        return_mem,
        K=num_steps,
    )


def RTRL(
    net,
    data,
    target,
    num_steps,
    batch_size,
    optimizer,
    criterion,
    time_varying_data=True,
    return_spk=False,
    return_mem=False,
):
    """Real-time Recurrent Learning. LIF layers require parameter ``hidden_init = True``.
    A forward pass, backward pass and parameter update are applied at each time step.
    RTRL is equivalent to TBPTT for the case where ``K = 1``.

    :param net: Network model
    :type net: nn.Module

    :param data: Data tensor for a single batch
    :type data: torch.Tensor

    :param target: Target tensor for a single batch
    :type target: torch.Tensor

    :param num_steps: Number of time steps
    :type num_steps: int

    :param batch_size: Number of samples in a single batch
    :type batch_size: int

    :param optimizer: Optimizer used, e.g., torch.optim.adam.Adam
    :type optimizer: torch.optim

    :param criterion: Loss criterion, e.g., torch.nn.modules.loss.CrossEntropyLoss
    :type criterion: torch.nn.modules.loss

    :param time_varying_data: True if input data is time-varying [T x B x dims], defaults to ``True``
    :type time_varying_data: Bool, optional

    :param return_spk: Option to return output spikes, defaults to ``False``
    :type return_spk: Bool, optional

    :param return_mem: Option to return output membrane potential traces, defaults to ``False``
    :type return_mem: Bool, optional

    :return: average loss for a single minibatch
    :rtype: torch.Tensor

    :return: optionally return output spikes over time
    :rtype: torch.Tensor

    :return: optionally return output membrane potential trace over time
    :rtype: torch.Tensor

    """
    #  Net requires hidden instance variables rather than global instance variables for TBPTT
    return TBPTT(
        net,
        data,
        target,
        num_steps,
        batch_size,
        optimizer,
        criterion,
        time_varying_data,
        return_spk,
        return_mem,
        K=1,
    )


def _layer_init(net):
    """Check for the types of LIF neurons contained in net.
    Reset their hidden parameters to zero and detach them
    from the current computation graph."""

    global is_stein
    global is_srm0
    global is_leaky
    global is_lapicque
    global is_synaptic

    is_stein = False
    is_srm0 = False
    is_leaky = False
    is_synaptic = False
    is_lapicque = False

    _layer_check(net=net)

    _layer_reset()


def _layer_check(net):
    """Check for the types of LIF neurons contained in net."""

    global is_leaky
    global is_lapicque
    global is_synaptic
    global is_stein
    global is_srm0

    for idx in range(len(list(net._modules.values()))):
        if isinstance(list(net._modules.values())[idx], snn.Lapicque):
            is_lapicque = True
        if isinstance(list(net._modules.values())[idx], snn.Synaptic):
            is_synaptic = True
        if isinstance(list(net._modules.values())[idx], snn.Leaky):
            is_leaky = True
        if isinstance(list(net._modules.values())[idx], snn.Stein):
            is_stein = True
        if isinstance(list(net._modules.values())[idx], snn.SRM0):
            is_srm0 = True


def _layer_reset():
    """Reset hidden parameters to zero and detach them from the current computation graph."""

    if is_lapicque:
        snn.Lapicque.zeros_hidden()  # reset hidden state to 0's
        snn.Lapicque.detach_hidden()
    if is_synaptic:
        snn.Synaptic.zeros_hidden()  # reset hidden state to 0's
        snn.Synaptic.detach_hidden()
    if is_leaky:
        snn.Leaky.zeros_hidden()  # reset hidden state to 0's
        snn.Leaky.detach_hidden()
    if is_stein:
        snn.Stein.zeros_hidden()  # reset hidden state to 0's
        snn.Stein.detach_hidden()
    if is_srm0:
        snn.SRM0.zeros_hidden()  # reset hidden state to 0's
        snn.SRM0.detach_hidden()
