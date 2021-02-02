import snntorch as snn
import torch


def BPTT(net, data, target, num_steps, batch_size, optimizer, criterion):
    """Backpropagation through time. LIF layers require parameter ``hidden_init = True``
    rather than hidden global variables. BPTT is equivalent to TBPTT for the case where
    num_steps = K.

               Parameters
               ----------
               net : nn.Module
                   Network model.
               data :  torch, tensor
                   Data tensor for a single batch.
               target : torch tensor
                   Target tensor for a single batch.
               num_steps : int
                   Number of time steps.
               batch_size : int
                   Number of samples in a single batch.
               optimizer : torch.optim
                    Optimizer used, e.g., torch.optim.adam.Adam.
               criterion : torch.nn.modules.loss
                    Loss criterion, e.g., torch.nn.modules.loss.CrossEntropyLoss

                Returns
                -------
                torch.Tensor
                    average loss for a single minibatch.
    """
    #  Net requires hidden instance variables rather than global instance variables for TBPTT
    return TBPTT(
        net, data, target, num_steps, batch_size, optimizer, criterion, K=num_steps
    )


def TBPTT(net, data, target, num_steps, batch_size, optimizer, criterion, K=1):
    """Truncated backpropagation through time. LIF layers require parameter ``hidden_init = True``
    rather than hidden global variables.

               Parameters
               ----------
               net : nn.Module
                   Network model.
               data :  torch, tensor
                   Data tensor for a single batch.
               target : torch Tensor
                   Target tensor for a single batch.
               num_steps : int
                   Number of time steps.
               batch_size : int
                   Number of samples in a single batch.
               optimizer : torch.optim
                    Optimizer used, e.g., torch.optim.adam.Adam.
               criterion : torch.nn.modules.loss
                    Loss criterion, e.g., torch.nn.modules.loss.CrossEntropyLoss
               K : int, optional
                    Number of time steps to process per weight update (default: ``num_steps``).

                Returns
                -------
                torch.Tensor
                    average loss for a single minibatch.
    """

    _layer_init(net=net)  # Check which LIF neurons are in net. Reset and detach them.

    t = 0
    loss_trunc = 0  # reset every K time steps
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


def _layer_init(net):
    """Check for the types of LIF neurons contained in net.
    Reset their hidden parameters to zero and detach them
    from the current computation graph."""

    global is_stein
    global is_srm0

    is_stein = False
    is_srm0 = False

    _layer_check(net=net)

    _layer_reset()


def _layer_check(net):
    """Check for the types of LIF neurons contained in net."""

    global is_stein
    global is_srm0

    for idx in range(len(list(net._modules.values()))):
        if isinstance(list(net._modules.values())[idx], snn.Stein):
            is_stein = True
        if isinstance(list(net._modules.values())[idx], snn.SRM0):
            is_srm0 = True


def _layer_reset():
    """Reset hidden parameters to zero and detach them from the current computation graph."""

    if is_stein:
        snn.Stein.zeros_hidden()  # reset hidden state to 0's
        snn.Stein.detach_hidden()
    if is_srm0:
        snn.SRM0.zeros_hidden()  # reset hidden state to 0's
        snn.SRM0.detach_hidden()
