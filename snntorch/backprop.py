import snntorch as snn
import torch
from snntorch import utils
from snntorch import functional as SF


# consider turning into a class s.t. dictionary params can be parsed at __init__
# and never touched again
def TBPTT(
    net,
    dataloader,
    num_steps,  # must be specified in case data in is static
    optimizer,
    criterion,
    time_var,  # add to doc_strings - specifies if data is time_varying
    regularization=False,  # add to doc_strings
    device="cpu",  # add to docstrings
    K=1,
):
    """Truncated backpropagation through time. LIF layers require parameter ``init_hidden = True``.
    Forward and backward passes are performed at every time step. Weight updates are performed every ``K`` time steps.

    :param net: Network model
    :type net: nn.Module

    :param dataloader: DataLoader containing data and targets
    :type dataloader: torch.DataLoader

    :param num_steps: Number of time steps
    :type num_steps: int

    :param optimizer: Optimizer used, e.g., torch.optim.adam.Adam
    :type optimizer: torch.optim

    :param criterion: Loss criterion from snntorch.functional, e.g., snn.functional.mse_count_loss()
    :type criterion: snn.functional.LossFunctions

    :param time_var: Set to ``True`` if input data is time-varying [T x B x dims]. Otherwise, set to false if input data is time-static [B x dims].
    :type time_var: Bool

    :param regularization: Option to add a regularization term to the loss function
    :type regularization: snn.functional regularization function, optional

    :param device: Specify either "cuda" or "cpu", defaults to "cpu"
    :type device: string, optional

    :param K: Number of time steps to process per weight update, defaults to ``1``
    :type K: int, optional

    :return: return average loss for one epoch
    :rtype: torch.Tensor

    :return: return output spikes over time
    :rtype: torch.Tensor

    :return: return output membrane potential trace over time
    :rtype: torch.Tensor
    """

    if K > num_steps:
        raise ValueError("K must be less than or equal to num_steps.")

    # triggers global variables is_lapicque etc for neurons_dict
    # redo reset in training loop
    utils.reset(net=net)

    neurons_dict = {  # confirm this is even useful
        utils.is_lapicque: snn.Lapicque,
        utils.is_leaky: snn.Leaky,
        utils.is_synaptic: snn.Synaptic,
        utils.is_alpha: snn.Alpha,
        utils.is_stein: snn.Stein,
    }

    # element 1: if true: spk, if false, mem
    # element 2: if true: time_varying_targets

    criterion_dict = {
        "mse_membrane_loss": [
            False,
            True,
        ],  # if time_var_target is true, need a flag to let mse_mem_loss know when to re-start iterating targets from
        "ce_max_membrane_loss": [False, False],
        "ce_rate_loss": [True, False],
        "ce_count_loss": [True, False],
        "mse_count_loss": [True, False],
    }  # note: when using mse_count_loss, the target spike-count should be for a truncated time, not for the full time

    reg_dict = {"l1_rate_sparsity": True}

    # acc_dict = {
    #     SF.accuracy_rate : [False, False, False, True]
    # }

    time_var_targets = False
    counter = len(criterion_dict)
    for criterion_key in criterion_dict:
        if criterion_key == criterion.__name__:
            loss_spk, time_var_targets = criterion_dict[
                criterion_key
            ]  # m: mem, s: spk // s: every step, e: end
            if time_var_targets:
                time_var_targets = criterion.time_var_targets  # check this
        counter -= 1
    if counter:  # fix the print statement
        raise TypeError(
            "``criterion`` must be one of the loss functions in ``snntorch.functional``: e.g., 'mse_membrane_loss', 'ce_max_membrane_loss', 'ce_rate_loss' etc."
        )

    if regularization:
        for reg_item in reg_dict:
            if reg_item == regularization.__name__:
                reg_spk = reg_dict[reg_item]  # m: mem, s: spk // s: every step, e: end

    num_return = utils._final_layer_check(net)  # number of outputs

    step_trunc = 0  # ranges from 0 to K, resetting every K time steps
    K_count = 0
    loss_trunc = 0  # reset every K time steps
    loss_avg = 0

    mem_rec = []
    spk_rec = []
    mem_rec_trunc = []
    spk_rec_trunc = []

    net = net.to(device)

    data_iterator = iter(dataloader)
    for data, targets in data_iterator:
        net.train()
        data = data.to(device)
        targets = targets.to(device)

        utils.reset(net)

        for step in range(num_steps):
            if num_return == 2:
                if time_var:
                    spk, mem = net(data[step])
                else:
                    spk, mem = net(data)

            elif num_return == 3:
                if time_var:
                    spk, _, mem = net(data[step])
                else:
                    spk, _, mem = net(data)

            elif num_return == 4:
                if time_var:
                    spk, _, _, mem = net(data[step])
                else:
                    spk, _, _, mem = net(data)

            # else:  # assume not an snn.Layer returning 1 val
            #     if time_var:
            #         spk = net(data[step])
            #     else:
            #         spk = net(data)
            #     spk_rec.append(spk)

            spk_rec_trunc.append(spk)
            mem_rec_trunc.append(mem)

            step_trunc += 1
            if step_trunc == K:
                spk_rec += spk_rec_trunc
                mem_rec += mem_rec_trunc

                spk_rec_trunc = torch.stack(spk_rec_trunc, dim=0)
                mem_rec_trunc = torch.stack(mem_rec_trunc, dim=0)

                # loss_spk is True if input to criterion is spk;
                # reg_spk is True if input to reg is spk

                # catch case for time_varying_targets?
                if time_var_targets:
                    if loss_spk:
                        loss = criterion(
                            spk_rec_trunc,
                            targets[int(K_count * K) : int((K_count + 1) * K)],
                        )
                    else:
                        loss = criterion(
                            mem_rec_trunc,
                            targets[int(K_count * K) : int((K_count + 1) * K)],
                        )
                else:
                    if loss_spk:
                        loss = criterion(spk_rec_trunc, targets)
                    else:
                        loss = criterion(mem_rec_trunc, targets)

                if regularization:
                    if reg_spk:
                        loss += regularization(spk_rec_trunc)
                    else:
                        loss += regularization(mem_rec_trunc)

                loss_trunc += loss
                loss_avg += loss / K

                optimizer.zero_grad()
                loss_trunc.backward()
                optimizer.step()

                for neuron in neurons_dict:
                    if neuron:
                        neurons_dict[
                            neuron
                        ].detach_hidden()  # might need to swap detach_hidden --> _reset_hidden

                K_count += 1
                step_trunc = 0
                loss_trunc = 0
                spk_rec_trunc = []
                mem_rec_trunc = []

        if (step == num_steps - 1) and (num_steps % K):
            spk_rec += spk_rec_trunc
            mem_rec += mem_rec_trunc

            spk_rec_trunc = torch.stack(spk_rec_trunc, dim=0)
            mem_rec_trunc = torch.stack(mem_rec_trunc, dim=0)

            if time_var_targets:
                if loss_spk:
                    loss = criterion(
                        spk_rec_trunc,
                        targets[int(K_count * K) : int(K_count * K + num_steps % K)],
                    )
                else:
                    loss = criterion(
                        mem_rec_trunc,
                        targets[int(K_count * K) : int(K_count * K + num_steps % K)],
                    )
            else:
                if loss_spk:
                    loss = criterion(spk_rec_trunc, targets)
                else:
                    loss = criterion(mem_rec_trunc, targets)

            if regularization:
                if reg_spk:
                    loss += regularization(spk_rec_trunc)
                else:
                    loss += regularization(mem_rec_trunc)

            loss_trunc += loss
            loss_avg += loss / int(num_steps % K)

            optimizer.zero_grad()
            loss_trunc.backward()
            optimizer.step()

            K_count = 0
            step_trunc = 0
            loss_trunc = 0
            spk_rec_trunc = []
            mem_rec_trunc = []

            for neuron in neurons_dict:
                if neuron:
                    neurons_dict[neuron].detach_hidden()

    return loss_avg, spk_rec, mem_rec


def BPTT(
    net,
    data,
    target,
    num_steps,
    batch_size,
    optimizer,
    criterion,
    time_var,
    return_spk=False,
    return_mem=False,
):
    """Backpropagation through time. LIF layers require parameter ``init_hidden = True``.
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
    :param time_var: True if input data is time-varying [T x B x dims], defaults to ``True``
    :type time_var: Bool, optional
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
        time_var,
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
    time_var,
    return_spk=False,
    return_mem=False,
):
    """Real-time Recurrent Learning. LIF layers require parameter ``init_hidden = True``.
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
    :param time_var_data: True if input data is time-varying [T x B x dims], defaults to ``True``
    :type time_var_data: Bool, optional
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
        time_var,
        return_spk,
        return_mem,
        K=1,
    )


def _rec_trunc(
    spk,
    mem,
    regularization,
    loss_spk,
    reg_spk=False,
    spk_rec_trunc=False,
    mem_rec_trunc=False,
):
    if regularization:
        if loss_spk and reg_spk:
            spk_rec_trunc.append(spk)
        if loss_spk != reg_spk:
            spk_rec_trunc.append(spk)
            mem_rec_trunc.append(mem)
        else:
            mem_rec_trunc.append(mem)
    else:
        if loss_spk:  # risk: removing option for losses with both mem and spk
            spk_rec_trunc.append(spk)
        else:
            mem_rec_trunc.append(mem)
