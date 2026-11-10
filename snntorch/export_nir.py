#from typing import Optional
from typing import Optional, Tuple, Union
import torch
import os
import sys
import nir
import numpy as np
import nirtorch
import snntorch as snn



def _broadcast_param(param, view):
    """Broadcast a neuron parameter to the measured layer view.

    Scalar parameters (e.g. ``beta=0.9``) carry no layer-width information:
    exported as-is they produce zero-dimensional NIR parameter arrays, which
    degenerate the node's port types to ``[]`` and fail strict type checking.
    Broadcasting them to the layer's measured output view gives every NIR
    parameter the same shape and a well-formed port type. Parameters that
    already carry per-neuron values are reshaped to the view when the element
    count matches.
    """
    arr = np.asarray(param)
    view = tuple(int(d) for d in view)
    if arr.shape == view:
        return arr
    if arr.size == int(np.prod(view)):
        return arr.reshape(view)
    try:
        return np.broadcast_to(arr, view).copy()
    except ValueError as exc:
        raise ValueError(
            f"neuron parameter of shape {arr.shape} cannot be broadcast to "
            f"the measured layer output shape {view}"
        ) from exc


def _measure_output_views(module, sample_data, model_fwd_args, ignore_dims):
    """Run one forward pass with hooks and record each submodule's output view.

    The returned dict maps ``id(submodule)`` to its output shape with
    ``ignore_dims`` removed — the broadcast target for scalar neuron
    parameters. Interior NIR port types are batch-free (Linear/Affine derive
    them from weights and Flatten drops the batch dim), so when ``ignore_dims``
    is empty the leading dim of a multi-dim output is dropped the same way.
    Modules that return tuples (``init_hidden=False``, or ``output=True``)
    report the shape of their first element, which is the spike tensor that
    flows to the next layer.
    """
    shapes = {}
    tuple_modules = []
    hooks = []

    def make_hook(mod):
        def hook(_m, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(output, tuple):
                tuple_modules.append(mod)
            if isinstance(out, torch.Tensor):
                shapes[id(mod)] = tuple(out.shape)

        return hook

    for _name, mod in module.named_modules():
        hooks.append(mod.register_forward_hook(make_hook(mod)))

    # The probe forward must be side-effect free: init_hidden neurons keep
    # their membrane/spike/synaptic state between calls, so snapshot and
    # restore it. snnTorch state updates rebind these attributes (never
    # in-place), so saving the original objects is sufficient.
    saved_state = []
    for mod in module.modules():
        for attr in ("mem", "spk", "syn", "_threshold_adapt"):
            if hasattr(mod, attr):
                saved_state.append((mod, attr, getattr(mod, attr)))

    try:
        with torch.no_grad():
            module(sample_data, *model_fwd_args)
    except TypeError as exc:
        if tuple_modules:
            raise TypeError(
                "export_to_nir cannot trace the model: a module returned a "
                "tuple that was fed into the next layer (init_hidden=False "
                "or output=True on a non-final spiking module). Only the "
                "final module may return a tuple; use init_hidden=True for "
                "intermediate spiking layers."
            ) from exc
        raise
    finally:
        for handle in hooks:
            handle.remove()
        for mod, attr, value in saved_state:
            setattr(mod, attr, value)

    drop_dims = list(ignore_dims) if ignore_dims is not None else []
    views = {}
    for mod_id, shape in shapes.items():
        dims = drop_dims if drop_dims else ([0] if len(shape) >= 2 else [])
        views[mod_id] = tuple(np.delete(np.asarray(shape), dims))
    return views


def _declared_port_shape(port_type):
    """Single-port declared shape, or None when undeclared or degenerate."""
    if not port_type or len(port_type) != 1:
        return None
    value = next(iter(port_type.values()))
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0 or arr.dtype == object:
        return None
    return arr


def _align_endpoint_types(nir_graph):
    """Re-derive Input/Output endpoint port types from their graph neighbours.

    ``extract_nir_graph`` builds the endpoints from the raw sample and module
    output shapes, which keep the batch dim while interior port types are
    batch-free. Copying the neighbour's declared shape (under the endpoint's
    own port names) keeps the endpoint edges consistent; endpoints with
    undeclared or disagreeing neighbour types are left for ``infer_types``
    to check.
    """
    nodes = nir_graph.nodes
    for key, node in nodes.items():
        if isinstance(node, nir.Input):
            # input -> successor: the successor's input faces this edge
            neighbours = [d for s, d in nir_graph.edges if s == key]
            attr = "input_type"
        elif isinstance(node, nir.Output):
            # predecessor -> output: the predecessor's output faces this edge
            neighbours = [s for s, d in nir_graph.edges if d == key]
            attr = "output_type"
        else:
            continue
        shapes = [
            _declared_port_shape(getattr(nodes[n_key], attr, None))
            for n_key in dict.fromkeys(neighbours)
        ]
        shapes = [s for s in shapes if s is not None]
        if not shapes or not all(np.array_equal(s, shapes[0]) for s in shapes):
            continue
        for own_attr in ("input_type", "output_type"):
            own = getattr(node, own_attr, None)
            if own:
                setattr(
                    node, own_attr, {k: np.array(shapes[0]) for k in own}
                )


def _extract_snntorch_module(
    module: torch.nn.Module, output_view: Optional[Tuple[int, ...]] = None
) -> Optional[nir.NIRNode]:
    """Convert a single snnTorch module to the equivalent object in the Neuromorphic
    Intermediate Representation (NIR). This function is used internally by the export_to_nir
    function to convert each submodule/layer of the network to the NIR.

    Currently supported snnTorch modules: Leaky, Linear, Synaptic, RLeaky, RSynaptic.

    Note that recurrent layers such as RLeaky and RSynaptic will be converted to a NIR graph,
    which will then be embedded as a subgraph into the main NIR graph.

    :param module: snnTorch module
    :type module: torch.nn.Module

    :param output_view: measured output shape of the module (without
        ``ignore_dims``); scalar neuron parameters are broadcast to it
    :type output_view: Optional[Tuple[int, ...]]

    :return: return the NIR node
    :rtype: Optional[nir.NIRNode]
    """
    # Adding Conv2d layer
    """
    if isinstance(module, torch.nn.Conv2d):
        return nir.Conv2d(
            input_shape=None,
            weight=module.weight.detach(),
            bias=module.bias.detach(),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
    """
    #modifiying bias of the conv2d layer extraction 
    if isinstance(module, torch.nn.Conv2d):
        return nir.Conv2d(
            input_shape=None,
            weight=module.weight.detach(),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            #better handle for the bias if it's False
            bias=(
                module.bias.detach()
                if isinstance(module.bias, torch.Tensor)
                else torch.zeros((module.weight.shape[0]))
            ),
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        return nir.AvgPool2d(
            kernel_size=module.kernel_size,  # (Height, Width)
            stride=module.kernel_size
            if module.stride is None
            else module.stride,  # (Height, width)
            padding=(0, 0),  # (Height, width)
        )

    elif isinstance(module, snn.Leaky):
        dt = 1e-4

        beta = module.beta.detach().cpu().numpy()
        vthr = module.threshold.detach().cpu().numpy()
        if output_view is not None:
            beta = _broadcast_param(beta, output_view)
            vthr = _broadcast_param(vthr, output_view)
        else:
            vthr = np.array([vthr]) if isinstance(vthr, (int, float)) else vthr
        tau_mem = dt / (1 - beta)
        r = tau_mem / dt
        v_leak = np.zeros_like(beta)

        return nir.LIF(
            tau=tau_mem,
            v_threshold=vthr,
            v_leak=v_leak,
            r=r,
            v_reset=np.zeros_like(vthr),
        )

    elif isinstance(module, torch.nn.Linear):
        if module.bias is None:
            return nir.Linear(weight=module.weight.data.detach().cpu().numpy())
        else:
            return nir.Affine(
                weight=module.weight.data.detach().cpu().numpy(),
                bias=module.bias.data.detach().cpu().numpy(),
            )

    elif isinstance(module, snn.Synaptic):
        dt = 1e-4

        # TODO: assert that size of the current layer is correct
        alpha = module.alpha.detach().cpu().numpy()
        beta = module.beta.detach().cpu().numpy()
        vthr = module.threshold.detach().cpu().numpy()
        if output_view is not None:
            alpha = _broadcast_param(alpha, output_view)
            beta = _broadcast_param(beta, output_view)
            vthr = _broadcast_param(vthr, output_view)
        else:
            vthr = np.array([vthr]) if isinstance(vthr, (int, float)) else vthr

        tau_syn = dt / (1 - alpha)
        tau_mem = dt / (1 - beta)
        r = tau_mem / dt
        v_leak = np.zeros_like(beta)
        w_in = tau_syn / dt

        return nir.CubaLIF(
            tau_syn=tau_syn,
            tau_mem=tau_mem,
            v_threshold=vthr,
            v_leak=v_leak,
            r=r,
            w_in=w_in,
            v_reset=np.zeros_like(vthr),
        )

    elif isinstance(module, snn.RLeaky):
        if module.all_to_all:
            w_rec = _extract_snntorch_module(module.recurrent)
            if hasattr(w_rec, "weight"):
                n_neurons = w_rec.weight.shape[0]
            else:
                # fallback for conv layers
                n_neurons = w_rec.weight.shape[0]
        else:
            if len(module.recurrent.V.shape) == 0:
                raise ValueError(
                    "V must be a vector, cannot infer layer size for scalar V"
                )
            n_neurons = module.recurrent.V.shape[0]
            w = np.diag(module.recurrent.V.data.detach().cpu().numpy())
            w_rec = nir.Linear(weight=w)

        dt = 1e-4

        beta = module.beta.detach().cpu().numpy()
        vthr = module.threshold.detach().cpu().numpy()
        beta = np.ones(n_neurons) * beta
        vthr = np.ones(n_neurons) * vthr

        tau_mem = dt / (1 - beta)
        r = tau_mem / dt
        v_leak = np.zeros_like(beta)

        return nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=[n_neurons]),
                "lif": nir.LIF(
                    v_threshold=vthr,
                    tau=tau_mem,
                    r=r,
                    v_leak=v_leak,
                    v_reset=np.zeros_like(vthr),
                ),
                "w_rec": w_rec,
                "output": nir.Output(output_type=[n_neurons]),
            },
            edges=[
                ("input", "lif"),
                ("lif", "w_rec"),
                ("w_rec", "lif"),
                ("lif", "output"),
            ],
        )

    elif isinstance(module, snn.RSynaptic):
        if module.all_to_all:
            w_rec = _extract_snntorch_module(module.recurrent)
            n_neurons = w_rec.weight.shape[0]
        else:
            if len(module.recurrent.V.shape) == 0:
                # TODO: handle this better - if V is a scalar, then the weight has wrong shape
                raise ValueError(
                    "V must be a vector, cannot infer layer size for scalar V"
                )
            n_neurons = module.recurrent.V.shape[0]
            w = np.diag(module.recurrent.V.data.detach().cpu().numpy())
            w_rec = nir.Linear(weight=w)

        dt = 1e-4

        alpha = module.alpha.detach().cpu().numpy()
        beta = module.beta.detach().cpu().numpy()
        vthr = module.threshold.detach().cpu().numpy()
        alpha = np.ones(n_neurons) * alpha
        beta = np.ones(n_neurons) * beta
        vthr = np.ones(n_neurons) * vthr

        tau_syn = dt / (1 - alpha)
        tau_mem = dt / (1 - beta)
        r = tau_mem / dt
        v_leak = np.zeros_like(beta)
        w_in = tau_syn / dt

        return nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=[n_neurons]),
                "lif": nir.CubaLIF(
                    v_threshold=vthr,
                    tau_mem=tau_mem,
                    tau_syn=tau_syn,
                    r=r,
                    v_leak=v_leak,
                    w_in=w_in,
                    v_reset=np.zeros_like(vthr),
                ),
                "w_rec": w_rec,
                "output": nir.Output(output_type=[n_neurons]),
            },
            edges=[
                ("input", "lif"),
                ("lif", "w_rec"),
                ("w_rec", "lif"),
                ("lif", "output"),
            ],
        )
    elif isinstance(module, torch.nn.Flatten):
        # Getting rid of the batch dimension for NIR
        start_dim = (
            module.start_dim - 1 if module.start_dim > 0 else module.start_dim
        )
        end_dim = module.end_dim - 1 if module.end_dim > 0 else module.end_dim
        return nir.Flatten(
            input_type=None,
            start_dim=start_dim,
            end_dim=end_dim,
        )

    else:
        print(f"[WARNING] module not implemented: {module.__class__.__name__}")
        return None


def export_to_nir(
    module: torch.nn.Module,
    sample_data: torch.Tensor,
    model_name: str = "snntorch",
    model_fwd_args=[],
    ignore_dims=[],
) -> nir.NIRNode:
    """Convert an snnTorch module to the Neuromorphic Intermediate Representation (NIR).
    This function uses nirtorch to extract the computational graph of the torch module,
    and the _extract_snntorch_module method is used to convert each module in the graph
    to the corresponding NIR module.

    The NIR is a graph-based representation of a spiking neural network, which can be used to
    port the network to different neuromorphic hardware and software platforms.

    Example::

        import snntorch as snn
        import torch
        from snntorch.export_nir import export_to_nir

        lif1 = snn.Leaky(beta=0.9, init_hidden=True)
        lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(784, 500),
            lif1,
            torch.nn.Linear(500, 10),
            lif2
        )

        sample_data = torch.randn(1, 784)
        nir_graph = export_to_nir(net, sample_data)

    :param module: Network model (either wrapped in Sequential container or as a class)
    :type module: torch.nn.Module

    :param sample_data: Sample input data to the network
    :type sample_data: torch.Tensor

    :param model_name: Name of the model
    :type model_name: str, optional

    :param model_fwd_args: Arguments to pass to the forward function of the model
    :type model_fwd_args: list, optional

    :param ignore_dims: List of dimensions to ignore when extracting the NIR
    :type ignore_dims: list, optional

    :return: return the NIR graph
    :rtype: nir.NIRNode
    """
    output_views = _measure_output_views(
        module, sample_data, model_fwd_args, ignore_dims
    )
    nir_graph = nirtorch.extract_nir_graph(
        module,
        lambda mod: _extract_snntorch_module(mod, output_views.get(id(mod))),
        sample_data,
        model_name=model_name,
        ignore_submodules_of=[snn.RLeaky, snn.RSynaptic],
        model_fwd_args=model_fwd_args,
        ignore_dims=ignore_dims,
    )
    _align_endpoint_types(nir_graph)
    # ensure node input and output types are fully defined
    nir_graph.infer_types()
    return nir_graph
