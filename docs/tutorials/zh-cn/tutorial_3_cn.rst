================================================
教程（三） - 一个前馈脉冲神经网络
================================================

本教程出自 Jason K. Eshraghian (`www.ncg.ucsc.edu <https://www.ncg.ucsc.edu>`_)

 `English <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html#>`_ 

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_3_feedforward_snn.ipynb

snnTorch 教程系列基于以下论文。如果您发现这些资源或代码对您的工作有用, 请考虑引用以下来源：
   
    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  本教程是不可编辑的静态版本。交互式可编辑版本可通过以下链接获取：
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_3_feedforward_snn.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


简介
-------------

在本教程中, 你将: 

* 了解如何简化LIF神经元，使其适合深度学习 
* 实现前馈脉冲神经网络（SNN）

安装 snnTorch 的最新 PyPi 发行版：

::

    $ pip install snntorch

::

    # imports
    import snntorch as snn
    from snntorch import spikeplot as splt
    from snntorch import spikegen
    
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt


1. 简化的LIF神经元模型
----------------------------------------------------------

在前一个教程中，我们设计了自己的LIF神经元模型。但它相当复杂，并添加了一系列需要调整的超参数，
包括 :math:`R`、:math:`C`、:math:`\Delta t`、:math:`U_{\rm thr}` 和复位机制的选择。
这是很多需要跟踪的内容，在扩展到完整的SNN时会变得更加繁琐。所以让我们进行一些简化。


1.1 衰减率：beta
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在之前的教程中，我们使用欧拉方法推导出了被动膜模型的以下解：

.. math:: U(t+\Delta t) = (1-\frac{\Delta t}{\tau})U(t) + \frac{\Delta t}{\tau} I_{\rm in}(t)R \tag{1}

现在假设没有输入电流，即 :math:`I_{\rm in}(t)=0 A`：

.. math:: U(t+\Delta t) = (1-\frac{\Delta t}{\tau})U(t) \tag{2}

令 :math:`U` 的连续值之比，即 :math:`U(t+\Delta t)/U(t)`，为膜电位的衰减率，也称为逆时间常数：

.. math:: U(t+\Delta t) = \beta U(t) \tag{3}

根据 :math:`(1)`，这意味着：

.. math:: \beta = (1-\frac{\Delta t}{\tau}) \tag{4}

为了保证合理的准确性，:math:`\Delta t << \tau`。

1.2 加权输入电流
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果我们假设 :math:`t` 代表序列中的时间步长，而不是连续时间，那么我们可以设置 :math:`\Delta t = 1`。
为了进一步减少超参数的数量，假设 :math:`R=1`。根据 :math:`(4)`，这些假设导致：

.. math:: \beta = (1-\frac{1}{C}) \implies (1-\beta)I_{\rm in} = \frac{1}{\tau}I_{\rm in} \tag{5}

输入电流由 :math:`(1-\beta)` 加权。通过额外假设输入电流瞬间对膜电位产生影响：

.. math:: U[t+1] = \beta U[t] + (1-\beta)I_{\rm in}[t+1] \tag{6}

请注意，时间的离散化意味着我们假设每个时间间隔 :math:`t` 足够短，以至于一个神经元在此区间内最多只能发射一个脉冲。

在深度学习中，输入的加权因子通常是一个可学习的参数。暂时抛开到目前为止所做的符合物理可行性的假设，
我们将 :math:`(6)` 中的 :math:`(1-\beta)` 效应纳入一个可学习的权重 :math:`W` 中，并相应地用输入 :math:`X[t]` 替换 :math:`I_{\rm in}[t]`：

.. math:: WX[t] = I_{\rm in}[t] \tag{7}

这可以这样理解。:math:`X[t]` 是一个输入电压或脉冲，通过 :math:`W` 的突触电导缩放，以产生对神经元的电流注入。这给我们以下结果：

.. math:: U[t+1] = \beta U[t] + WX[t+1] \tag{8}

在未来的模拟中，:math:`W` 和 :math:`\beta` 的效应是分开的。:math:`W` 是一个独立于 :math:`\beta` 更新的可学习参数。

1.3 脉冲发射和重置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们现在引入脉冲发射和重置机制。回想一下，如果膜电位超过阈值，那么神经元将发射一个输出脉冲：

.. math::

   S[t] = \begin{cases} 1, &\text{if}~U[t] > U_{\rm thr} \\
   0, &\text{otherwise} \end{cases}

.. math::
   
   \tag{9}

如果触发了脉冲，膜电位应该被重置。
*通过减法重置*机制可以这样建模：

.. math:: U[t+1] = \underbrace{\beta U[t]}_\text{decay} + \underbrace{WX[t+1]}_\text{input} - \underbrace{S[t]U_{\rm thr}}_\text{reset} \tag{10}

由于 :math:`W` 是一个可学习参数，而 :math:`U_{\rm thr}` 通常只是设为 :math:`1` （尽管也可以调整），这样就只剩下衰减率 :math:`\beta` 作为需要指定的唯一超参数。
这就完成了本教程中繁琐的部分。

.. 请注意::

   一些实现可能会做出略有不同的假设。
   例如，:math:`(9)` 中的:math:`S[t] \rightarrow S[t+1]` ，或
   :math:`(10)` 中的:math:`X[t] \rightarrow X[t+1]` 。以上推导是在snntorch中使用的，
   因为我们发现它直观地映射到循环神经网络的表示中，且不会影响性能。

1.4 代码实现
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

用 Python 实现这个神经元的代码如下所示：

::

    def leaky_integrate_and_fire(mem, x, w, beta, threshold=1):
      spk = (mem > threshold) # 如果膜电位超过阈值，spk=1，否则为0
      mem = beta * mem + w*x - spk*threshold
      return spk, mem

为了设置 :math:`\beta`，我们可以选择使用方程
:math:`(3)` 来定义它，或者直接硬编码。这里，我们将使用
:math:`(3)` 作为示范，但在未来，我们将直接硬编码，因为我们更关注的是实际效果而不是生物学精度。

方程 :math:`(3)` 告诉我们 :math:`\beta` 是
连续两个时间步骤中膜电位的比率。使用连续时间依赖形式的方程（假设
没有电流注入）来解决这个问题，这在 `教程
2 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__ 中已经推导出来了：

.. math:: U(t) = U_0e^{-\frac{t}{\tau}}

:math:`U_0` 是在 :math:`t=0` 时初始的膜电位。假设时间依赖方程是在
:math:`t, (t+\Delta t), (t+2\Delta t)~...~` 的离散步骤中计算的，那么我们可以找到
连续步骤之间的膜电位比率：

.. math:: \beta = \frac{U_0e^{-\frac{t+\Delta t}{\tau}}}{U_0e^{-\frac{t}{\tau}}} = \frac{U_0e^{-\frac{t + 2\Delta t}{\tau}}}{U_0e^{-\frac{t+\Delta t}{\tau}}} =~~...

.. math:: \implies \beta = e^{-\frac{\Delta t}{\tau}} 

::

    # 设置神经元参数
    delta_t = torch.tensor(1e-3)
    tau = torch.tensor(5e-3)
    beta = torch.exp(-delta_t/tau)
   
::

    >>> print(f"衰减率是: {beta:.3f}")
    衰减率是: 0.819

运行一个快速模拟，以检查神经元对阶跃电压输入的响应是否正确：

::

    num_steps = 200
    
    # initialize inputs/outputs + small step current input
    x = torch.cat((torch.zeros(10), torch.ones(190)*0.5), 0)
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec = []
    spk_rec = []
    
    # neuron parameters
    w = 0.4
    beta = 0.819
    
    # neuron simulation
    for step in range(num_steps):
      spk, mem = leaky_integrate_and_fire(mem, x[step], w=w, beta=beta)
      mem_rec.append(mem)
      spk_rec.append(spk)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(x*w, mem_rec, spk_rec, thr_line=1,ylim_max1=0.5,
                     title="LIF Neuron Model With Weighted Step Voltage")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/_static/lif_step.png?raw=true
        :align: center
        :width: 400


2. 在 snnTorch 中的Leaky神经元模型
---------------------------------------

我们可以通过实例化 ``snn.Leaky`` 来实现相同的功能，在这方面和我们在上一个教程中使用的 ``snn.Lapicque`` 类似，但参数更少：

::

    lif1 = snn.Leaky(beta=0.8)

现在神经元模型存储在 ``lif1`` 中。使用这个神经元：

**输入** 

* ``cur_in``: :math:`W\times X[t]` 的每个元素依次作为输入传递
* ``mem``: 之前步骤的膜电位，:math:`U[t-1]`，也作为输入传递。

**输出** 

* ``spk_out``: 输出脉冲 :math:`S[t]` （如果有脉冲为‘1’；没有脉冲为‘0’）
* ``mem``: 当前步骤的膜电位 :math:`U[t]`

这些都需要是 ``torch.Tensor`` 类型。请注意，在这里，我们假设输入电流在传递到
``snn.Leaky`` 神经元之前已经被加权。当我们构建一个网络规模模型时，这将更有意义。此外，方程 :math:`(10)` 在不失一般性的情况下向后移动了一个步骤。

::

    # 小幅度电流输入
    w=0.21
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*w), 0)
    mem = torch.zeros(1)
    spk = torch.zeros(1)
    mem_rec = []
    spk_rec = []
    
    # 神经元模拟
    for step in range(num_steps):
      spk, mem = lif1(cur_in[step], mem)
      mem_rec.append(mem)
      spk_rec.append(spk)
    
    # 将列表转换为张量
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, ylim_max1=0.5,
                     title="snn.Leaky 神经元模型")

将这个图表与手动推导的泄漏积分-脱火神经元进行比较。
膜电位重置略微弱些：即，它使用了
*软重置*。
这样做是有意为之，因为它在一些深度学习基准测试中能够获得更好的性能。
相反使用的方程是：

.. math:: U[t+1] = \underbrace{\beta U[t]}_\text{衰减} + \underbrace{WX[t+1]}_\text{输入} - \underbrace{\beta S[t]U_{\rm thr}}_\text{软重置} \tag{11}


这个模型和 Lapicque 神经元模型一样，有相同的可选输入参数 ``reset_mechanism``
和 ``threshold``。

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/_static/snn.leaky_step.png?raw=true
        :align: center
        :width: 450


3. 一个前馈脉冲神经网络
---------------------------------------------

到目前为止，我们只考虑了单个神经元对输入刺激的响应。snnTorch使将其扩展为深度神经网络变得简单。在本节中，我们将创建一个3层全连接神经网络，维度为784-1000-10。
与迄今为止的模拟相比，每个神经元现在将整合更多的输入脉冲。

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_8_fcn.png?raw=true
        :align: center
        :width: 600

PyTorch用于形成神经元之间的连接，snnTorch用于创建神经元。首先，初始化所有层。

::

    # 层参数
    num_inputs = 784
    num_hidden = 1000
    num_outputs = 10
    beta = 0.99
    
    # 初始化层
    fc1 = nn.Linear(num_inputs, num_hidden)
    lif1 = snn.Leaky(beta=beta)
    fc2 = nn.Linear(num_hidden, num_outputs)
    lif2 = snn.Leaky(beta=beta)

接下来，初始化每个脉冲神经元的隐藏变量和输出。随着网络规模的增加，这变得更加繁琐。可以使用静态方法 ``init_leaky()`` 来处理这个问题。
snnTorch中的所有神经元都有自己的初始化方法，遵循相同的语法，例如 ``init_lapicque()``。隐藏状态的形状会在第一次前向传递期间根据输入数据的维度自动初始化。

::

    # 初始化隐藏状态
    mem1 = lif1.init_leaky()
    mem2 = lif2.init_leaky()
    
    # 记录输出
    mem2_rec = []
    spk1_rec = []
    spk2_rec = []

创建一个输入脉冲列以传递给网络。需要模拟784个输入神经元的200个时间步骤，即原始输入的维度为 :math:`200 \times 784`。
然而，神经网络通常以小批量方式处理数据。snnTorch使用时间优先的维度：

[:math:`时间 \times 批次大小 \times 特征维度`]

因此，将输入沿着 ``dim=1`` 进行“unsqueeze”以指示“一个批次”的数据。这个输入张量的维度必须是 200 :math:`\times` 1 :math:`\times` 784：

::

    spk_in = spikegen.rate_conv(torch.rand((200, 784))).unsqueeze(1)
    >>> print(f"spk_in的维度: {spk_in.size()}")
    "spk_in的维度: torch.Size([200, 1, 784])"

现在终于是时候运行完整的模拟了。将PyTorch和snnTorch协同工作的直观方式是，PyTorch将神经元连接在一起，而snnTorch将结果加载到脉冲神经元模型中。
从编写网络的角度来看，这些脉冲神经元可以像时变激活函数一样处理。

以下是正在发生的事情的顺序说明：

-  从 ``spk_in`` 的第 :math:`i^{th}` 输入到第 :math:`j^{th}` 神经元的权重由 ``nn.Linear`` 中初始化的参数加权：
   :math:`X_{i} \times W_{ij}`
-  这生成了方程 :math:`(10)` 中输入电流项的输入，贡献给脉冲神经元的 :math:`U[t+1]`
-  如果 :math:`U[t+1] > U_{\rm thr}`，则从该神经元触发一个脉冲
-  这个脉冲由第二层权重加权，然后对所有输入、权重和神经元重复上述过程。
-  如果没有脉冲，那么不会传递任何东西给 postsynaptic 神经元。

与迄今为止的模拟唯一的区别是，现在我们使用由 ``nn.Linear`` 生成的权重来缩放输入电流，而不是手动设置 :math:`W`。

::

    # network simulation
    for step in range(num_steps):
        cur1 = fc1(spk_in[step]) # post-synaptic current <-- spk_in x weight
        spk1, mem1 = lif1(cur1, mem1) # mem[t+1] <--post-syn current + decayed membrane
        cur2 = fc2(spk1)
        spk2, mem2 = lif2(cur2, mem2)
    
        mem2_rec.append(mem2)
        spk1_rec.append(spk1)
        spk2_rec.append(spk2)
    
    # convert lists to tensors
    mem2_rec = torch.stack(mem2_rec)
    spk1_rec = torch.stack(spk1_rec)
    spk2_rec = torch.stack(spk2_rec)
    
    plot_snn_spikes(spk_in, spk1_rec, spk2_rec, "Fully Connected Spiking Neural Network")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/_static/mlp_raster.png?raw=true
        :align: center
        :width: 450

在这个阶段，脉冲还没有任何实际意义。输入和权重都是随机初始化的，还没有进行任何训练。但是脉冲应该从第一层传播到输出。
如果您没有看到任何脉冲，那么您可能在权重初始化方面运气不佳 - 您可以尝试重新运行最后四个代码块。

``spikeplot.spike_count`` 可以创建输出层的脉冲计数器。以下动画将需要一些时间来生成。

   注意：如果您在本地桌面上运行代码，请取消下面的行的注释，并修改路径以指向您的 ffmpeg.exe

::

    from IPython.display import HTML
    
    fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
    spk2_rec = spk2_rec.squeeze(1).detach().cpu()
    
    # plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'
    
    # 绘制脉冲计数直方图
    anim = splt.spike_count(spk2_rec, fig, ax, labels=labels, animate=True)
    HTML(anim.to_html5_video())
    # anim.save("spike_bar.mp4")

.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/_static/spike_bar.mp4?raw=true"></video>
  </center>

``spikeplot.traces`` 让您可以可视化膜电位轨迹。我们将绘制10个输出神经元中的9个。将其与上面的动画和 raster 图进行比较，看看是否可以将轨迹与神经元匹配。

::

    # 绘制膜电位轨迹
    splt.traces(mem2_rec.squeeze(1), spk=spk2_rec.squeeze(1))
    fig = plt.gcf() 
    fig.set_size_inches(8, 6)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial3/_static/traces.png?raw=true
        :align: center
        :width: 450

一些神经元在发放脉冲，而其他神经元则完全不发放脉冲是相当正常的。再次强调，直到权重被训练之前，这些脉冲都没有任何实际意义。

结论
-----------

这涵盖了如何简化漏电积分-放电神经元模型，然后使用它构建脉冲神经网络。在实践中，我们几乎总是倾向于在训练网络时使用 ``snn.Leaky`` 而不是 ``snn.Lapicque``，因为后者的超参数搜索空间更小。

`教程（四） <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__
详细介绍了2阶 ``snn.Synaptic`` 和 ``snn.Alpha`` 模型。如果您希望直接进入使用snnTorch进行深度学习，
那么可以跳转到 `教程（五） <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__。

如果您喜欢这个项目，请考虑在GitHub上为仓库加星⭐，因为这是支持它的最简单和最好的方式。

参考文档 `可以在这里找到
<https://snntorch.readthedocs.io/en/latest/snntorch.html>`__。

更多阅读
---------------

-  `在这里查看 snnTorch GitHub 项目。 <https://github.com/jeshraghian/snntorch>`__
-  `snnTorch
   文档 <https://snntorch.readthedocs.io/en/latest/snntorch.html>`__
   的 Lapicque、Leaky、Synaptic 和 Alpha 模型
-  由 Wulfram Gerstner、Werner M. Kistler、Richard Naud 和 Liam Paninski 编写的 `神经元动力学：从单个神经元到认知网络和模型
   <https://neuronaldynamics.epfl.ch/index.html>`__。
-  由 Laurence F. Abbott 和 Peter Dayan 编写的 `理论神经科学：神经系统的计算和数学建模
   <https://mitpress.mit.edu/books/theoretical-neuroscience>`__。
