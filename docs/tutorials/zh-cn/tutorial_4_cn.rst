===========================
教程（四） - 二阶脉冲神经元模型
===========================

本教程出自 Jason K. Eshraghian (`www.ncg.ucsc.edu <https://www.ncg.ucsc.edu>`_)

 `English <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_4.html#>`_ 

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_4_advanced_neurons.ipynb

snnTorch 教程系列基于以下论文。如果您发现这些资源或代码对您的工作有用, 请考虑引用以下来源：

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  本教程是不可编辑的静态版本。交互式可编辑版本可通过以下链接获取：
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_4_advanced_neurons.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_



简介
-------------

在本教程中, 你将: 

* 了解更先进的LIF神经元模型： ``Synaptic（突触传导）`` 和 ``Alpha`` 

安装 snnTorch 的最新 PyPi 发行版。

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


1. 基于突触传导的LIF神经元模型
------------------------------------------------

在前几个教程中探讨的神经元模型中，我们假设输入电压脉冲会导致突触电流瞬间跃升，然后对膜电位产生影响。
但实际上，一个脉冲将导致神经递质从前突触神经元（pre-synaptic neuron）逐渐释放到后突触神经元（post-synaptic neuron）。基于突触（synapse）传导的LIF模型考虑了输入电流的渐变时间动态。

1.1 建模突触电流
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

从生物学角度讲，如果前突触神经元发放脉冲，电压脉冲将传递到神经元的轴突（axon）。
它触发囊泡释放神经递质到突触间隙。这些神经递质激活后突触受体，直接影响流入后突触神经元的有效电流。
下面显示了两种兴奋性受体，AMPA和NMDA。

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_6_synaptic.png?raw=true
        :align: center
        :width: 600

最简单的突触电流模型假定电流在极快的时间尺度上不断增加，随后出现相对缓慢的指数衰减，正如上文 AMPA 受体反应所示。这与 Lapicque 模型中的膜电位动态非常相似。

突触模型有两个指数衰减项：:math:`I_{\rm syn}(t)` 和 :math:`U_{\rm mem}(t)`。 :math:`I_{\rm syn}(t)` 的后续项之间的比例（即衰减率）设置为 :math:`\alpha`，:math:`U(t)` 的比例设置为 :math:`\beta`：

.. math::  \alpha = e^{-\Delta t/\tau_{\rm syn}}

.. math::  \beta = e^{-\Delta t/\tau_{\rm mem}}

其中单个时间步长的持续时间规范化为 :math:`\Delta t = 1`。 :math:`\tau_{\rm syn}` 以类似的方式模拟突触电流的时间常数，就像 :math:`\tau_{\rm mem}` 模拟膜电位的时间常数一样。 :math:`\beta` 以与前一个教程相同的方式派生，对 :math:`\alpha` 采用类似的方法：

.. math:: I_{\rm syn}[t+1]=\underbrace{\alpha I_{\rm syn}[t]}_\text{衰减} + \underbrace{WX[t+1]}_\text{输入}

.. math:: U[t+1] = \underbrace{\beta U[t]}_\text{衰减} + \underbrace{I_{\rm syn}[t+1]}_\text{输入} - \underbrace{R[t]}_\text{复位}

与之前的LIF神经元一样，触发脉冲的条件仍然成立：

.. math::

   S_{\rm out}[t] = \begin{cases} 1, &\text{如果}~U[t] > U_{\rm thr} \\
   0, &\text{否则}\end{cases}

1.2 snnTorch中的突触神经元模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

突触传导模型将突触电流动力学与被动膜结合在一起。它必须使用两个输入参数实例化：

* :math:`\alpha`：突触电流的衰减率
* :math:`\beta`：膜电位的衰减率（与Lapicque模型相同）

::

    # 时间动态
    alpha = 0.9
    beta = 0.8
    num_steps = 200
    
    # 初始化2阶LIF神经元
    lif1 = snn.Synaptic(alpha=alpha, beta=beta)

使用这个神经元与之前的LIF神经元完全相同，但现在加入了突触电流``syn``作为输入和输出：

**输入** 

* ``spk_in``：每个加权输入电压脉冲 :math:`WX[t]` 被顺序传递
* ``syn``：上一时间步的突触电流 :math:`I_{\rm syn}[t-1]`
* ``mem``：上一时间步的膜电位 :math:`U[t-1]`

**输出** 

* ``spk_out``：输出脉冲 :math:`S[t]`（如果有脉冲则为'1'；如果没有脉冲则为'0'）
* ``syn``：当前时间步的突触电流 :math:`I_{\rm syn}[t]`
* ``mem``：当前时间步的膜电位 :math:`U[t]`

这些都需要是 ``torch.Tensor`` 类型。请注意，神经元模型已经向后移动了一步，不过无所谓。

应用周期性的脉冲输入，观察电流和膜随时间的演变：

::

    # 周期性脉冲输入，spk_in = 0.2 V
    w = 0.2
    spk_period = torch.cat((torch.ones(1)*w, torch.zeros(9)), 0)
    spk_in = spk_period.repeat(20)
    
    # 初始化隐藏状态和输出
    syn, mem = lif1.init_synaptic()
    spk_out = torch.zeros(1) 
    syn_rec = []
    mem_rec = []
    spk_rec = []
    
    # 模拟神经元
    for step in range(num_steps):
      spk_out, syn, mem = lif1(spk_in[step], syn, mem)
      spk_rec.append(spk_out)
      syn_rec.append(syn)
      mem_rec.append(mem)
    
    # 将列表转换为张量
    spk_rec = torch.stack(spk_rec)
    syn_rec = torch.stack(syn_rec)
    mem_rec = torch.stack(mem_rec)
    
    plot_spk_cur_mem_spk(spk_in, syn_rec, mem_rec, spk_rec, 
                         "带输入脉冲的突触传导型神经元模型")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial4/_static/syn_cond_spk.png?raw=true
        :align: center
        :width: 450

该模型还具有可选的输入参数 ``reset_mechanism`` 和 ``threshold`` ，如Lapicque的神经元模型所述。
总之，每个脉冲都会对突触电流 :math:`I_{\rm syn}` 产生一个平移的指数衰减，然后将它们全部相加。
然后，这个电流由在 `教程（二） <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_ 中导出的被动膜方程进行积分，从而生成输出脉冲。下图示意了这个过程。

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_7_stein.png?raw=true
        :align: center
        :width: 450

1.3 一阶神经元与二阶神经元
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

一个自然而然的问题是 - *我什么时候应该使用一阶LIF神经元，什么时候应该使用这种二阶LIF神经元？* 虽然这个问题还没有真正解决，但我的实验给了我一些可能有用的灵感。

**二阶神经元更好的情况** 

* 如果你的输入数据的时间关系发生在长时间尺度上，
* 或者如果输入的脉冲模式是稀疏的

通过有两个循环方程和两个衰减项（:math:`\alpha` 和 :math:`\beta`），这种神经元模型能够在更长的时间内“维持”输入脉冲。这对于保持长期关系是有益的。

另一种可能的用例是：

- 当时间编码很重要时

如果你关心一个脉冲的精确时间，对于二阶神经元来说，控制起来似乎更容易。
在 ``Leaky`` 模型中，一个脉冲将直接与输入同步触发。
对于二阶模型，膜电位被“平滑处理”（即，突触电流模型对膜电位进行低通滤波），这意味着可以为 :math:`U[t]` 使用有限的上升时间。
这在之前的模拟中很明显，其中输出脉冲相对于输入脉冲有所延迟。

**一阶神经元更好的情况** 

* 任何不属于上述情况的情况，有时，甚至包括上述情况。

一阶神经元模型（如 ``Leaky``）只有一个方程，使得反向传播过程稍微简单一些。
尽管如此， ``Synaptic`` 模型在 :math:`\alpha=0.` 时功能上等同于 ``Leaky`` 模型。
在我对简单数据集进行的超参数扫描中，最佳结果似乎将 :math:`\alpha` 尽可能接近 0。
随着数据复杂性的增加，:math:`\alpha` 可能会变大。


1.3 一阶神经元与二阶神经元
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

一个自然而然的问题是 - *我什么时候应该使用一阶LIF神经元，什么时候应该使用这种二阶LIF神经元？* 
虽然这个问题还没有真正解决，但我的实验给了我一些可能有用的直觉。

**二阶神经元更好的情况**

* 如果你的输入数据的时间关系发生在长时间尺度上，
* 或者如果输入的脉冲模式是稀疏的

通过有两个循环方程和两个衰减项（:math:`\alpha` 和 :math:`\beta`），
这种神经元模型能够在更长的时间内“维持”输入脉冲。这对于保持长期关系是有益的。

另一种可能的用例是：

- 当时间编码很重要时

如果你关心一个脉冲的精确时间，对于二阶神经元来说，控制起来似乎更容易。在 ``Leaky`` 模型中，
一个脉冲将直接与输入同步触发。对于二阶模型，膜电位被“平滑处理”（即，突触电流模型对膜电位进行低通滤波），
这意味着可以为 :math:`U[t]` 使用有限的上升时间。这在之前的模拟中很明显，其中输出脉冲相对于输入脉冲有所延迟。

**一阶神经元更好的情况**

* 任何不属于上述情况的情况，有时，甚至包括上述情况。

一阶神经元模型（如 ``Leaky``）只有一个方程，使得反向传播过程稍微简单一些。
尽管如此，``Synaptic`` 模型在 :math:`\alpha=0.` 时功能上等同于 ``Leaky`` 模型。
在我对简单数据集进行的超参数扫描中，最佳结果似乎将 :math:`\alpha` 尽可能接近 0。随着数据复杂性的增加，:math:`\alpha` 可能会变大。


2.1 建模 Alpha 神经元模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

正式一点，这个过程由下式表示：

.. math:: U_{\rm mem}(t) = \sum_i W(\epsilon * S_{\rm in})(t)

其中，进入的脉冲 :math:`S_{\rm in}` 与脉冲响应核 :math:`\epsilon( \cdot )` 进行卷积。脉冲响应通过突触权重 :math:`W` 进行缩放。
在顶部的图形中，核是一个指数衰减函数，相当于Lapicque的一阶神经元模型。在底部，核是一个alpha函数：

.. math:: \epsilon(t) = \frac{t}{\tau}e^{1-t/\tau}\Theta(t)

其中 :math:`\tau` 是 alpha 核的时间常数，:math:`\Theta` 是 Heaviside 阶跃函数。大多数基于核的方法采用 alpha 函数，因为它提供了对于关心指定神经元精确脉冲时间的时间编码很有用的时间延迟。

在 snnTorch 中，脉冲响应模型不是直接作为滤波器实现的。相反，它被重构成递归形式，这样只需要前一个时间步的值就可以计算下一组值。这减少了所需的内存。

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_9_alpha.png?raw=true
        :align: center
        :width: 550

由于膜电位现在由两个指数之和决定，因此这些指数每个都有自己的独立衰减率。:math:`\alpha` 定义正指数的衰减率，:math:`\beta` 定义负指数的衰减率。

::

    alpha = 0.8
    beta = 0.7
    
    # 初始化神经元
    lif2 = snn.Alpha(alpha=alpha, beta=beta, threshold=0.5)

使用这种神经元与之前的神经元相同，但是两个指数函数之和要求将突触电流 ``syn`` 分成 ``syn_exc`` 和 ``syn_inh`` 两个部分：

**输入** 

* ``spk_in``：每个加权输入电压脉冲 :math:`WX[t]` 依次传入 
* ``syn_exc``：前一个时间步的兴奋性突触后电流 :math:`I_{\rm syn-exc}[t-1]` 
* ``syn_inh``：前一个时间步的抑制性突触后电流 :math:`I_{\rm syn-inh}[t-1]` 
* ``mem``：当前时间 :math:`t` 前一个时间步的膜电位 :math:`U_{\rm mem}[t-1]`

**输出** 

* ``spk_out``：当前时间步的输出脉冲 :math:`S_{\rm out}[t]`（如果有脉冲则为‘1’；如果没有脉冲则为‘0’）
* ``syn_exc``：当前时间步 :math:`t` 的兴奋性突触后电流 :math:`I_{\rm syn-exc}[t]` 
* ``syn_inh``：当前时间步 :math:`t` 的抑制性突触后电流 :math:`I_{\rm syn-inh}[t]` 
* ``mem``：当前时间步的膜电位 :math:`U_{\rm mem}[t]`

与所有其他神经元模型一样，这些必须是 ``torch.Tensor`` 类型。

::

    # 输入脉冲：初始脉冲，然后是周期性脉冲
    w = 0.85
    spk_in = (torch.cat((torch.zeros(10), torch.ones(1), torch.zeros(89), 
                         (torch.cat((torch.ones(1), torch.zeros(9)),0).repeat(10))), 0) * w).unsqueeze(1)
    
    # 初始化参数
    syn_exc, syn_inh, mem = lif2.init_alpha()
    mem_rec = []
    spk_rec = []
    
    # 运行模拟
    for step in range(num_steps):
      spk_out, syn_exc, syn_inh, mem = lif2(spk_in[step], syn_exc, syn_inh, mem)
      mem_rec.append(mem.squeeze(0))
      spk_rec.append(spk_out.squeeze(0))
    
    # 将列表转换为张量
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_spk_mem_spk(spk_in, mem_rec, spk_rec, "Alpha 神经元模型带输入脉冲")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial4/_static/alpha.png?raw=true
        :align: center
        :width: 500

与 Lapicque 和 Synaptic 模型一样，Alpha 模型也有修改阈值和重置机制的选项。

2.2 实际考虑
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如同前面对突触神经元的讨论，模型越复杂，训练过程中的反向传播过程也越复杂。
在我自己的实验中，我还没有发现 Alpha 神经元在性能上超越突触和Leaky神经元模型的案例。
通过正负指数进行学习似乎只会增加梯度计算过程的难度，抵消更复杂的神经元动力学可能带来的好处。

然而，当SRM模型被表示为时变核（而不是像这里这样的递归模型）时，它似乎与简单的神经元模型表现得一样好。例如，参见以下论文：

   `Sumit Bam Shrestha 和 Garrick Orchard, “SLAYER: Spike layer error
   reassignment in time”, Proceedings of the 32nd International
   Conference on Neural Information Processing Systems, pp. 1419-1328,
   2018. <https://arxiv.org/abs/1810.08646>`__

加入 Alpha 神经元的目的是为将基于 SRM 的模型移植到 snnTorch 提供一个选项，尽管在 snnTorch 中对它们进行本机训练似乎不太有效。

结论
------------

我们已经覆盖了 snnTorch 中可用的所有LIF神经元模型。简要总结一下：

-  **Lapicque**：基于 RC-电路参数的物理精确模型
-  **Leaky**：简化的一阶模型
-  **Synaptic**：考虑突触电流演变的二阶模型
-  **Alpha**：膜电位跟踪 alpha 函数的二阶模型

一般来说， ``Leaky`` 和 ``Synaptic`` 似乎对于训练网络最有用。 ``Lapicque`` 适用于演示物理精确模型，而 ``Alpha`` 只旨在捕捉SRM神经元的行为。

使用这些稍微高级一些的神经元构建网络的过程与 `教程3 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_ 中的过程完全相同。

如果您喜欢这个项目，请考虑在 GitHub 上给仓库点赞⭐，这是支持它的最简单也是最好的方式。

参考文献，可在 `这里找到
<https://snntorch.readthedocs.io/en/latest/snntorch.html>`__。

进一步阅读
---------------

-  `在这里查看 snnTorch GitHub 项目。 <https://github.com/jeshraghian/snntorch>`__
-  关于 Lapicque, Leaky, Synaptic, 和 Alpha 模型的 `snnTorch文档 <https://snntorch.readthedocs.io/en/latest/snntorch.html>`__
-  `神经动力学：从单个神经元到网络和认知模型
   <https://neuronaldynamics.epfl.ch/index.html>`__ ，由 Wulfram
   Gerstner, Werner M. Kistler, Richard Naud 和 Liam Paninski 著。
-  `理论神经科学：计算和数学建模的神经
   系统 <https://mitpress.mit.edu/books/theoretical-neuroscience>`__
   ，由 Laurence F. Abbott 和 Peter Dayan 著。

