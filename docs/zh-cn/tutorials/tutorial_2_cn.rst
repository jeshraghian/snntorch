======================================================
教程（二） - Leaky Integrate-and-Fire（LIF）神经元
======================================================

本教程出自 Jason K. Eshraghian (`www.ncg.ucsc.edu <https://www.ncg.ucsc.edu>`_)

 `English <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html#>`_ 

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_2_lif_neuron.ipynb

snnTorch 教程系列基于以下论文。如果您发现这些资源或代码对您的工作有用, 请考虑引用以下来源：

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. arXiv preprint arXiv:2109.12894,
    September 2021. <https://arxiv.org/abs/2109.12894>`_

.. note::
  本教程是不可编辑的静态版本。交互式可编辑版本可通过以下链接获取：
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_2_lif_neuron.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


简介
-------------

在本教程中, 你将: 

* 学习leaky integrate-and-fire (LIF) 神经元模型的基础知识
* 使用snntorch实现一阶LIF神经元

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
    
    import numpy as np
    import matplotlib.pyplot as plt


1. 神经元模型的分类
---------------------------------------

神经元模型种类繁多, 从精确的生物物理模型(比如说Hodgkin-Huxley模型)
到极其简单的人工神经元, 它们遍及现代深度学习的所有方面。

**Hodgkin-Huxley Neuron Models**\ :math:`-`\ 虽然生物物理模型可以高度准确地
再现电生理结果, 但其复杂性使其目前难以使用。

**Artificial Neuron Model**\ :math:`-`\ 人工神经元则是另一方面。
输入乘以相应的权重, 然后通过激活函数。这种简化使深度学习研究人员在计算机视觉、
自然语言处理和许多其他机器学习领域的任务中取得了令人难以置信的成就。

**Leaky Integrate-and-Fire Neuron Models**\ :math:`-`\ Leaky Integrate-and-Fire（LIF）
神经元模型处于两者之间的中间位置。它接收加权输入的总和, 与人工神经元非常相似。
但它并不直接将输入传递给激活函数, 而是在一段时间内通过泄漏对输入进行累积, 
这与 RC 电路非常相似。如果累积值超过阈值, 那么 LIF 神经元就会发出电压脉冲。
LIF 神经元会提取出输出脉冲的形状和轮廓；它只是将其视为一个离散事件。
因此, 信息并不是存储在脉冲中, 而是存储在脉冲的时长（或频率）中。
简单的脉冲神经元模型为神经代码、记忆、网络动力学以及最近的深度学习提供了很多启示。
LIF 神经元介于生物合理性和实用性之间。

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_1_neuronmodels.png?raw=true
        :align: center
        :width: 1000

不同版本的 LIF 模型都有各自的动态特性和用途。snnTorch 目前支持以下 LIF 神经元：

* Lapicque’s RC 模型: ``snntorch.Lapicque`` 
* 一阶模型: ``snntorch.Leaky`` 
* 基于突触电导的神经元模型: ``snntorch.Synaptic``
* 递归一阶模型: ``snntorch.RLeaky``
* 基于递归突触电导的神经元模型: ``snntorch.RSynaptic``
* Alpha神经元模型: ``snntorch.Alpha``

当然也包含一些非LIF脉冲神经元。
本教程主要介绍其中的第一个模型。它将被用来建立 `以下其他模型 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_.

2. Leaky Integrate-and-Fire（LIF） 神经元模型
--------------------------------------------------

2.1 脉冲神经元: 灵感
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在我们的大脑中, 一个神经元可能与1000 - 10000个其他神经元相连。
如果一个神经元脉冲, 所有下坡神经元都可能感受到。但是, 是什么决定了
神经元是否会出现峰值呢？过去一个世纪的实验表明, 如果神经元在输入时受到
*足够的* 刺激, 那么它可能会变得兴奋, 并发出自己的脉冲。

这种刺激从何而来？它可以来自：

* 外围感官, 
* 一种侵入性的电极人工地刺激神经元, 或者在多数情况下, 
* 来自突触前神经元。


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_2_intuition.png?raw=true
        :align: center
        :width: 600

考虑到这些脉冲电位是非常短的电位爆发, 
不太可能所有输入尖峰电位都精确一致地到达神经元体。这表明有时间动态在
‘维持’ 输入脉冲, 就像是延迟.

2.2 被动细胞膜
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

与所有细胞一样, 神经元周围也有一层薄薄的膜。这层膜是一层脂质双分子层, 
将神经元内的导电生理盐水, 与细胞外介质隔离开来。
在电学上, 被绝缘体隔开的两种导电溶液就像一个电容器。

这层膜的另一个作用是控制进出细胞的物质 (比如说钠离子Na\ :math:`^+`). 
神经元膜通常不让离子渗透过去, 这就阻止了离子进出神经元体。但是, 
膜上有一些特定的通道, 当电流注入神经元时, 这些通道就会被触发打开。
这种电荷移动用电阻器来模拟。


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_3_passivemembrane.png?raw=true
        :align: center
        :width: 450

下面的代码块将从头开始推导LIF神经元的行为。如果你想跳过数学, 那请继续往下翻；
在推导之后, 我们将采用更实际的方法来理解LIF神经元动力学。

------------------------

**选读: LIF神经元模型的推导**

现在假设一些任意的时变电流 :math:`I_{\rm in}(t)` 注入了神经元, 
可能是通过电刺激, 也可能是来自其他神经元。 电路中的总电流是守恒的, 所以：

.. math:: I_{\rm in}(t) = I_{R} + I_{C}

根据欧姆定律, 神经元内外测得的膜电位 :math:`U_{\rm mem}` 与通过电阻的电流成正比:

.. math:: I_{R}(t) = \frac{U_{\rm mem}(t)}{R}

电容是神经元上存储的电荷 :math:`Q` 与 :math:`U_{\rm mem}(t)`之间的比例常数:

.. math:: Q = CU_{\rm mem}(t)

电荷变化率给出通过电容的电流:

.. math:: \frac{dQ}{dt}=I_C(t) = C\frac{dU_{\rm mem}(t)}{dt}

因此:

.. math:: I_{\rm in}(t) = \frac{U_{\rm mem}(t)}{R} + C\frac{dU_{\rm mem}(t)}{dt}

.. math:: \implies RC \frac{dU_{\rm mem}(t)}{dt} = -U_{\rm mem}(t) + RI_{\rm in}(t)

等式右边的单位是电压 **\[Voltage]**。在等式的左边, :math:`\frac{dU_{\rm mem}(t)}{dt}` 这一项的单位是 **\[Voltage/Time]**. 为了让等式的两边的单位相等 (都为电压), 
:math:`RC` 的单位必须是 **\[Time]**. 我们称 :math:`\tau = RC` 为电路的时间常数：

.. math:: \tau \frac{dU_{\rm mem}(t)}{dt} = -U_{\rm mem}(t) + RI_{\rm in}(t)

被动细胞膜此时成为了一个线性微分方程。

函数的导数要与原函数的形式相同, 即, :math:`\frac{dU_{\rm mem}(t)}{dt} \propto U_{\rm mem}(t)`, 
这意味着方程的解是带有时间常数 :math:`\tau`的指数函数。

假设神经元从某个值 :math:`U_{0}` 开始, 也没什么进一步的输入, 
即 :math:`I_{\rm in}(t)=0.` 其线性微分方程的解最终是：

.. math:: U_{\rm mem}(t) = U_0e^{-\frac{t}{\tau}}

整体解法如下所示：

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_4_RCmembrane.png?raw=true
        :align: center
        :width: 450

------------------------


**选读: 前向欧拉法解LIF神经元模型**

我们设法找到了 LIF 神经元的解析解, 但还不清楚这在神经网络中会有什么用处。
这一次, 让我们改用前向欧拉法来求解之前的线性常微分方程（ODE）。
这种方法看似繁琐, 但却能为我们提供 LIF 神经元的离散、递归形式。
一旦我们得到这种解法, 它就可以直接应用于神经网络。与之前一样, 描述 RC 电路的线性 ODE 为：

.. math:: \tau \frac{dU(t)}{dt} = -U(t) + RI_{\rm in}(t)

:math:`U(t)` 的下标从简省略。

首先让我们来在不求极限的情况下解这个导数
:math:`\Delta t \rightarrow 0`:

.. math:: \tau \frac{U(t+\Delta t)-U(t)}{\Delta t} = -U(t) + RI_{\rm in}(t)

对于足够小的 :math:`\Delta t`, 这给出了连续时间积分的一个足够好的近似值。
在下一时间段隔离膜, 得出

.. math:: U(t+\Delta t) = U(t) + \frac{\Delta t}{\tau}\big(-U(t) + RI_{\rm in}(t)\big)

下面的函数表示了这个等式：

::

    def leaky_integrate_neuron(U, time_step=1e-3, I=0, R=5e7, C=1e-10):
      tau = R*C
      U = U + (time_step/tau)*(-U + I*R)
      return U

默认参数设置为 :math:`R=50 M\Omega` 与
:math:`C=100pF` (i.e., :math:`\tau=5ms`). 这与真实的生物神经元相差无几。

现在循环这个函数, 每次迭代一个时间段。
膜电位初始化为 :math:`U=0.9 V`, 也假设没有任何注入电流 :math:`I_{\rm in}=0 A`.
在以毫秒 :math:`\Delta t=1\times 10^{-3}`\ s 为精度的条件下执行模拟。


::

    num_steps = 100
    U = 0.9
    U_trace = []  # keeps a record of U for plotting
    
    for step in range(num_steps):
      U_trace.append(U)
      U = leaky_integrate_neuron(U)  # solve next step of U
    
    plot_mem(U_trace, "Leaky Neuron Model")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/leaky1.png?raw=true
        :align: center
        :width: 300

这种指数衰减看起来与我们的预期相符！

3 Lapicque’s LIF Neuron Model
--------------------------------

`路易-拉皮克（Louis Lapicque）在 1907 年 <https://pubmed.ncbi.nlm.nih.gov/17968583/>`__ 
观察到神经膜和 RC 电路之间的这种相似性。他用短暂的电脉冲刺激青蛙的神经纤维, 
发现神经元膜可以近似为具有漏电的电容器。我们以他的名字命名 snnTorch 中的基本 LIF 神经元模型, 
以此向他的发现表示敬意。

Lapicque 模型中的大多数概念都可以应用到其他 LIF 神经元模型中。
现在是使用 snnTorch 模拟这个神经元的时候了。

3.1 Lapicque: 无人工刺激
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用下面的代码实现Lapicque的神经元。R & C改为更简单的值,
同时保持之前的时间常数 :math:`\tau=5\times10^{-3}`\ s.

::

    time_step = 1e-3
    R = 5
    C = 1e-3
    
    # leaky integrate and fire neuron, tau=5e-3
    lif1 = snn.Lapicque(R=R, C=C, time_step=time_step)

神经元模型现在储存在 ``lif1`` 中。要使用这个神经元:

**输入** 

* ``spk_in``:  :math:`I_{\rm in}` 中的每个元素依次作为输入传递 (现在是0) 
* ``mem``: 代表膜电位, 之前写作 :math:`U[t]`, 也作为输入传递。随便将其初始化为 :math:`U[0] = 0.9~V`.

**输出** 

* ``spk_out``: 下一个时间段的输出脉冲 :math:`S_{\rm out}[t+\Delta t]` (如果产生脉冲则为 ‘1’ ; 如果没有则为 ‘0’ ) 
* ``mem``: 下一个时间段的膜电位 :math:`U_{\rm mem}[t+\Delta t]` 

这些都必须是 ``torch.Tensor`` 类型。

::

    # Initialize membrane, input, and output
    mem = torch.ones(1) * 0.9  # U=0.9 at t=0
    cur_in = torch.zeros(num_steps)  # I=0 for all t 
    spk_out = torch.zeros(1)  # initialize output spikes

这些值只针对初始时间段 :math:`t=0`. 
要分析 ``mem`` 值随着时间的迭代, 我们可以创建一个 ``mem_rec`` 来记录这些值。

::

    # A list to store a recording of membrane potential
    mem_rec = [mem]

是时候运行模拟了! 在每个时间段,  ``mem`` 都会被更新并保存在 ``mem_rec`` 中:

::

    # pass updated value of mem and cur_in[step]=0 at every time step
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in[step], mem)
    
      # Store recordings of membrane potential
      mem_rec.append(mem)
    
    # convert the list of tensors into one tensor
    mem_rec = torch.stack(mem_rec)
    
    # pre-defined plotting function
    plot_mem(mem_rec, "Lapicque's Neuron Model Without Stimulus")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque.png?raw=true
        :align: center
        :width: 300

在没有任何输入刺激的情况下, 膜电位会随时间衰减。

3.2 Lapicque: 阶跃输入
~~~~~~~~~~~~~~~~~~~~~~~~~~

现在应用一个在 :math:`t=t_0` 时切换的阶跃电流 :math:`I_{\rm in}(t)`。
根据线性一阶微分方程：

.. math::  \tau \frac{dU_{\rm mem}}{dt} = -U_{\rm mem} + RI_{\rm in}(t),

一般解为：

.. math:: U_{\rm mem}=I_{\rm in}(t)R + [U_0 - I_{\rm in}(t)R]e^{-\frac{t}{\tau}}

如果膜电位初始化为 :math:`U_{\rm mem}(t=0) = 0 V`, 那么：

.. math:: U_{\rm mem}(t)=I_{\rm in}(t)R [1 - e^{-\frac{t}{\tau}}]

基于这个明确的时间依赖形式, 我们期望 :math:`U_{\rm mem}` 会指数级地
向 :math:`I_{\rm in}R` 收敛。让我们通过在 :math:`t_0 = 10ms` 时
触发电流脉冲来可视化这是什么样子。

::

    # 初始化输入电流脉冲
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.1), 0)  # 输入电流在 t=10 时打开
    
    # 初始化膜、输出和记录
    mem = torch.zeros(1)  # t=0 时膜电位为0
    spk_out = torch.zeros(1)  # 神经元需要一个地方顺序存储输出的脉冲
    mem_rec = [mem]

这一次, 新的 ``cur_in`` 值传递给了神经元：

::

    num_steps = 200
    
    # 在每个时间步骤中传递 mem 和 cur_in[step] 的更新值
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in[step], mem)
      mem_rec.append(mem)
    
    # 将张量列表合并成一个张量
    mem_rec = torch.stack(mem_rec)
    
    plot_step_current_response(cur_in, mem_rec, 10)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_step.png?raw=true
        :align: center
        :width: 450

当 :math:`t\rightarrow \infty` 时, 膜电位 :math:`U_{\rm mem}` 指数级地收敛到 :math:`I_{\rm in}R`：

::

    >>> print(f"计算得到的输入脉冲 [A] x 电阻 [Ω] 的值为: {cur_in[11]*lif1.R} V")
    >>> print(f"模拟得到的稳态膜电位值为: {mem_rec[200][0]} V")
    
    计算得到的输入脉冲 [A] x 电阻 [Ω] 的值为: 0.5 V
    模拟得到的稳态膜电位值为: 0.4999999403953552 V

足够接近！

3.3 Lapicque: 冲激输入
~~~~~~~~~~~~~~~~~~~~~~

那么如果阶跃输入在 :math:`t=30ms` 处被截断会怎么样呢？

::

    # 初始化电流脉冲、膜电位和输出
    cur_in1 = torch.cat((torch.zeros(10), torch.ones(20)*(0.1), torch.zeros(170)), 0)  # 输入在 t=10 开始, t=30 结束
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec1 = [mem]

::

    # 神经元模拟
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in1[step], mem)
      mem_rec1.append(mem)
    mem_rec1 = torch.stack(mem_rec1)
    
    plot_current_pulse_response(cur_in1, mem_rec1, "Lapicque神经元模型的输入脉冲", 
                                vline1=10, vline2=30)


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_pulse1.png?raw=true
        :align: center
        :width: 450

:math:`U_{\rm mem}` 就像对于阶跃输入一样上升, 
但现在它会像在我们的第一个模拟中那样以 :math:`\tau` 的时间常数下降。

让我们在半个时间内提供大致相同的电荷 :math:`Q = I \times t` 给电路。
这意味着必须稍微增加输入电流的幅度, 缩小时间窗口。

::

    # 增加电流脉冲的幅度；时间减半。
    cur_in2 = torch.cat((torch.zeros(10), torch.ones(10)*0.111, torch.zeros(180)), 0)  # 输入在 t=10 开始, t=20 结束
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec2 = [mem]
    
    # 神经元模拟
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in2[step], mem)
      mem_rec2.append(mem)
    mem_rec2 = torch.stack(mem_rec2)
    
    plot_current_pulse_response(cur_in2, mem_rec2, "Lapicque神经元模型的输入脉冲：x1/2 脉宽",
                                vline1=10, vline2=20)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_pulse2.png?raw=true
        :align: center
        :width: 450


让我们再来一次, 但使用更快的输入脉冲和更大的幅度：

::

    # 增加电流脉冲的幅度；时间缩短四分之一。
    cur_in3 = torch.cat((torch.zeros(10), torch.ones(5)*0.147, torch.zeros(185)), 0)  # 输入在 t=10 开始, t=15 结束
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec3 = [mem]
    
    # 神经元模拟
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in3[step], mem)
      mem_rec3.append(mem)
    mem_rec3 = torch.stack(mem_rec3)
    
    plot_current_pulse_response(cur_in3, mem_rec3, "Lapicque神经元模型的输入脉冲：x1/4 脉宽",
                                vline1=10, vline2=15)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_pulse3.png?raw=true
        :align: center
        :width: 450


现在将所有三个实验在同一图上进行比较：

::

    compare_plots(cur_in1, cur_in2, cur_in3, mem_rec1, mem_rec2, mem_rec3, 10, 15, 
                  20, 30, "Lapicque神经元模型的输入脉冲：不同的输入")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/compare_pulse.png?raw=true
        :align: center
        :width: 450

随着输入电流脉冲幅度的增加, 膜电位的上升时间加快。
当输入电流脉冲的宽度趋于无穷小时, :math:`T_W \rightarrow 0s`, 
膜电位将在几乎零上升时间内迅速上升：

::

    # 当前脉冲输入
    cur_in4 = torch.cat((torch.zeros(10), torch.ones(1)*0.5, torch.zeros(189)), 0)  # 输入仅在1个时间步上打开
    mem = torch.zeros(1) 
    spk_out = torch.zeros(1)
    mem_rec4 = [mem]
    
    # 神经元模拟
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in4[step], mem)
      mem_rec4.append(mem)
    mem_rec4 = torch.stack(mem_rec4)
    
    plot_current_pulse_response(cur_in4, mem_rec4, "Lapicque神经元模型的输入脉冲",
                                vline1=10, ylim_max1=0.6)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_spike.png?raw=true
        :align: center
        :width: 450


当前脉冲的宽度现在如此短, 实际上看起来像脉冲。
也就是说, 电荷在无限短的时间内传递, :math:`I_{\rm in}(t) = Q/t_0`, 
其中 :math:`t_0 \rightarrow 0`。
更正式地：

.. math:: I_{\rm in}(t) = Q \delta (t-t_0),

其中 :math:`\delta (t-t_0)` 是狄拉克-δ函数。从物理角度来看, 不可能“瞬间”存放电荷。
但积分 :math:`I_{\rm in}` 给出了一个在物理上有意义的结果, 
因为我们可以得到传递的电荷：

.. math:: 1 = \int^{t_0 + a}_{t_0 - a}\delta(t-t_0)dt

.. math:: f(t_0) = \int^{t_0 + a}_{t_0 - a}f(t)\delta(t-t_0)dt

在这里, 
:math:`f(t_0) = I_{\rm in}(t_0=10) = 0.5A \implies f(t) = Q = 0.5C`。

希望您对膜电位在静息状态下泄漏并积分输入电流有了一个很好的感觉。
这涵盖了神经元的“泄漏（Leaky）”和“累积（Integrate）”部分。那么如何引发“放电（Fire）”呢？

3.4 Lapicque: 放电
~~~~~~~~~~~~~~~~~~~~~~

到目前为止, 我们只看到神经元对输入的脉冲作出反应。
要使神经元在输出端产生并发出自己的脉冲, 必须将被动膜模型与阈值结合起来。

如果膜电位超过此阈值, 则会在被动膜模型外部生成一个电压脉冲。


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_4_spiking.png?raw=true
        :align: center
        :width: 400

修改之前的 ``leaky_integrate_neuron`` 函数以添加脉冲响应。

::

    # 用于说明的 R=5.1, C=5e-3
    def leaky_integrate_and_fire(mem, cur=0, threshold=1, time_step=1e-3, R=5.1, C=5e-3):
      tau_mem = R*C
      spk = (mem > threshold) # 如果膜超过阈值, 则 spk=1, 否则为0
      mem = mem + (time_step/tau_mem)*(-mem + cur*R)
      return mem, spk

设置 ``threshold=1``, 并应用阶跃电流以使该神经元发放脉冲。

::

    # 小步电流输入
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.2), 0)
    mem = torch.zeros(1)
    mem_rec = []
    spk_rec = []
    
    # 神经元模拟
    for step in range(num_steps):
      mem, spk = leaky_integrate_and_fire(mem, cur_in[step])
      mem_rec.append(mem)
      spk_rec.append(spk)
    
    # 将列表转换为张量
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, vline=109, ylim_max2=1.3, 
                     title="带无控制放电的LIF神经元模型")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lif_uncontrolled.png?raw=true
        :align: center
        :width: 450


哎呀 - 输出脉冲失控了！这是因为我们忘记了添加复位机制。
实际上, 每当神经元放电时, 膜电位都应该超极化（hyperpolarizes）回到其静息电位。

将此复位机制实施到我们的神经元中：

::

    # 带复位机制的LIF
    def leaky_integrate_and_fire(mem, cur=0, threshold=1, time_step=1e-3, R=5.1, C=5e-3):
      tau_mem = R*C
      spk = (mem > threshold)
      mem = mem + (time_step/tau_mem)*(-mem + cur*R) - spk*threshold  # 每次 spk=1 时, 减去阈值
      return mem, spk

::

    # 小步电流输入
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.2), 0)
    mem = torch.zeros(1)
    mem_rec = []
    spk_rec = []
    
    # 神经元模拟
    for step in range(num_steps):
      mem, spk = leaky_integrate_and_fire(mem, cur_in[step])
      mem_rec.append(mem)
      spk_rec.append(spk)
    
    # 将列表转换为张量
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, vline=109, ylim_max2=1.3, 
                     title="带复位的LIF神经元模型")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/reset_2.png?raw=true
        :align: center
        :width: 450

现在我们有了一个功能完善的LIF神经元模型, 好耶！

请注意, 如果 :math:`I_{\rm in}=0.2 A` 并且 :math:`R<5 \Omega`, 那么 :math:`I\times R < 1 V`。如果 ``threshold=1``, 则不会发生放电。请随意返回到上面, 更改值并测试。

与之前一样, 通过调用内置的snntorch中的Lapicque神经元模型, 所有这些代码都被压缩：

::

    # 使用snntorch创建与之前相同的神经元
    lif2 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3)
    
    >>> print(f"膜电位时间常数: {lif2.R * lif2.C:.3f}s")
    "膜电位时间常数: 0.025s"

::

    # 初始化输入和输出
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.2), 0)
    mem = torch.zeros(1)
    spk_out = torch.zeros(1) 
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # 在100个时间步骤内进行模拟运行。
    for step in range(num_steps):
      spk_out, mem = lif2(cur_in[step], mem)
      mem_rec.append(mem)
      spk_rec.append(spk_out)
    
    # 将列表转换为张量
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, vline=109, ylim_max2=1.3, 
                     title="带阶跃输入的Lapicque神经元模型")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_reset.png?raw=true
        :align: center
        :width: 450

膜电位呈指数上升, 然后达到阈值, 此时膜电位复位。我们大致可以看到这发生在 :math:`105ms < t_{\rm spk} < 115ms` 之间。出于好奇, 让我们看看脉冲记录实际包括什么内容：

::

    >>> print(spk_rec[105:115].view(-1))
    tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])

脉冲的缺失由 :math:`S_{\rm out}=0` 表示, 
而脉冲的发生由 :math:`S_{\rm out}=1` 表示。在这里, 
脉冲发生在 :math:`S_{\rm out}[t=109]=1`。
如果您想知道为什么每个这些条目都被存储为张量, 那是因为在未来的教程中, 
我们将模拟大规模的神经网络。每个条目将包含许多神经元的脉冲响应, 
并且可以将张量加载到GPU内存以加速训练过程。

如果增加 :math:`I_{\rm in}`, 则膜电位会更快地接近阈值 :math:`U_{\rm thr}`：

::

    # 初始化输入和输出
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.3), 0)  # 增加电流
    mem = torch.zeros(1)
    spk_out = torch.zeros(1) 
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # 神经元模拟
    for step in range(num_steps):
      spk_out, mem = lif2(cur_in[step], mem)
      mem_rec.append(mem)
      spk_rec.append(spk_out)
    
    # 将列表转换为张量
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, ylim_max2=1.3, 
                     title="带周期性放电的Lapicque神经元模型")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/periodic.png?raw=true
        :align: center
        :width: 450

通过降低阈值也可以诱发类似的放电频率增加。这需要初始化一个新的神经元模型, 但上面的代码块的其余部分完全相同：

::

    # 阈值减半的神经元
    lif3 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3, threshold=0.5)
    
    # 初始化输入和输出
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.3), 0) 
    mem = torch.zeros(1)
    spk_out = torch.zeros(1) 
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # 神经元模拟
    for step in range(num_steps):
      spk_out, mem = lif3(cur_in[step], mem)
      mem_rec.append(mem)
      spk_rec.append(spk_out)
    
    # 将列表转换为张量
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=0.5, ylim_max2=1.3, 
                     title="带更低阈值的Lapicque神经元模型")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/threshold.png?raw=true
        :align: center
        :width: 450

这是一个常数电流注入的情况。但在深度神经网络和生物大脑中, 
大多数神经元都将连接到其他神经元。它们更有可能接收脉冲, 而不是持续电流的注入。


3.5 Lapicque: 脉冲输入
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


让我们利用我们在 `教程（一） <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb>`_ 
中学到的一些技能, 并使用 ``snntorch.spikegen`` 模块创建一些随机生成的输入脉冲。

::

    # 创建一个1-D的随机脉冲序列。每个元素有40%的概率发放。
    spk_in = spikegen.rate_conv(torch.ones((num_steps)) * 0.40)

运行以下代码块以查看生成了多少脉冲。

::

    >>> print(f"在{len(spk_in)}个时间步骤中, 总共生成了{int(sum(spk_in))}个脉冲。")
    There are 85 total spikes out of 200 time steps.

::

    fig = plt.figure(facecolor="w", figsize=(8, 1))
    ax = fig.add_subplot(111)
    
    splt.raster(spk_in.reshape(num_steps, -1), ax, s=100, c="black", marker="|")
    plt.title("输入脉冲")
    plt.xlabel("时间步骤")
    plt.yticks([])
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/spikes.png?raw=true
        :align: center:
        :width: 400

::

    # 初始化输入和输出
    mem = torch.ones(1)*0.5
    spk_out = torch.zeros(1)
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # 神经元模拟
    for step in range(num_steps):
      spk_out, mem = lif3(spk_in[step], mem)
      spk_rec.append(spk_out)
      mem_rec.append(mem)
    
    # 将列表转换为张量
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_spk_mem_spk(spk_in, mem_rec, spk_out, "具有输入脉冲的Lapicque神经元模型")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/spk_mem_spk.png?raw=true
        :align: center:
        :width: 450


3.6 Lapicque: Reset Mechanisms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们已经从头开始实现了重置机制, 但让我们再深入一点。
膜电位的急剧下降促进了脉冲生成的减少, 这是有关大脑如何如此高效的一部分理论的补充。
在生物学上, 膜电位的这种下降被称为“去极化”。
在此之后, 很短的时间内很难引发神经元的另一个脉冲。
在这里, 我们使用重置机制来模拟去极化。

有两种实现重置机制的方法：

1. *减法重置*（默认）：每次生成脉冲时, 从膜电位中减去阈值；
2. *归零重置*：每次生成脉冲时, 将膜电位强制归零。
3. *不重置*：不采取任何措施, 让脉冲潜在地不受控制。

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_5_reset.png?raw=true
        :align: center
        :width: 400

实例化另一个神经元模型, 以演示如何在重置机制之间切换。默认情况下, 
snnTorch神经元模型使用 ``reset_mechanism = "subtract"``。
可以通过传递参数 ``reset_mechanism = "zero"`` 来明确覆盖默认设置。

::

    # 重置机制设置为“zero”的神经元
    lif4 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3, threshold=0.5, reset_mechanism="zero")
        
    # 初始化输入和输出
    spk_in = spikegen.rate_conv(torch.ones((num_steps)) * 0.40)
    mem = torch.ones(1)*0.5
    spk_out = torch.zeros(1)
    mem_rec0 = [mem]
    spk_rec0 = [spk_out]
        
    # 神经元模拟
    for step in range(num_steps):
      spk_out, mem = lif4(spk_in[step], mem)
      spk_rec0.append(spk_out)
      mem_rec0.append(mem)
        
    # 将列表转换为张量
    mem_rec0 = torch.stack(mem_rec0)
    spk_rec0 = torch.stack(spk_rec0)

    plot_reset_comparison(spk_in, mem_rec, spk_rec, mem_rec0, spk_rec0)



.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/comparison.png?raw=true
        :align: center
        :width: 550


请特别关注膜电位的演变, 尤其是在它达到阈值后的瞬间。
您可能会注意到, “重置为零”后, 膜电位被迫在每次脉冲后归零。

那么哪种方法更好？应用 ``"subtract"`` （重置机制的默认值）更不会丢失信息, 
因为它不会忽略膜电位超过阈值的程度。

另一方面, 采用 ``"zero"`` 的强制重置会促进稀疏性, 
并在专用的神经形态硬件上运行时可能降低功耗。您可以尝试使用这两种选项。

这涵盖了LIF神经元模型的基础知识！


Conclusion
---------------

实际上，我们可能不会用这个神经元模型来训练神经网络。
Lapicque LIF 模型增加了很多需要调整的超参数：:math:`R`, :math:`C`, :math:`\Delta t`, :math:`U_{\rm thr}`，
以及重置机制的选择。这一切都有点令人生畏。
因此， `下一个教程 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_ 将取消大部分超参数，
并引入更适合大规模深度学习的神经元模型。

如果你喜欢这个项目，请考虑在 GitHub 上给代码仓库点亮星星⭐，
因为这是支持它的最简单的、最好的方式。

参考文档在 `这里 <https://snntorch.readthedocs.io/en/latest/snntorch.html>`__.

更多阅读
---------------

-  `Check out the snnTorch GitHub project here. <https://github.com/jeshraghian/snntorch>`__
-  `snnTorch
   documentation <https://snntorch.readthedocs.io/en/latest/snntorch.html>`__
   of the Lapicque, Leaky, Synaptic, and Alpha models
-  `Neuronal Dynamics: From single neurons to networks and models of
   cognition <https://neuronaldynamics.epfl.ch/index.html>`__ by Wulfram
   Gerstner, Werner M. Kistler, Richard Naud and Liam Paninski.
-  `Theoretical Neuroscience: Computational and Mathematical Modeling of
   Neural
   Systems <https://mitpress.mit.edu/books/theoretical-neuroscience>`__
   by Laurence F. Abbott and Peter Dayan
