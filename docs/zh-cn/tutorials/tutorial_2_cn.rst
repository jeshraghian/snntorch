======================================================
教程（二） - Leaky Integrate-and-Fire（LIF）神经元
======================================================

本教程出自 Jason K. Eshraghian (`www.ncg.ucsc.edu <https://www.ncg.ucsc.edu>`_)

 `English <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html#>`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_2_lif_neuron.ipynb

snnTorch 教程系列基于以下论文。如果您发现这些资源或代码对您的工作有用，请考虑引用以下来源：

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

在本教程中，你将: 

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

神经元模型种类繁多，从精确的生物物理模型(比如说Hodgkin-Huxley模型)
到极其简单的人工神经元，它们遍及现代深度学习的所有方面。

**Hodgkin-Huxley Neuron Models**\ :math:`-`\ 虽然生物物理模型可以高度准确地
再现电生理结果，但其复杂性使其目前难以使用。

**Artificial Neuron Model**\ :math:`-`\ 人工神经元则是另一方面。
输入乘以相应的权重，然后通过激活函数。这种简化使深度学习研究人员在计算机视觉、
自然语言处理和许多其他机器学习领域的任务中取得了令人难以置信的成就。

**Leaky Integrate-and-Fire Neuron Models**\ :math:`-`\ 渗漏累加-激活（LIF）
神经元模型处于两者之间的中间位置。它接收加权输入的总和，与人工神经元非常相似。
但它并不直接将输入传递给激活函数，而是在一段时间内通过泄漏对输入进行累积，
这与 RC 电路非常相似。如果积分值超过阈值，那么 LIF 神经元就会发出电压脉冲。
LIF 神经元会提取出输出脉冲的形状和轮廓；它只是将其视为一个离散事件。
因此，信息并不是存储在脉冲中，而是存储在脉冲的时长（或频率）中。
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

2. 渗透累加-激活（LIF） 神经元模型
--------------------------------------------------

2.1 脉冲神经元: 灵感
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在我们的大脑中, 一个神经元可能与1000 - 10000个其他神经元相连。
如果一个神经元脉冲，所有下坡神经元都可能感受到。但是，是什么决定了
神经元是否会出现峰值呢？过去一个世纪的实验表明, 如果神经元在输入时受到
*足够的* 刺激, 那么它可能会变得兴奋，并发出自己的脉冲。

这种刺激从何而来？它可以来自：

* 外围感官, 
* 一种侵入性的电极人工地刺激神经元，或者在多数情况下，
* 来自突触前神经元。


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_2_intuition.png?raw=true
        :align: center
        :width: 600

考虑到这些脉冲电位是非常短的电位爆发，
不太可能所有输入尖峰电位都精确一致地到达神经元体。这表明有时间动态在
‘维持’ 输入脉冲, 就像是延迟.

2.2 被动细胞膜
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

与所有细胞一样，神经元周围也有一层薄薄的膜。这层膜是一层脂质双分子层，
将神经元内的导电生理盐水，与细胞外介质隔离开来。
在电学上，被绝缘体隔开的两种导电溶液就像一个电容器。

这层膜的另一个作用是控制进出细胞的物质 (比如说钠离子\ :math:`^+`). 
神经元膜通常不让离子渗透过去，这就阻止了离子进出神经元体。但是，
膜上有一些特定的通道，当电流注入神经元时，这些通道就会被触发打开。
这种电荷移动用电阻器来模拟。


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_3_passivemembrane.png?raw=true
        :align: center
        :width: 450

下面的代码块将从头开始推导LIF神经元的行为。如果你想跳过数学, 那请继续往下翻；
在推导之后, 我们将采用更实际的方法来理解LIF神经元动力学。

------------------------

**选读: LIF神经元模型的推导**

现在假设一些任意的时变电流 :math:`I_{\rm in}(t)` 注入了神经元, 
可能是通过电刺激，也可能是来自其他神经元。 电路中的总电流是守恒的，所以：

.. math:: I_{\rm in}(t) = I_{R} + I_{C}

根据欧姆定律，神经元内外测得的膜电位 :math:`U_{\rm mem}` 与通过电阻的电流成正比:

.. math:: I_{R}(t) = \frac{U_{\rm mem}(t)}{R}

电容是神经元上存储的电荷 :math:`Q` 与 :math:`U_{\rm mem}(t)`之间的比例常数:

.. math:: Q = CU_{\rm mem}(t)

电荷变化率给出通过电容的电流:

.. math:: \frac{dQ}{dt}=I_C(t) = C\frac{dU_{\rm mem}(t)}{dt}

因此:

.. math:: I_{\rm in}(t) = \frac{U_{\rm mem}(t)}{R} + C\frac{dU_{\rm mem}(t)}{dt}

.. math:: \implies RC \frac{dU_{\rm mem}(t)}{dt} = -U_{\rm mem}(t) + RI_{\rm in}(t)

等式右边的单位是电压 **\[Voltage]**. 在等式的左边, 
 :math:`\frac{dU_{\rm mem}(t)}{dt}` 这一项的单位是 
**\[Voltage/Time]**. 为了让等式的两边的单位相等 (都为电压), 
:math:`RC` 的单位必须是 **\[Time]**. 我们称 :math:`\tau = RC` 为电路的时间常数：

.. math:: \tau \frac{dU_{\rm mem}(t)}{dt} = -U_{\rm mem}(t) + RI_{\rm in}(t)

被动细胞膜此时成为了一个线性微分方程。

函数的导数要与原函数的形式相同, 即, :math:`\frac{dU_{\rm mem}(t)}{dt} \propto U_{\rm mem}(t)`, 
这意味着方程的解是带有时间常数 :math:`\tau`的指数函数。

假设神经元从某个值 :math:`U_{0}` 开始，也没什么进一步的输入, 
即 :math:`I_{\rm in}(t)=0.` 其线性微分方程的解最终是：

.. math:: U_{\rm mem}(t) = U_0e^{-\frac{t}{\tau}}

整体解法如下所示：

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_4_RCmembrane.png?raw=true
        :align: center
        :width: 450

------------------------


**选读: 前向欧拉法解LIF神经元模型**

我们设法找到了 LIF 神经元的解析解，但还不清楚这在神经网络中会有什么用处。
这一次，让我们改用前向欧拉法来求解之前的线性常微分方程（ODE）。
这种方法看似繁琐，但却能为我们提供 LIF 神经元的离散、递归形式。
一旦我们得到这种解法，它就可以直接应用于神经网络。与之前一样，描述 RC 电路的线性 ODE 为：

.. math:: \tau \frac{dU(t)}{dt} = -U(t) + RI_{\rm in}(t)

 :math:`U(t)` 的下标从简省略。

首先让我们来在不求极限的情况下解这个导数
:math:`\Delta t \rightarrow 0`:

.. math:: \tau \frac{U(t+\Delta t)-U(t)}{\Delta t} = -U(t) + RI_{\rm in}(t)

对于足够小的 :math:`\Delta t`, 这给出了连续时间积分的一个足够好的近似值。
在下一时间段隔离膜，得出

.. math:: U(t+\Delta t) = U(t) + \frac{\Delta t}{\tau}\big(-U(t) + RI_{\rm in}(t)\big)

下面的函数表示了这个等式：

::

    def leaky_integrate_neuron(U, time_step=1e-3, I=0, R=5e7, C=1e-10):
      tau = R*C
      U = U + (time_step/tau)*(-U + I*R)
      return U

默认参数设置为 :math:`R=50 M\Omega` 与
:math:`C=100pF` (i.e., :math:`\tau=5ms`). 这与真实的生物神经元相差无几。

现在循环这个函数，每次迭代一个时间段。
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
观察到神经膜和 RC 电路之间的这种相似性。他用短暂的电脉冲刺激青蛙的神经纤维，
发现神经元膜可以近似为具有漏电的电容器。我们以他的名字命名 snnTorch 中的基本 LIF 神经元模型，
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

是时候运行模拟了! 在每个时间段， ``mem`` 都会被更新并保存在 ``mem_rec``中:

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

The membrane potential decays over time in the absence of any input
stimuli.

3.2 Lapicque: Step Input
~~~~~~~~~~~~~~~~~~~~~~~~~~

Now apply a step current :math:`I_{\rm in}(t)` that switches on at
:math:`t=t_0`. Given the linear first-order differential equation:

.. math::  \tau \frac{dU_{\rm mem}}{dt} = -U_{\rm mem} + RI_{\rm in}(t),

the general solution is:

.. math:: U_{\rm mem}=I_{\rm in}(t)R + [U_0 - I_{\rm in}(t)R]e^{-\frac{t}{\tau}}

If the membrane potential is initialized to
:math:`U_{\rm mem}(t=0) = 0 V`, then:

.. math:: U_{\rm mem}(t)=I_{\rm in}(t)R [1 - e^{-\frac{t}{\tau}}]

Based on this explicit time-dependent form, we expect
:math:`U_{\rm mem}` to relax exponentially towards :math:`I_{\rm in}R`.
Let’s visualize what this looks like by triggering a current pulse of
:math:`I_{in}=100mA` at :math:`t_0 = 10ms`.

::

    # Initialize input current pulse
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.1), 0)  # input current turns on at t=10
    
    # Initialize membrane, output and recordings
    mem = torch.zeros(1)  # membrane potential of 0 at t=0
    spk_out = torch.zeros(1)  # neuron needs somewhere to sequentially dump its output spikes
    mem_rec = [mem]

This time, the new values of ``cur_in`` are passed to the neuron:

::

    num_steps = 200
    
    # pass updated value of mem and cur_in[step] at every time step
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in[step], mem)
      mem_rec.append(mem)
    
    # crunch -list- of tensors into one tensor
    mem_rec = torch.stack(mem_rec)
    
    plot_step_current_response(cur_in, mem_rec, 10)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_step.png?raw=true
        :align: center
        :width: 450

As :math:`t\rightarrow \infty`, the membrane potential
:math:`U_{\rm mem}` exponentially relaxes to :math:`I_{\rm in}R`:

::

    >>> print(f"The calculated value of input pulse [A] x resistance [Ω] is: {cur_in[11]*lif1.R} V")
    >>> print(f"The simulated value of steady-state membrane potential is: {mem_rec[200][0]} V")
    
    The calculated value of input pulse [A] x resistance [Ω] is: 0.5 V
    The simulated value of steady-state membrane potential is: 0.4999999403953552 V

Close enough!

3.3 Lapicque: Pulse Input
~~~~~~~~~~~~~~~~~~~~~~~~~~

Now what if the step input was clipped at :math:`t=30ms`?

::

    # Initialize current pulse, membrane and outputs
    cur_in1 = torch.cat((torch.zeros(10), torch.ones(20)*(0.1), torch.zeros(170)), 0)  # input turns on at t=10, off at t=30
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec1 = [mem]

::

    # neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in1[step], mem)
      mem_rec1.append(mem)
    mem_rec1 = torch.stack(mem_rec1)
    
    plot_current_pulse_response(cur_in1, mem_rec1, "Lapicque's Neuron Model With Input Pulse", 
                                vline1=10, vline2=30)


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_pulse1.png?raw=true
        :align: center
        :width: 450

:math:`U_{\rm mem}` rises just as it did for the step input, but now it
decays with a time constant of :math:`\tau` as in our first simulation.

Let’s deliver approximately the same amount of charge
:math:`Q = I \times t` to the circuit in half the time. This means the
input current amplitude must be increased by a little, and the
time window must be decreased.

::

    # Increase amplitude of current pulse; half the time.
    cur_in2 = torch.cat((torch.zeros(10), torch.ones(10)*0.111, torch.zeros(180)), 0)  # input turns on at t=10, off at t=20
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec2 = [mem]
    
    # neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in2[step], mem)
      mem_rec2.append(mem)
    mem_rec2 = torch.stack(mem_rec2)
    
    plot_current_pulse_response(cur_in2, mem_rec2, "Lapicque's Neuron Model With Input Pulse: x1/2 pulse width",
                                vline1=10, vline2=20)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_pulse2.png?raw=true
        :align: center
        :width: 450


Let’s do that again, but with an even faster input pulse and higher
amplitude:

::

    # Increase amplitude of current pulse; quarter the time.
    cur_in3 = torch.cat((torch.zeros(10), torch.ones(5)*0.147, torch.zeros(185)), 0)  # input turns on at t=10, off at t=15
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec3 = [mem]
    
    # neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in3[step], mem)
      mem_rec3.append(mem)
    mem_rec3 = torch.stack(mem_rec3)
    
    plot_current_pulse_response(cur_in3, mem_rec3, "Lapicque's Neuron Model With Input Pulse: x1/4 pulse width",
                                vline1=10, vline2=15)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_pulse3.png?raw=true
        :align: center
        :width: 450


Now compare all three experiments on the same plot:


::

    compare_plots(cur_in1, cur_in2, cur_in3, mem_rec1, mem_rec2, mem_rec3, 10, 15, 
                  20, 30, "Lapicque's Neuron Model With Input Pulse: Varying inputs")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/compare_pulse.png?raw=true
        :align: center
        :width: 450

As the input current pulse amplitude increases, the rise time of the
membrane potential speeds up. In the limit of the input current pulse
width becoming infinitesimally small, :math:`T_W \rightarrow 0s`, the
membrane potential will jump straight up in virtually zero rise time:

::

    # Current spike input
    cur_in4 = torch.cat((torch.zeros(10), torch.ones(1)*0.5, torch.zeros(189)), 0)  # input only on for 1 time step
    mem = torch.zeros(1) 
    spk_out = torch.zeros(1)
    mem_rec4 = [mem]
    
    # neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif1(cur_in4[step], mem)
      mem_rec4.append(mem)
    mem_rec4 = torch.stack(mem_rec4)
    
    plot_current_pulse_response(cur_in4, mem_rec4, "Lapicque's Neuron Model With Input Spike", 
                                vline1=10, ylim_max1=0.6)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_spike.png?raw=true
        :align: center
        :width: 450


The current pulse width is now so short, it effectively looks like a
spike. That is to say, charge is delivered in an infinitely short period
of time, :math:`I_{\rm in}(t) = Q/t_0` where :math:`t_0 \rightarrow 0`.
More formally:

.. math:: I_{\rm in}(t) = Q \delta (t-t_0),

where :math:`\delta (t-t_0)` is the Dirac-Delta function. Physically, it
is impossible to ‘instantaneously’ deposit charge. But integrating
:math:`I_{\rm in}` gives a result that makes physical sense, as we can
obtain the charge delivered:

.. math:: 1 = \int^{t_0 + a}_{t_0 - a}\delta(t-t_0)dt

.. math:: f(t_0) = \int^{t_0 + a}_{t_0 - a}f(t)\delta(t-t_0)dt

Here,
:math:`f(t_0) = I_{\rm in}(t_0=10) = 0.5A \implies f(t) = Q = 0.5C`.

Hopefully you have a good feel of how the membrane potential leaks at
rest, and integrates the input current. That covers the ‘leaky’ and
‘integrate’ part of the neuron. How about the fire?

3.4 Lapicque: Firing
~~~~~~~~~~~~~~~~~~~~~~

So far, we have only seen how a neuron will react to spikes at the
input. For a neuron to generate and emit its own spikes at the output,
the passive membrane model must be combined with a threshold.

If the membrane potential exceeds this threshold, then a voltage spike
will be generated, external to the passive membrane model.


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_4_spiking.png?raw=true
        :align: center
        :width: 400

Modify the ``leaky_integrate_neuron`` function from before to add
a spike response.

::

    # R=5.1, C=5e-3 for illustrative purposes
    def leaky_integrate_and_fire(mem, cur=0, threshold=1, time_step=1e-3, R=5.1, C=5e-3):
      tau_mem = R*C
      spk = (mem > threshold) # if membrane exceeds threshold, spk=1, else, 0
      mem = mem + (time_step/tau_mem)*(-mem + cur*R)
      return mem, spk

Set ``threshold=1``, and apply a step current to get this neuron
spiking.

::

    # Small step current input
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.2), 0)
    mem = torch.zeros(1)
    mem_rec = []
    spk_rec = []
    
    # neuron simulation
    for step in range(num_steps):
      mem, spk = leaky_integrate_and_fire(mem, cur_in[step])
      mem_rec.append(mem)
      spk_rec.append(spk)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, vline=109, ylim_max2=1.3, 
                     title="LIF Neuron Model With Uncontrolled Spiking")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lif_uncontrolled.png?raw=true
        :align: center
        :width: 450


Oops - the output spikes have gone out of control! This is
because we forgot to add a reset mechanism. In reality, each time a
neuron fires, the membrane potential hyperpolarizes back to its resting
potential.

Implementing this reset mechanism into our neuron:

::

    # LIF w/Reset mechanism
    def leaky_integrate_and_fire(mem, cur=0, threshold=1, time_step=1e-3, R=5.1, C=5e-3):
      tau_mem = R*C
      spk = (mem > threshold)
      mem = mem + (time_step/tau_mem)*(-mem + cur*R) - spk*threshold  # every time spk=1, subtract the threhsold
      return mem, spk

::

    # Small step current input
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.2), 0)
    mem = torch.zeros(1)
    mem_rec = []
    spk_rec = []
    
    # neuron simulation
    for step in range(num_steps):
      mem, spk = leaky_integrate_and_fire(mem, cur_in[step])
      mem_rec.append(mem)
      spk_rec.append(spk)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, vline=109, ylim_max2=1.3, 
                     title="LIF Neuron Model With Reset")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/reset_2.png?raw=true
        :align: center
        :width: 450

Bam. We now have a functional leaky integrate-and-fire neuron model!

Note that if :math:`I_{\rm in}=0.2 A` and :math:`R<5 \Omega`, then
:math:`I\times R < 1 V`. If ``threshold=1``, then no spiking would
occur. Feel free to go back up, change the values, and test it out.

As before, all of that code is condensed by calling the built-in Lapicque neuron model from snnTorch:

::

    # Create the same neuron as before using snnTorch
    lif2 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3)
    
    >>> print(f"Membrane potential time constant: {lif2.R * lif2.C:.3f}s")
    "Membrane potential time constant: 0.025s"

::

    # Initialize inputs and outputs
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.2), 0)
    mem = torch.zeros(1)
    spk_out = torch.zeros(1) 
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # Simulation run across 100 time steps.
    for step in range(num_steps):
      spk_out, mem = lif2(cur_in[step], mem)
      mem_rec.append(mem)
      spk_rec.append(spk_out)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, vline=109, ylim_max2=1.3, 
                     title="Lapicque Neuron Model With Step Input")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/lapicque_reset.png?raw=true
        :align: center
        :width: 450

The membrane potential exponentially rises and then hits the threshold,
at which point it resets. We can roughly see this occurs between
:math:`105ms < t_{\rm spk} < 115ms`. As a matter of curiousity, let’s
see what the spike recording actually consists of:

::

    >>> print(spk_rec[105:115].view(-1))
    tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])

The absence of a spike is represented by :math:`S_{\rm out}=0`, and the
occurrence of a spike is :math:`S_{\rm out}=1`. Here, the spike occurs
at :math:`S_{\rm out}[t=109]=1`. If you are wondering why each of these entries is stored as a tensor, it
is because in future tutorials we will simulate large scale neural
networks. Each entry will contain the spike responses of many neurons,
and tensors can be loaded into GPU memory to speed up the training
process.

If :math:`I_{\rm in}` is increased, then the membrane potential
approaches the threshold :math:`U_{\rm thr}` faster:

::

    # Initialize inputs and outputs
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.3), 0)  # increased current
    mem = torch.zeros(1)
    spk_out = torch.zeros(1) 
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif2(cur_in[step], mem)
      mem_rec.append(mem)
      spk_rec.append(spk_out)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, ylim_max2=1.3, 
                     title="Lapicque Neuron Model With Periodic Firing")

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/periodic.png?raw=true
        :align: center
        :width: 450

A similar increase in firing frequency can also be induced by decreasing
the threshold. This requires initializing a new neuron model, but the
rest of the code block is the exact same as above:

::

    # neuron with halved threshold
    lif3 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3, threshold=0.5)
    
    # Initialize inputs and outputs
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.3), 0) 
    mem = torch.zeros(1)
    spk_out = torch.zeros(1) 
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # Neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif3(cur_in[step], mem)
      mem_rec.append(mem)
      spk_rec.append(spk_out)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=0.5, ylim_max2=1.3, 
                     title="Lapicque Neuron Model With Lower Threshold")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/threshold.png?raw=true
        :align: center
        :width: 450

That’s what happens for a constant current injection. But in both deep
neural networks and in the biological brain, most neurons will be
connected to other neurons. They are more likely to receive spikes,
rather than injections of constant current.

3.5 Lapicque: Spike Inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s harness some of the skills we learnt in `Tutorial
1 <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb>`__,
and use the ``snntorch.spikegen`` module to create some randomly
generated input spikes.

::

    # Create a 1-D random spike train. Each element has a probability of 40% of firing.
    spk_in = spikegen.rate_conv(torch.ones((num_steps)) * 0.40)

Run the following code block to see how many spikes have been generated.

::

    >>> print(f"There are {int(sum(spk_in))} total spikes out of {len(spk_in)} time steps.")
    There are 85 total spikes out of 200 time steps.

::

    fig = plt.figure(facecolor="w", figsize=(8, 1))
    ax = fig.add_subplot(111)
    
    splt.raster(spk_in.reshape(num_steps, -1), ax, s=100, c="black", marker="|")
    plt.title("Input Spikes")
    plt.xlabel("Time step")
    plt.yticks([])
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/spikes.png?raw=true
        :align: center
        :width: 400

::

    # Initialize inputs and outputs
    mem = torch.ones(1)*0.5
    spk_out = torch.zeros(1)
    mem_rec = [mem]
    spk_rec = [spk_out]
    
    # Neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif3(spk_in[step], mem)
      spk_rec.append(spk_out)
      mem_rec.append(mem)
    
    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)
    
    plot_spk_mem_spk(spk_in, mem_rec, spk_out, "Lapicque's Neuron Model With Input Spikes")


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/spk_mem_spk.png?raw=true
        :align: center
        :width: 450

3.6 Lapicque: Reset Mechanisms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We already implemented a reset mechanism from scratch, but let’s dive a
little deeper. This sharp drop of membrane potential promotes a
reduction of spike generation, which supplements part of the theory on
how brains are so power efficient. Biologically, this drop of membrane
potential is known as ‘hyperpolarization’. Following that, it is
momentarily more difficult to elicit another spike from the neuron.
Here, we use a reset mechanism to model hyperpolarization.

There are two ways to implement the reset mechanism:

1. *reset by subtraction* (default) :math:`-` subtract the threshold
   from the membrane potential each time a spike is generated;
2. *reset to zero* :math:`-` force the membrane potential to zero each
   time a spike is generated.
3. *no reset* :math:`-` do nothing, and let the firing go potentially uncontrolled.

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/2_5_reset.png?raw=true
        :align: center
        :width: 400

Instantiate another neuron model to demonstrate how to alternate
between reset mechanisms. By default, snnTorch neuron models use ``reset_mechanism = "subtract"``.
This can be explicitly overridden by passing the argument
``reset_mechanism =  "zero"``.

::

    # Neuron with reset_mechanism set to "zero"
    lif4 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3, threshold=0.5, reset_mechanism="zero")
    
    # Initialize inputs and outputs
    spk_in = spikegen.rate_conv(torch.ones((num_steps)) * 0.40)
    mem = torch.ones(1)*0.5
    spk_out = torch.zeros(1)
    mem_rec0 = [mem]
    spk_rec0 = [spk_out]
    
    # Neuron simulation
    for step in range(num_steps):
      spk_out, mem = lif4(spk_in[step], mem)
      spk_rec0.append(spk_out)
      mem_rec0.append(mem)
    
    # convert lists to tensors
    mem_rec0 = torch.stack(mem_rec0)
    spk_rec0 = torch.stack(spk_rec0)
    
    plot_reset_comparison(spk_in, mem_rec, spk_rec, mem_rec0, spk_rec0)


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial2/_static/comparison.png?raw=true
        :align: center
        :width: 550

Pay close attention to the evolution of the membrane potential,
especially in the moments after it reaches the threshold. You may notice
that for “Reset to Zero”, the membrane potential is forced back to zero
after each spike.

So which one is better? Applying ``"subtract"`` (the default value in
``reset_mechanism``) is less lossy, because it does not ignore how much
the membrane exceeds the threshold by.

On the other hand, applying a hard reset with ``"zero"`` promotes
sparsity and potentially less power consumption when running on
dedicated neuromorphic hardware. Both options are available for you to
experiment with.

That covers the basics of a LIF neuron model!

Conclusion
---------------

In practice, we probably wouldn’t use this neuron model to train a
neural network. The Lapicque LIF model has added a lot of
hyperparameters to tune: :math:`R`, :math:`C`, :math:`\Delta t`,
:math:`U_{\rm thr}`, and the choice of reset mechanism. It’s all a
little bit daunting. So the `next tutorial <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_ will eliminate most of these
hyperparameters, and introduce a neuron model that is better suited for
large-scale deep learning.

If you like this project, please consider starring ⭐ the repo on GitHub as it is the easiest and best way to support it.

For reference, the documentation `can be found
here <https://snntorch.readthedocs.io/en/latest/snntorch.html>`__.

Further Reading
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
