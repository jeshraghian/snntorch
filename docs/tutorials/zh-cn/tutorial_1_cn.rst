===========================
教程（一）- 脉冲编码
===========================

本教程出自 Jason K. Eshraghian (`www.ncg.ucsc.edu <https://www.ncg.ucsc.edu>`_)

 `English <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html#>`_ 

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb

snnTorch 教程系列基于以下论文。如果您发现这些资源或代码对您的工作有用，请考虑引用以下来源：

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. arXiv preprint arXiv:2109.12894,
    September 2021. <https://arxiv.org/abs/2109.12894>`_

.. note::
  本教程是不可编辑的静态版本。交互式可编辑版本可通过以下链接获取：
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_

在本教程中，你会学到如何使用snntorch来：

  * 将数据集（datasets）转化为脉冲数据集（spiking datasets）
  * 如何将它们可视化
  * 以及如何生成随机脉冲训练。


简介
-------------

当视网膜将光子转化为脉冲时，我们看到光。当挥发的分子转化为脉冲时，我们闻到气味。
当神经末梢将触觉压力转化为脉冲时，我们的大脑知道我们碰到了东西。
大脑以*脉冲*为全局货币进行交易。

因此，如果我们的最终目标是构建一个脉冲神经网络（SNN），以脉冲作为输入是理所当然的。
虽然使用非脉冲输入也很常见（如后续教程 3 所示），但数据编码的魅力部分来自三个 S：
脉冲（Spikes）、稀疏性（Sparsity）和静态抑制（Static suppression）。

1. **脉冲** : 
    (a-b) 生物神经元通过脉冲进行处理和交流，脉冲是振幅约为 100 mV 的
    电脉冲信号。(c) 许多神经元计算模型将这种电压脉冲简化为离散的单比特事件："1 "或 "0"。
    这比高精度值更容易用硬件表示。

2. **稀疏性** : 
    (c) 神经元的大部分时间都处于静息状态，在任何给定时间内，
    大部分激活都会沉默为零。稀疏向量/张量（包含大量零）不仅存储成本低，而且
    我们需要将稀疏激活与突触权重（synaptic weights）相乘。如果大多数值都乘以 "0"，
    那么我们就不需要从内存中读取许多网络参数。这意味着神经形态硬件可以非常高效。

3. **静态抑制 （又称事件驱动处理）**: 
    （d-e）感官外围（sensory periphery）只有当新信息需要处理时才会处理信息。
    在（e）中，每个像素都会对亮度（illuminance）的变化做出反应，
    因此大部分图像都被遮挡住了。传统的信号处理要求所有通道/像素都遵守全局采样/快门速度
    （global sampling/shutter rate），这就降低了感知的频率。现在，静态抑制通过
    阻断那些不变的输入，不仅提高了稀疏性和功耗效率，而且处理速度往往快得多。

   .. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/3s.png?raw=true
            :align: center
            :width: 800


在本教程中，我们将假设我们有一些非脉冲输入数据 （即MNIST数据集），
尝试使用几种不同的技术将这些数据编码为脉冲。让我们开始吧！

安装 snnTorch 的最新 PyPi 发行版：

::

    $ pip install snntorch

1. 配置MNIST数据集
-------------------------------

1.1. 配置库与环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    import snntorch as snn
    import torch

::

    # Training Parameters
    batch_size=128
    data_path='/tmp/data/mnist'
    num_classes = 10  # MNIST has 10 output classes
    
    # Torch Variables
    dtype = torch.float

1.2 下载数据集
~~~~~~~~~~~~~~~~~~~~

::

    from torchvision import datasets, transforms
    
    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
    
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)

如果上面的代码块报错，例如 MNIST 服务器崩了（MNIST servers are down），
取消注释以下代码即可。这是一个临时的数据下载器。

::

    # # temporary dataloader if MNIST service is unavailable
    # !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
    # !tar -zxvf MNIST.tar.gz
    
    # mnist_train = datasets.MNIST(root = './', train=True, download=True, transform=transform)

虽然已经下载了，但是在真正开始训练网络之前，我们不需要巨量数据。
``snntorch.utils`` 中有一些可以帮我们编辑数据集的函数。你可以应用 ``data_subset`` 函数来裁剪数据集，
裁剪后的大小基于变量 ``subset`` 的值. 比如说，当 ``subset=10`` , 包含60,000个数据的训练集将减少到 6,000 个。

::

    from snntorch import utils
    
    subset = 10
    mnist_train = utils.data_subset(mnist_train, subset)

::

    >>> print(f"The size of mnist_train is {len(mnist_train)}")
    The size of mnist_train is 6000


1.3 创建数据加载器
~~~~~~~~~~~~~~~~~~~~~~

上面创建的数据集对象会将数据加载到内存中，而
数据加载器将分批提供数据。PyTorch 中的数据加载器是一个
方便的接口，用于将数据传递到网络中。它们返回一个
被分成大小为 ``batch_size`` 的迭代器。

::

    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

2. 脉冲编码
-----------------

脉冲神经网络（SNN） 旨在利用时变数据（time-varing data）。 
然而, MNIST不是一个时变数据集。将MNIST与SNN一起使用有两种选择: 

1. 在每个时间段（time step）内，重复地将相同的训练样本
   :math:`\mathbf{X}\in\mathbb{R}^{m\times n}` 传递给神经网络。
   这就像把MNIST数据集转化为静态不变的视频。训练样本中的每个元素
    :math:`\mathbf{X}` 都可以取一个在0和1 之间归一化（Normalized）的高精度值: 
    :math:`X_{ij}\in [0, 1]`.
   

   .. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_1_static.png?raw=true
            :align: center
            :width: 800

2. 将输入转换成长度为 ``num_steps`` 的脉冲序列, 
    其每个特征/像素都有一个位于 :math:`X_{i,j} \in \{0, 1\}`之间的离散值. 
    在这种情况下, MNIST数据集被转化为了一个 与原始图像有关的 时变脉冲序列。

    .. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_2_spikeinput.png?raw=true
              :align: center
              :width: 800

第一种方法十分简单, 但是并没有充分利用SNN的时间动态（temporal dynamics）。
因此我们将详细展开第二种方法中数据到脉冲的转换（编码）。

``snntorch.spikegen`` 模块（脉冲生成模块）包含一系列可以简化这个转换过程的功能。
现在我们在 ``snntorch`` 库中有三个选择可用于脉冲编码:

1. 脉冲率编码（Rate coding）:
   `spikegen.rate <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html#snntorch.spikegen.rate>`__
2. 延迟编码（Latency coding）:
   `spikegen.latency <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html#snntorch.spikegen.latency>`__
3. 增量调制（Delta modulation）:
   `spikegen.delta <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html#snntorch.spikegen.delta>`__

这些方法有何不同？

1. *脉冲率编码* 用输入特征来确定 **脉冲频率**
2. *延迟编码* 利用输入特征来确定 **脉冲时长**
3. *增量调制* 利用输入特征的时态 **变化** 来生成脉冲

2.1 MNIST的脉冲率编码（Rate Coding）
~~~~~~~~~~~~~~~~~~~~~~~~

一个将输入数据转化为概率编码的示例如下。
每个归一化的输入特征 :math:`X_{ij}` 都被用作一个事件（脉冲）在任意时间段发生的概率，
其返回一个经过率编码的值 :math:`R_{ij}`. 这可以被视作伯努利试验（Bernoulli trial）:
:math:`R_{ij}\sim B(n,p)`, 其中实验的数量为 :math:`n=1`,
实验成功（产生脉冲）的概率为 :math:`p=X_{ij}`.
换句话说，脉冲发生的概率为：

.. math:: {\rm P}(R_{ij}=1) = X_{ij} = 1 - {\rm P}(R_{ij} = 0)

创建一个填充值为0.5的向量，并先应用上述伯努利实验的类比来进行概率编码：

::

    # Temporal Dynamics
    num_steps = 10
    
    # create vector filled with 0.5
    raw_vector = torch.ones(num_steps)*0.5
    
    # pass each sample through a Bernoulli trial
    rate_coded_vector = torch.bernoulli(raw_vector)

::
    >>> print(f"Converted vector: {rate_coded_vector}")
    Converted vector: tensor([1., 1., 1., 0., 0., 1., 1., 0., 1., 0.])
    
    >>> print(f"The output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time.")
    The output is spiking 60.00% of the time.

增加 ``raw_vector`` 的长度，再试一次:

::

    num_steps = 100
    
    # create vector filled with 0.5
    raw_vector = torch.ones(num_steps)*0.5
    
    # pass each sample through a Bernoulli trial
    rate_coded_vector = torch.bernoulli(raw_vector)
    >>> print(f"The output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time.")
    The output is spiking 48.00% of the time.
 
当 ``num_steps`` \ :math:`\rightarrow\infty`, 脉冲的比例（脉冲率）将接近原始值

对于一个MNIST图像, 此概率意味着其像素的值。一个白色像素对应100%的脉冲概率，
而一个黑色像素对应0%。也许下图能给你更多的关于概率编码的灵感。

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_3_spikeconv.png?raw=true
        :align: center
        :width: 1000

以类似的方式 ``spikegen.rate`` 可以代替上述对伯努利试验的类比，生成概率编码数据样本。
由于MNIST的每个样本只是一个图像, 我们可以用 ``num_steps`` 来随着时间的推移重复它。

::

    from snntorch import spikegen
    
    # Iterate through minibatches
    data = iter(train_loader)
    data_it, targets_it = next(data)
    
    # Spiking Data
    spike_data = spikegen.rate(data_it, num_steps=num_steps)

如果输入的值不在 :math:`[0,1]`这个区间, 它就不能表示概率了。
这种情况函数会自动将其裁剪回这个区间以确保其表示的仍然是概率。

输入数据的结构为
``[num_steps x batch_size x input dimensions]``:

::

    >>> print(spike_data.size())
    torch.Size([100, 128, 1, 28, 28])

2.2 可视化
~~~~~~~~~~~~~~~~~

2.2.1 动画
^^^^^^^^^^^^^^^

snnTorch中有一个让可视化过程变得非常简单的模块:
`snntorch.spikeplot <https://snntorch.readthedocs.io/en/latest/snntorch.spikeplot.html>`__


::

    import matplotlib.pyplot as plt
    import snntorch.spikeplot as splt
    from IPython.display import HTML

若要绘制一个数据样本，请从批次 （B） 维度索引到单个样本: ``spike_data``, ``[T x B x 1 x 28 x 28]``:

::

    spike_data_sample = spike_data[:, 0, 0]
    >>> print(spike_data_sample.size())
    torch.Size([100, 28, 28])

``spikeplot.animator`` 模块使得动画化2D数据非常简单, 
但是请注意: 如果你选择在本地运行这个函数, 你需要先安装ffmpeg用作视频格式的转换。
然后取消注释并编辑你ffmpeg.exe的路径。

::

    fig, ax = plt.subplots()
    anim = splt.animator(spike_data_sample, fig, ax)
    # plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'
    
    HTML(anim.to_html5_video())

.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/splt.animator.mp4?raw=true"></video>
  </center>

::

    # If you're feeling sentimental, you can save the animation: .gif, .mp4 etc.
    anim.save("spike_mnist_test.mp4")

可以按如下方式为关联的目标标签编制索引：

::

    >>> print(f"The corresponding target is: {targets_it[0]}")
    The corresponding target is: 7

MNIST具有灰度图像, 而其中的白色文本保证100%在每个时间段都会发生脉冲。
因此，让我们再次执行此操作，但减少脉冲频率。这可以通过设置参数 ``gain`` 来实现。
在这里, 我们将把脉冲频率降低到25%.

::

    spike_data = spikegen.rate(data_it, num_steps=num_steps, gain=0.25)
    
    spike_data_sample2 = spike_data[:, 0, 0]
    fig, ax = plt.subplots()
    anim = splt.animator(spike_data_sample2, fig, ax)
    HTML(anim.to_html5_video())

.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/splt.animator-25.mp4?raw=true"></video>
  </center>

::

    # Uncomment for optional save
    # anim.save("spike_mnist_test2.mp4")

现在平均一下一段时间内的脉冲，并重构图像。

::

    plt.figure(facecolor="w")
    plt.subplot(1,2,1)
    plt.imshow(spike_data_sample.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
    plt.axis('off')
    plt.title('Gain = 1')
    
    plt.subplot(1,2,2)
    plt.imshow(spike_data_sample2.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
    plt.axis('off')
    plt.title('Gain = 0.25')
    
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/gain.png?raw=true
        :align: center
        :width: 300

 ``gain=0.25`` 时生成的图片颜色明显比 ``gain=1`` 时生成的浅, 
 因为脉冲率降低了 :math:`\times 4`倍。

2.2.2 栅格图
^^^^^^^^^^^^^^^^^^

或者, 我们也可以选择生成输入样本的栅格图（raster plot）。这需要将样本重塑为2D张量, 
其中“时间”是第一维度。将样本传递到 ``spikeplot.raster`` 模块。

::

    # Reshape
    spike_data_sample2 = spike_data_sample2.reshape((num_steps, -1))
    
    # raster plot
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(spike_data_sample2, ax, s=1.5, c="black")
    
    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster.png?raw=true
        :align: center
        :width: 600

以下代码片段显示了如何索引到单个神经元中。 根据输入数据,
您可能需要尝试在0和784之间找到一个真正发生了脉冲的神经元。

::
    
    idx = 210  # index into 210th neuron

    fig = plt.figure(facecolor="w", figsize=(8, 1))
    ax = fig.add_subplot(111)
    
    splt.raster(spike_data_sample.reshape(num_steps, -1)[:, idx].unsqueeze(1), ax, s=100, c="black", marker="|")
    
    plt.title("Input Neuron")
    plt.xlabel("Time step")
    plt.yticks([])
    plt.show()


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster1.png?raw=true
        :align: center
        :width: 400

2.2.3 脉冲率编码总结
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

脉冲率编码实际上是一个十分有争议的主意。尽管大伙非常自信脉冲率编码有被应用在我们的周围感官，
但是大伙都不相信周围感官全都是基于脉冲率的。几个令人信服的里有包括：

-  **功耗:** 大自然会优化效率（能耗比）。完成任何类型的任务都需要几个脉冲，
    而每个脉冲都要消耗能量。事实上, `Olshausen和Field的 “What is the
   other 85% of V1
   doing?” <http://www.rctn.org/bruno/papers/V1-chapter.pdf>`__ 
    中证明脉冲率编码最多只能解释 初级视觉皮层 （V1） 中 15% 的神经元的活动。
    不太可能是脑内唯一的机制，因为大脑是出了名的 资源有限且效率高。

-  **响应时间:** 我们知道人类的反应时间大约是250毫秒。
    如果神经元的平均脉冲率在人脑中是10Hz的数量级, 那么在反应时间范围内, 
    人类只能处理约 2 个脉冲。

那么，如果速率码在能效或延迟方面不是最佳的，我们为什么还要使用它们呢？
这是因为即使我们的大脑不按速率处理数据，我们也相当确信我们的生物传感器会这样做。
功率/延迟方面的劣势被巨大的噪声鲁棒性所部分抵消：
即使有些脉冲无法产生也没关系，因为还会有更多的脉冲出现。

此外，你可能听说过  `Hebbian的理论 “neurons that
fire together, wire together” <https://doi.org/10.2307/1418888>`__。
如果出现大量的脉冲，这可能表明需要大量的学习。
在某些情况下, 训练神经元网络（SNN）具有挑战性,
而通过脉冲率编码鼓励更多神经元发射则是一种可能的解决方案。 

几乎可以肯定，脉冲率编码与大脑中的其他编码方案共同发挥作用。
我们将在接下来的章节中讨论这些其他编码机制。以上介绍了 spikegen.rate 函数。
更多信息请 `参阅此处的文档 <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html>`__.

2.3 MNIST的延迟编码（Latency Coding）
~~~~~~~~~~~~~~~~~~~~~~~~~~~

时序编码能捕捉神经元精确发射时间的信息；与依赖发射频率的脉冲率编码相比，
单个脉冲的意义要大得多。虽然这样更容易受到噪声的影响，
但也能将运行 SNN 算法的硬件功耗降低几个数量级。 

``spikegen.latency`` 函数允许每个输入在整个扫描时间内最多触发 **一次**。
接近 ``1`` 的特征会更早触发，接近 ``0`` 的特征会更晚触发。 
也就是说，在我们的 MNIST 案例中，亮像素会更早触发，暗像素会更晚触发。 

后续的代码块介绍了这一工作原理。如果你已经忘记了电路理论和/或数学知识，那也不用担心！
重要的是： **大** 输入意味着 **早** 触发脉冲; **小** 输入意味着
**晚** 触发脉冲.

------------------------

*选读: 延迟编码的推导*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

默认情况下，脉冲计时的计算方法是将输入特征视为注入 RC 电路的电流 :math:`I_{in}` 。
该电流会将电荷移动到电容器上，从而增加其两端的电压 :math:`V(t)`. 
我们假设存在一个触发电压, :math:`V_{thr}`, 一旦达到该电压，就会产生脉冲。
那么问题来了: *对于给定的输入电流（等同于输入特征），产生脉冲需要多长时间？*

从基尔霍夫电流定律开始, :math:`I_{in} = I_R + I_C`, 其余的推导将我们引向时间与输入之间的对数关系。

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_4_latencyrc.png?raw=true
        :align: center
        :width: 500

------------------------

以下函数利用上述推导结果将强度为
:math:`X_{ij}\in [0,1]` 的输入特征转化为延迟编码响应 :math:`L_{ij}`.

::

    def convert_to_time(data, tau=5, threshold=0.01):
      spike_time = tau * torch.log(data / (data - threshold))
      return spike_time 

现在我们可以用这个函数来可视化输入特征强度和其对应的脉冲时间的关系。

::

    raw_input = torch.arange(0, 5, 0.05) # tensor from 0 to 5
    spike_times = convert_to_time(raw_input)
    
    plt.plot(raw_input, spike_times)
    plt.xlabel('Input Value')
    plt.ylabel('Spike Time (s)')
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/spike_time.png?raw=true
        :align: center
        :width: 400

可以看出，数值越小，脉冲发生的时间越晚，且呈指数关系。 

向量 ``spike_times`` 包含脉冲触发的时间, 而不是包含脉冲本身（1 和 0）的稀疏张量。
在运行 SNN 仿真时，我们需要使用 1/0 表示来获得使用脉冲的所有优点。
整个过程可以使用 ``spikegen.latency`` 自动完成, 只需我们给 `data_it`传递一个来自MNIST数据集的迷你批次:

::

    spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)

此函数的参数包括：

-  ``tau``: 电路的 RC 时间常数。默认情况下，输入特征被视为注入 RC 电路的恒定电流。 ``tau`` 越大，激活（firing）速度越慢
-  ``threshold``: 膜电位点火阈值。低于该阈值的输入值没有闭式解（又称解析解），因为输入电流不足以驱动膜达到阈值。所有低于阈值的值都会被剪切并分配到最后一个时间段（time step）。

2.3.1 栅格图
^^^^^^^^^^^^^^^^^

::

    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")
    
    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()
    
    # optional save
    # fig.savefig('destination_path.png', format='png', dpi=300)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster2.png?raw=true
        :align: center
        :width: 600

要想理解这个栅格图，我们再次强调高强度的输入特征先激活（靠左），低强度的输入特征后激活（靠右）。

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_5_latencyraster.png?raw=true
        :align: center
        :width: 800

对数代码加上缺乏不同的输入值（即缺乏中间色调/灰度特征）导致图中两个区域出现明显的聚类（clustering）, 
换句话说就是咱这图片非黑即白，太极端了。亮像素在运行开始时激活，而暗像素则在运行结束时激活。 
我们可以增加 ``tau`` 来减慢脉冲时间, 或者通过设置可选参数 ``linear=True`` 来线性化脉冲时间。

::

    spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, linear=True)
    
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")
    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster3.png?raw=true
        :align: center
        :width: 600

现在，发射时间的分布更加均匀。这是通过将对数方程线性化来实现的，具体规则如下。
与 RC 模型不同，该模型没有物理基础。它只是更简单而已。

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_6_latencylinear.png?raw=true
        :align: center
        :width: 600

但请注意，所有激活都发生在前 5 个时间段内，而模拟范围是 100 个时间段。
这表明有大部分多余的时间段什么也没做。要解决这个问题，可以通过增加 ``tau`` 来减小时间常数，
或者设置可选参数  ``normalize=True`` 来跨越 ``num_steps`` 的整个范围。

::

    spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01,
                                  normalize=True, linear=True)
    
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")
    
    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster4.png?raw=true
        :align: center
        :width: 600

延迟编码对比脉冲率编码来说，一个主要的好处是其稀疏性。
如果神经元被限制在感兴趣的时间历程中最多被激活一次，那么这将促进低功耗运行。 

在上图所示的场景中，大部分脉冲发生在最后一个时间步长，此时输入特征低于阈值。
从某种意义上说, MNIST 样本的深色背景并不包含有用的信息。

我们可以通过设置 ``clip=True`` 来去除这些冗余特征。

::

    spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, 
                                  clip=True, normalize=True, linear=True)
    
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")
    
    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/raster5.png?raw=true
        :align: center
        :width: 600

可以看到右面那一排代表黑色像素的脉冲消失了。

2.3.2 动画化
^^^^^^^^^^^^^^^

跟之前跑的代码一模一样：

::

    >>> spike_data_sample = spike_data[:, 0, 0]
    >>> print(spike_data_sample.size())
    torch.Size([100, 28, 28])

::

    fig, ax = plt.subplots()
    anim = splt.animator(spike_data_sample, fig, ax)
    
    HTML(anim.to_html5_video())

.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/splt.animator2.mp4?raw=true"></video>
  </center>

这段动画在视频形式下显然要难看得多, 
但如果眼睛敏锐, 就能瞥见大部分脉冲出现的初始帧。索引到相应的目标值, 查看其值。

::

    # Save output: .gif, .mp4 etc.
    # anim.save("mnist_latency.gif")

::

    >>> print(targets_it[0])
    tensor(4, device='cuda:0')


这就是 ``spikegen.latency`` 函数。更多信息可以在 `这些文档 <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html>`__中找到。

2.4 增量调制
~~~~~~~~~~~~~~~~~~~~

有理论认为，视网膜具有适应性：只有当有新信息需要处理时，视网膜才会处理信息。
如果你的视野没有变化，那么你的感光细胞就不会那么容易点亮。 

也就是说: **生物是事件驱动的**。 神经元见风使舵。

一个有趣的例子是，一些研究人员毕生致力于设计受视网膜启发的图像传感器，例如 `Dynamic
Vision
Sensor <https://ieeexplore.ieee.org/abstract/document/7128412/>`__.
尽管 `附加的链接是十多年前的，但是这段视频中的工作 <https://www.youtube.com/watch?v=6eOM15U_t1M&ab_channel=TobiDelbruck>`__
却是非常超前的。

增量调制基于事件驱动脉冲。 The
``snntorch.delta`` 函数接受时间序列张量作为输入。它获取所有时间段中每个后续特征之间的差值。
默认情况下, 如果此差值为 *正* 且 *大于阈值* :math:`V_{thr}`, 则会产生一个脉冲:

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/1_2_7_delta.png?raw=true
        :align: center
        :width: 600

为了说明，让我们首先举出一个人为的例子，创建我们自己的输入张量。

::

    # Create a tensor with some fake time-series data
    data = torch.Tensor([0, 1, 0, 2, 8, -20, 20, -5, 0, 1, 0])
    
    # Plot the tensor
    plt.plot(data)
    
    plt.title("Some fake time-series data")
    plt.xlabel("Time step")
    plt.ylabel("Voltage (mV)")
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/fake_data.png?raw=true
      :align: center
      :width: 300

将上述张量传递进 ``spikegen.delta`` 函数, 附带一个随便选的阈值 ``threshold=4``:

::

    # Convert data
    spike_data = spikegen.delta(data, threshold=4)
    
    # Create fig, ax
    fig = plt.figure(facecolor="w", figsize=(8, 1))
    ax = fig.add_subplot(111)
    
    # Raster plot of delta converted data
    splt.raster(spike_data, ax, c="black")
    
    plt.title("Input Neuron")
    plt.xlabel("Time step")
    plt.yticks([])
    plt.xlim(0, len(data))
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/delta.png?raw=true
        :align: center
        :width: 400


可以看出，有三个时间段中 :math:`data[T]`
与 :math:`data[T+1]` i之间的差大于等于 :math:`V_{thr}=4`.
这意味着有三个 *正脉冲（on-spikes）*.

到 :math:`-20` 的大幅下降没有被脉冲捕捉到。 
但是我们可能也会关心负波动，在这种情况下，我们可以启用可选参数  ``off_spike=True``.

::

    # Convert data
    spike_data = spikegen.delta(data, threshold=4, off_spike=True)
    
    # Create fig, ax
    fig = plt.figure(facecolor="w", figsize=(8, 1))
    ax = fig.add_subplot(111)
    
    # Raster plot of delta converted data
    splt.raster(spike_data, ax, c="black")
    
    plt.title("Input Neuron")
    plt.xlabel("Time step")
    plt.yticks([])
    plt.xlim(0, len(data))
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/delta2.png?raw=true
        :align: center
        :width: 400

我们得到了更多的脉冲，但这貌似并不能展示正负。

将张量以数字的形式打印出来能让我们更加直观明了地看到代表着负脉冲的 ``-1``.

::

    >>> print(spike_data)
    tensor([ 0.,  0.,  0.,  0.,  1., -1.,  1., -1.,  1.,  0.,  0.])

虽然 ``spikegen.delta`` 只在一个假数据样本上演示过，
但它的真正用途是通过只为足够大的变化/事件生成尖峰来压缩时间序列数据。 

以上就是三个主要的尖峰转换功能！本教程没有详细介绍这三种转换技术的其他功能。
特别是，我们只研究了输入数据的编码，还没有考虑如何对目标进行编码，以及何时需要进行编码。
我们建议您参考 `相关文档进行深入了解 <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html>`__。

3. 脉冲的生成 (选读)
------------------------------

现在，如果我们实际上没有任何数据，该怎么办？假设我们只想从头开始随机生成一个脉冲序列。
``spikegen.rate`` 内部有一个嵌套函数, ``rate_conv``, 实际执行脉冲转换步骤。 

我们所要做的就是初始化一个随机生成的 ``torchTensor`` 并将其传入。

::

    # Create a random spike train
    spike_prob = torch.rand((num_steps, 28, 28), dtype=dtype) * 0.5
    spike_rand = spikegen.rate_conv(spike_prob)

3.1 动画化
~~~~~~~~~~~~~

::

    fig, ax = plt.subplots()
    anim = splt.animator(spike_rand, fig, ax)
    
    HTML(anim.to_html5_video())

.. raw:: html

  <center>
    <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/rand_spikes.mp4?raw=true"></video>
  </center>


::

    # Save output: .gif, .mp4 etc.
    # anim.save("random_spikes.gif")

3.2 栅格图
~~~~~~~~~~

::

    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(spike_rand[:, 0].view(num_steps, -1), ax, s=25, c="black")
    
    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial1/_static/rand_raster.png?raw=true
      :align: center
      :width: 600

结论
--------------

本文讨论了脉冲转换和生成。这种方法不仅适用于图像，还适用于单维和多维的张量。

如果你喜欢这个项目，请考虑在 GitHub 上的 点亮星星⭐。因为这是最简单、最好的支持方式。

作为参考,  `spikegen文档在这里 <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html>`__
, 还有 `spikeplot文档在这 <https://snntorch.readthedocs.io/en/latest/snntorch.spikeplot.html>`__.

`在下一篇教程中 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__, 
您将学习脉冲神经元的基础知识以及如何使用它们。

其他资源 
---------------------

* `在这里探索snnTorch项目 <https://github.com/jeshraghian/snntorch>`__