===============================================================================================
教程（七） - 使用 Tonic + snnTorch 的神经形态数据集
===============================================================================================

本教程出自 Jason K. Eshraghian (`www.ncg.ucsc.edu <https://www.ncg.ucsc.edu>`_)

 `English <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html#>`_ 

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_7_neuromorphic_datasets.ipynb

snnTorch 教程系列基于以下论文。如果您发现这些资源或代码对您的工作有用，请考虑引用以下来源：

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  本教程是不可编辑的静态版本。交互式可编辑版本可通过以下链接获取：
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_7_neuromorphic_datasets.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


引言
---------------

在这个教程中，你将：

* 学习如何使用 `Tonic <https://github.com/neuromorphs/tonic>`__ 加载神经形态数据集
* 使用缓存来加速数据加载
* 使用 `Neuromorphic-MNIST <https://tonic.readthedocs.io/en/latest/datasets.html#n-mnist>`__ 数据集训练 CSNN

安装 snnTorch 的最新 PyPi 发行版:

::

    pip install tonic 
    pip install snntorch

1. 使用 Tonic 加载神经形态数据集
-------------------------------------------------

感谢 `Tonic <https://github.com/neuromorphs/tonic>`__，从神经形态传感器加载数据集变得非常简单，它的工作方式很像 PyTorch vision。

让我们开始加载 MNIST 数据集的神经形态版本，
称为
`N-MNIST <https://tonic.readthedocs.io/en/latest/reference/datasets.html#n-mnist>`__。
我们可以查看一些原始事件，以了解我们正在处理的内容。

::

    import tonic
    
    dataset = tonic.datasets.NMNIST(save_to='./data', train=True)
    events, target = dataset[0]

::

    >>> print(events)
    [(10, 30, 937, 1) (33, 20, 1030, 1) (12, 27, 1052, 1) ...
    ( 7, 15, 302706, 1) (26, 11, 303852, 1) (11, 17, 305341, 1)]

每一行对应一个单独的事件，包括四个参数：(*x 坐标, y 坐标, 时间戳, 极性*)。

-  x 和 y 坐标对应于 :math:`34 \times 34` 网格中的一个地址。

-  事件的时间戳以微秒为单位记录。

-  极性指的是发生了上升脉冲（+1）还是下降脉冲（-1）；
   即亮度增加或亮度减少。

如果我们将这些事件随时间累积，并将区间作为
图像进行绘制，看起来像这样：

::

    >>> tonic.utils.plot_event_grid(events)

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial7/tonic_event_grid.png?raw=true
        :align: center
        :width: 450


1.1 转换
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

然而，神经网络并不直接接受事件列表作为输入。原始数据必须转换为适合的表示形式，如张量。我们可以选择一组转换在将数据输入到网络之前应用于我们的数据。神经形态相机传感器的时间分辨率为微秒，转换为密集表示时，最终会产生非常大的张量。这就是为什么我们使用 `ToFrame 转换 <https://tonic.readthedocs.io/en/latest/reference/transformations.html#frames>`__ 将事件划分为较少数量的帧，这会降低时间精度，但也使我们能够以密集格式处理它。

-  ``time_window=1000`` 将事件整合到 1000\ :math:`~\mu`\ s 的区间内

-  Denoise 移除孤立的、一次性事件。如果在 ``filter_time`` 微秒内，1像素邻域内没有发生事件，该事件将被过滤。 ``filter_time`` 较小会过滤更多事件。

::

    import tonic.transforms as transforms
    
    sensor_size = tonic.datasets.NMNIST.sensor_size
    
    # Denoise 移除孤立的、一次性事件
    # time_window
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), 
                                          transforms.ToFrame(sensor_size=sensor_size, 
                                                             time_window=1000)
                                         ])
    
    trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)



1.2 快速数据加载
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

原始数据存储在读取速度慢的格式中。为了加速数据加载，我们可以利用磁盘缓存和批处理。这意味着一旦文件从原始数据集中加载，它们将被写入磁盘。

由于事件记录的长度不同，我们将提供一个整理函数 ``tonic.collation.PadTensors()`` 来填充较短的记录，以确保批处理中的所有样本具有相同的尺寸。

::  

    from torch.utils.data import DataLoader
    from tonic import DiskCachedDataset


    cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
    cached_dataloader = DataLoader(cached_trainset)

    batch_size = 128
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())

::

    def load_sample_batched():
        events, target = next(iter(cached_dataloader))

::

    >>> %timeit -o -r 10 load_sample_batched()
    4.2 ms ± 119 µs 每循环（均值 ± 标准差，10 次运行，每次 100 循环）


通过使用磁盘缓存和支持多线程和批处理的 PyTorch 数据加载器，我们显著减少了加载时间。

如果你有大量的 RAM 可用，可以通过将数据缓存到主内存而不是磁盘来进一步加速数据加载：

::

    from tonic import MemoryCachedDataset

    cached_trainset = MemoryCachedDataset(trainset)


2. 使用从事件创建的帧训练我们的网络
-----------------------------------------------------------

现在让我们实际上使用 N-MNIST 分类任务来训练一个网络。我们首先定义我们的缓存包装器和数据加载器。在此过程中，我们还将对训练数据应用一些增强。我们从缓存数据集接收到的样本是帧，因此我们可以利用 PyTorch Vision 应用任何我们想要的随机转换。

::

    import torch
    import torchvision
    
    transform = tonic.transforms.Compose([torch.from_numpy,
                                          torchvision.transforms.RandomRotation([-10,10])])
    
    cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')
    
    # 测试集不应用增强
    cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')
    
    batch_size = 128
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
    testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))

现在一个小批量的维度是（时间步，批量大小，通道，高度，宽度）。时间步的数量将被设置为小批量中最长记录的数量，所有其他样本将被填充零以匹配它。

::

    >>> event_tensor, target = next(iter(trainloader))
    >>> print(event_tensor.shape)
    torch.Size([311, 128, 2, 34, 34])


2.1 定义我们的网络
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们将使用 snnTorch + PyTorch 来构建一个 CSNN，就像在前一个教程中一样。将使用的卷积网络架构是：12C5-MP2-32C5-MP2-800FC10

-  12C5 是一个带有 12 个滤波器的 5 :math:`\times` 5 卷积核
-  MP2 是一个 2 :math:`\times` 2 最大池化函数
-  800FC10 是一个将 800 个神经元映射到 10 个输出的全连接层


::

    import snntorch as snn
    from snntorch import surrogate
    from snntorch import functional as SF
    from snntorch import spikeplot as splt
    from snntorch import utils
    import torch.nn as nn

::

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    # 神经元和仿真参数
    spike_grad = surrogate.atan()
    beta = 0.5
    
    # 初始化网络
    net = nn.Sequential(nn.Conv2d(2, 12, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Conv2d(12, 32, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Flatten(),
                        nn.Linear(32*5*5, 10),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(device)

::

    # 这次，我们不会返回膜电位，因为我们不需要它
    
    def forward_pass(net, data):  
      spk_rec = []
      utils.reset(net)  # 重置网络中所有 LIF 神经元的隐藏状态
    
      for step in range(data.size(0)):  # data.size(0) = 时间步的数量
          spk_out, mem_out = net(data[step])
          spk_rec.append(spk_out)
      
      return torch.stack(spk_rec)


2.2 训练
~~~~~~~~~~~~~~~~~

在前一个教程中，交叉熵损失被应用到总脉冲计数上，以最大化正确类别的脉冲数量。

``snn.functional`` 模块的另一个选项是指定正确和错误类别的目标脉冲数量。
下面的方法使用 *均方误差脉冲计数损失*，旨在使正确类别的神经元 80% 的时间内产生脉冲，错误类别的神经元 20% 的时间内产生脉冲。鼓励错误的神经元发放可能是为了避免死神经元。

::

    optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

训练神经形态数据是昂贵的，因为它需要顺序地遍历许多时间步（N-MNIST 数据集中大约 300 个时间步）。
以下模拟将需要一些时间，所以我们只会在 50 次迭代中训练（大约是一个完整轮次的十分之一）。
如果你有更多时间，可以更改 ``num_iters``。由于我们在每次迭代时都在打印结果，结果将非常嘈杂，并且在我们开始看到任何改进之前也需要一些时间。

在我们自己的实验中，大约需要 20 次迭代才看到任何改善，经过 50 次迭代后，大约达到了 60% 的准确率。

   警告：以下模拟将需要一些时间。去泡一杯咖啡，或者十杯。

::

    num_epochs = 1
    num_iters = 50
    
    loss_hist = []
    acc_hist = []
    
    # 训练循环
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
            data = data.to(device)
            targets = targets.to(device)
    
            net.train()
            spk_rec = forward_pass(net, data)
            loss_val = loss_fn(spk_rec, targets)
    
            # 梯度计算 + 权重更新
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
    
            # 存储未来绘图的损失历史
            loss_hist.append(loss_val.item())
     
            print(f"轮次 {epoch}, 迭代 {i} \n训练损失: {loss_val.item():.2f}")
    
            acc = SF.accuracy_rate(spk_rec, targets) 
            acc_hist.append(acc)
            print(f"准确率: {acc * 100:.2f}%\n")

            # 训练循环在 50 次迭代后中断
            if i == num_iters:
              break

输出应该看起来像这样：

::

    轮次 0, 迭代 0 
    训练损失: 31.00
    准确率: 10.16%

    轮次 0, 迭代 1 
    训练损失: 30.58
    准确率: 13.28%

再过一段时间：

::

    轮次 0, 迭代 49 
    训练损失: 8.78
    准确率: 47.66%

    轮次 0, 迭代 50 
    训练损失: 8.43
    准确率: 56.25%



3. 结果
-------------

3.1 绘制测试准确率
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    import matplotlib.pyplot as plt
    
    # 绘制损失
    fig = plt.figure(facecolor="w")
    plt.plot(acc_hist)
    plt.title("训练集准确率")
    plt.xlabel("迭代")
    plt.ylabel("准确率")
    plt.show()


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial7/train_acc.png?raw=true
        :align: center
        :width: 450


3.2 脉冲计数器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对一批数据进行前向传递以获取脉冲记录。

::

    spk_rec = forward_pass(net, data)

更改 ``idx`` 可以让您索引到模拟小批量中的不同样本。使用 ``splt.spike_count`` 来探索几个不同样本的脉冲行为。生成以下动画将需要一些时间。

   注意：如果您在桌面上本地运行笔记本，请
   取消下面这行的注释，并修改路径到您的 ffmpeg.exe

::

    from IPython.display import HTML
    
    idx = 0
    
    fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
    print(f"目标标签是: {targets[idx]}")
    
    # plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'
    
    # 绘制脉冲计数直方图
    anim = splt.spike_count(spk_rec[:, idx].detach().cpu(), fig, ax, labels=labels, 
                            animate=True, interpolate=1)
    
    HTML(anim.to_html5_video())
    # anim.save("spike_bar.mp4")

::
    
    目标标签是: 3

.. raw:: html

    <center>
        <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial7/spike_counter.mp4?raw=true"></video>
    </center>

结论
------------

如果你坚持到了这里，那么恭喜你 —— 你有一位和尚级别的耐心。你现在也应该理解如何使用 Tonic 加载神经形态数据集，并使用 snnTorch 训练网络。

这里结束了深入教程系列。
您可以查看 `高级教程 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__ 
来学习更高级的技术，如引入长期时间动态到我们的 SNN 中，种群编码，或在智能处理单元上加速。

如果你喜欢这个项目，请考虑在 GitHub 上给仓库点赞⭐，这是支持它的最简单也是最好的方式。

额外资源
------------------------

-  `在这里查看 snnTorch 的 GitHub 项目。 <https://github.com/jeshraghian/snntorch>`__
-  `Tonic GitHub 项目可以在这里找到。 <https://github.com/neuromorphs/tonic>`__
-  N-MNIST 数据集最初发表在以下论文中：
   `Orchard, G.; Cohen, G.; Jayawant, A.; 和 Thakor, N. “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades”, Frontiers in Neuroscience, vol.9, no.437,
   2015年10月。 <https://www.frontiersin.org/articles/10.3389/fnins.2015.00437/full>`__
-  有关如何创建 N-MNIST 的更多信息，请参考
   `Garrick Orchard 的网站。 <https://www.garrickorchard.com/datasets/n-mnist>`__
