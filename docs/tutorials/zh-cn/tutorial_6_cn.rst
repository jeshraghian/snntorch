===============================================================================================
教程（六） - 卷积 SNN 中的替代梯度下降
===============================================================================================

本教程出自 Jason K. Eshraghian (`www.ncg.ucsc.edu <https://www.ncg.ucsc.edu>`_)

 `English <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html#>`_ 

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_5_FCN.ipynb

snnTorch 教程系列基于以下论文。如果您发现这些资源或代码对您的工作有用，请考虑引用以下来源：

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  本教程是不可编辑的静态版本。交互式可编辑版本可通过以下链接获取：
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_5_FCN.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


简介
--------------

在这个教程中，你将：

* 学习如何修改替代梯度下降来克服死神经元（dead neuron）问题
* 构建并训练一个卷积脉冲神经网络
* 使用序列容器 ``nn.Sequential`` 简化模型构建

..

   这个教程的一部分受到了Friedemann Zenke在SNNs上广泛工作的启发。
   在 `这里 <https://github.com/fzenke/spytorch>`__查看他关于替代梯度的仓库，
   以及我的一篇最喜欢的论文：E. O. Neftci, H. Mostafa, F. Zenke, `在脉冲神经网络中的替代梯度学习：将基于梯度的优化带入脉冲神经
   网络。 <https://ieeexplore.ieee.org/document/8891809>`__ IEEE
   信号处理杂志 36, 51–63。

在教程的最后，我们将使用 MNIST 数据集训练一个卷积脉冲神经网络（CSNN）来进行图像分类。
背景理论基于 `教程 2, 4 和
5 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__，
如果需要复习，可以回顾一下。

安装 snnTorch 的最新 PyPi 发行版:

::

    $ pip install snntorch

::

    # imports
    import snntorch as snn
    from snntorch import surrogate
    from snntorch import backprop
    from snntorch import functional as SF
    from snntorch import utils
    from snntorch import spikeplot as splt
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torch.nn.functional as F
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

1. 替代梯度下降
--------------------------------

`教程（五）<https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_ 提出了 **死神经元问题** 。这是因为脉冲的不可微性引起的：

.. math:: S[t] = \Theta(U[t] - U_{\rm thr}) \tag{1}

.. math:: \frac{\partial S}{\partial U} = \delta(U - U_{\rm thr}) \in \{0, \infty\} \tag{2}

其中 :math:`\Theta(\cdot)` 是 Heaviside 阶跃函数，:math:`\delta(\cdot)` 是 Dirac-Delta 函数。我们之前使用了反向传播过程中替代的 *ArcTangent* 函数来克服这个问题。

其他常见的平滑函数包括 sigmoid 函数或快速 sigmoid 函数。sigmoid 函数也必须移动，使其以阈值 :math:`U_{\rm thr}` 为中心。定义膜电位超驱动为 :math:`U_{OD} = U - U_{\rm thr}`：

.. math:: \tilde{S} = \frac{U_{OD}}{1+k|U_{OD}|} \tag{3}

.. math:: \frac{\partial \tilde{S}}{\partial U} = \frac{1}{(k|U_{OD}|+1)^2}\tag{4}

其中 :math:`k` 调节替代函数的平滑度，被视为一个超参数。随着 :math:`k` 的增加，近似值趋向于 :math:`(2)` 中的原始导数：

.. math:: \frac{\partial \tilde{S}}{\partial U} \Bigg|_{k \rightarrow \infty} = \delta(U-U_{\rm thr})


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial6/surrogate.png?raw=true
        :align: center
        :width: 800


总结一下：

-  **前向传递**

   -  使用 :math:`(1)` 中移位的 Heaviside 函数确定 :math:`S`
   -  存储 :math:`U` 以便在反向传递期间使用

-  **反向传递**

   -  将 :math:`U` 传入 :math:`(4)` 来计算导数项

就像在 `教程（五）<https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`_ 中使用的 *ArcTangent* 方法一样，
快速 sigmoid 函数的梯度可以在泄漏积分-发放（LIF）神经元模型中替代 Dirac-Delta 函数：

::

    # 泄漏神经元模型，用自定义函数覆盖反向传递
    class LeakySigmoidSurrogate(nn.Module):
      def __init__(self, beta, threshold=1.0, k=25):

          # Leaky_Surrogate 在前一个教程中定义，这里不使用
          super(Leaky_Surrogate, self).__init__()
    
          # 初始化衰减率 beta 和阈值
          self.beta = beta
          self.threshold = threshold
          self.surrogate_func = self.FastSigmoid.apply
      
      # forward 函数在每次调用 Leaky 时被调用
      def forward(self, input_, mem):
        spk = self.surrogate_func((mem-self.threshold))  # 调用 Heaviside 函数
        reset = (spk - self.threshold).detach()
        mem = self.beta * mem + input_ - reset
        return spk, mem
    
      # 前向传递：Heaviside 函数
      # 反向传递：用快速 sigmoid 的梯度覆盖 Dirac Delta
      @staticmethod
      class FastSigmoid(torch.autograd.Function):  
        @staticmethod
        def forward(ctx, mem, k=25):
            ctx.save_for_backward(mem) # 存储膜电位以用于反向传递
            ctx.k = k
            out = (mem > 0).float() # 前向传递的 Heaviside 函数：Eq(1)
            return out
    
        @staticmethod
        def backward(ctx, grad_output): 
            (mem,) = ctx.saved_tensors  # 检索膜电位
            grad_input = grad_output.clone()
            grad = grad_input / (ctx.k * torch.abs(mem) + 1.0) ** 2  # 反向传递的快速 sigmoid 梯度：Eq(4)
            return grad, None

更好的是，所有这些可以通过使用 snnTorch 内置模块
``snn.surrogate`` 来简化，其中 :math:`(4)` 中的 :math:`k` 被表示为 ``slope``。替代梯度被作为参数传递到 ``spike_grad`` 中：

::

    spike_grad = surrogate.fast_sigmoid(slope=25)
    beta = 0.5
    
    lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

要探索其他可用的替代梯度函数，请 `查看文档 <https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html>`__


2. 设置 CSNN
------------------------

2.1 数据加载器
~~~~~~~~~~~~~~~~~

::

    # 数据加载器参数
    batch_size = 128
    data_path='/tmp/data/mnist'
    
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

::

    # 定义转换
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
    
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

2.2 定义网络
~~~~~~~~~~~~~~~~~~~~~~~~~

将要使用的卷积网络结构是：
12C5-MP2-64C5-MP2-1024FC10

-  12C5 是一个带有 12 个滤波器的 5 :math:`\times` 5 卷积核
-  MP2 是一个 2 :math:`\times` 2 最大池化函数
-  1024FC10 是一个将 1,024 个神经元映射到 10 个输出的全连接层

::

    # 神经元和仿真参数
    spike_grad = surrogate.fast_sigmoid(slope=25)
    beta = 0.5
    num_steps = 50

::

    # 定义网络
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
    
            # 初始化层
            self.conv1 = nn.Conv2d(1, 12, 5)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.conv2 = nn.Conv2d(12, 64, 5)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.fc1 = nn.Linear(64*4*4, 10)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
    
        def forward(self, x):
    
            # 在 t=0 初始化隐藏状态和输出
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky() 
            mem3 = self.lif3.init_leaky()
    
            cur1 = F.max_pool2d(self.conv1(x), 2)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = F.max_pool2d(self.conv2(spk1), 2)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc1(spk2.view(batch_size, -1))
            spk3, mem3 = self.lif3(cur3, mem3)
    
            return spk3, mem3

在前一个教程中，网络被封装在一个类中，如上所示。
随着网络复杂性的增加，这会增加很多我们可能希望避免的样板代码。另一种方法是使用 ``nn.Sequential`` 方法。

.. note::
    下面的代码块在单个时间步上模拟，需要一个单独的时间循环。

::

    # 初始化网络
    net = nn.Sequential(nn.Conv2d(1, 12, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Conv2d(12, 64, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Flatten(),
                        nn.Linear(64*4*4, 10),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(device)

``init_hidden`` 参数初始化神经元的隐藏状态（这里是膜电位）。这在后台作为实例变量发生。
如果激活了 ``init_hidden``，则膜电位不会显式返回给用户，确保只有输出脉冲被顺序地通过 ``nn.Sequential`` 包装的层传递。

要使用最后一层的膜电位训练模型，请设置参数 ``output=True``。
这使得最后一层能够返回神经元的脉冲和膜电位响应。


2.3 前向传递
~~~~~~~~~~~~~~~~~~~~

在 ``num_steps`` 的仿真时长内的前向传递看起来像这样：

::

    data, targets = next(iter(train_loader))
    data = data.to(device)
    targets = targets.to(device)
    
    for step in range(num_steps):
        spk_out, mem_out = net(data)

将其封装在一个函数中，记录膜电位和脉冲响应随时间的变化：

::

    def forward_pass(net, num_steps, data):
      mem_rec = []
      spk_rec = []
      utils.reset(net)  # 重置 net 中所有 LIF 神经元的隐藏状态
    
      for step in range(num_steps):
          spk_out, mem_out = net(data)
          spk_rec.append(spk_out)
          mem_rec.append(mem_out)
      
      return torch.stack(spk_rec), torch.stack(mem_rec)

::

    spk_rec, mem_rec = forward_pass(net, num_steps, data)

3. 训练循环
-----------------

3.1 使用 snn.Functional 的损失
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在前一个教程中，我们使用输出神经元的膜电位和目标之间的交叉熵损失来训练网络。
而这次，我们将使用每个神经元的总脉冲数来计算交叉熵。

``snn.functional`` 模块中包含了各种损失函数，类似于 PyTorch 中的 ``torch.nn.functional``。
这些实现了交叉熵和均方误差损失的混合，应用于脉冲和/或膜电位，以训练速率或延迟编码网络。

下面的方法将交叉熵损失应用于输出脉冲计数，以训练一个脉冲率编码网络：

::

    # 已经导入 snntorch.functional 作为 SF
    loss_fn = SF.ce_rate_loss()

将脉冲记录作为第一个参数传递给
``loss_fn``，并将目标神经元索引作为第二个参数来生成损失。 `这里提供了更多信息和示例。 <https://snntorch.readthedocs.io/en/latest/snntorch.functional.html#snntorch.functional.ce_rate_loss>`__

::

    loss_val = loss_fn(spk_rec, targets)

::

    >>> print(f"未训练网络的损失是 {loss_val.item():.3f}")
    未训练网络的损失是 2.303


3.2 使用 snn.Functional 的准确度
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SF.accuracy_rate()`` 函数的工作方式类似，预测的输出脉冲和实际目标作为参数提供。
``accuracy_rate`` 假设使用速率编码来解释输出，通过检查具有最高脉冲计数的神经元索引是否与目标索引匹配。

::

    acc = SF.accuracy_rate(spk_rec, targets)

::

    >>> print(f"使用未训练网络的单个批次的准确度是 {acc*100:.3f}%")
    使用未训练网络的单个批次的准确度是 10.938%

由于上述函数只返回单个批次数据的准确度，以下函数返回整个
DataLoader 对象的准确度：

::

    def batch_accuracy(train_loader, net, num_steps):
      with torch.no_grad():
        total = 0
        acc = 0
        net.eval()
        
        train_loader = iter(train_loader)
        for data, targets in train_loader:
          data = data.to(device)
          targets = targets.to(device)
          spk_rec, _ = forward_pass(net, num_steps, data)
    
          acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
          total += spk_rec.size(1)
    
      return acc/total

::

    test_acc = batch_accuracy(test_loader, net, num_steps)

::

    >>> print(f"测试集上的总准确度是: {test_acc * 100:.2f}%")
    测试集上的总准确度是: 8.59%

3.3 训练循环
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下训练循环在质量上类似于前一个教程。

::

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
    num_epochs = 1
    loss_hist = []
    test_acc_hist = []
    counter = 0

    # 外部训练循环
    for epoch in range(num_epochs):

        # 训练循环
        for data, targets in iter(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            # 前向传递
            net.train()
            spk_rec, _ = forward_pass(net, num_steps, data)

            # 初始化损失并在时间上求和
            loss_val = loss_fn(spk_rec, targets)

            # 梯度计算 + 权重更新
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # 存储损失历史以备将来绘图
            loss_hist.append(loss_val.item())

            # 测试集
            if counter % 50 == 0:
            with torch.no_grad():
                net.eval()

                # 测试集前向传递
                test_acc = batch_accuracy(test_loader, net, num_steps)
                print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                test_acc_hist.append(test_acc.item())

            counter += 1


输出应该看起来像这样：

::

    Iteration 0, Test Acc: 9.82%

    Iteration 50, Test Acc: 91.98%

    Iteration 100, Test Acc: 94.90%

    Iteration 150, Test Acc: 95.70%


尽管我们选择了一些相当普通的值和架构，
考虑到我们只训练了一会儿，测试集准确度应该相当有竞争力！


4. 结果
-----------

4.1 绘制测试准确率
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    # 绘制损失
    fig = plt.figure(facecolor="w")
    plt.plot(test_acc_hist)
    plt.title("测试集准确率")
    plt.xlabel("轮次")
    plt.ylabel("准确率")
    plt.show()


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial6/test_acc.png?raw=true
        :align: center
        :width: 450

4.2 脉冲计数器
~~~~~~~~~~~~~~~~~~~~~~~

对一批数据进行前向传递，以获得脉冲和膜电位读数。

::

    spk_rec, mem_rec = forward_pass(net, num_steps, data)

改变 ``idx`` 可以让你索引到模拟小批量中的不同样本。使用 ``splt.spike_count`` 探索几个不同样本的脉冲行为！

   注意：如果你在本地桌面上运行笔记本，请
   取消下面这行的注释，并修改路径到你的 ffmpeg.exe

::

    from IPython.display import HTML
    
    idx = 0
    
    fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
    
    # plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'
    
    # 绘制脉冲计数直方图
    anim = splt.spike_count(spk_rec[:, idx].detach().cpu(), fig, ax, labels=labels, 
                            animate=True, interpolate=4)
    
    HTML(anim.to_html5_video())
    # anim.save("spike_bar.mp4")


.. raw:: html

    <center>
        <video controls src="https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial6/spike_bar.mp4?raw=true"></video>
    </center>

::

    >>> print(f"目标标签是: {targets[idx]}")
    目标标签是: 3

结论
------------

你现在应该掌握了 snnTorch 的基本特性，
并能够开始进行你自己的实验。
在 `下一个教程 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__ 中，
我们将使用一个神经形态数据集来训练一个网络。

特别感谢 `Gianfrancesco Angelini <https://github.com/gianfa>`__ 对教程提供的宝贵反馈。

如果你喜欢这个项目，请考虑在 GitHub 上给仓库点赞⭐，这是支持它的最简单也是最好的方式。

额外资源
---------------------

- `在这里查看 snnTorch 的 GitHub 项目。 <https://github.com/jeshraghian/snntorch>`__
