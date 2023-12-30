=============================
SNNs回归（二）
=============================

基于回归的分类法与递归泄漏整合与发射神经元
-------------------------------------------------------------------------------

本教程出自 Alexander Henkes (`ORCID <https://orcid.org/0000-0003-4615-9271>`_) 与 Jason K. Eshraghian (`ncg.ucsc.edu <https://ncg.ucsc.edu>`_)

 `English <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_2.html#>`_ 

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_2.ipynb


snnTorch 教程系列基于以下论文。如果您发现这些资源或代码对您的工作有用，请考虑引用以下来源：

   `Alexander Henkes, Jason K. Eshraghian, and Henning Wessels. “Spiking
   neural networks for nonlinear regression”, arXiv preprint
   arXiv:2210.03515, October 2022. <https://arxiv.org/abs/2210.03515>`_

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  本教程是不可编辑的静态版本。交互式可编辑版本可通过以下链接获取：
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_2.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


在回归教程系列中，您将学习如何使用 snnTorch 执行回归，使用各种脉冲神经元模型，包括：

-  LIF神经元
-  递归 LIF 神经元
-  脉冲 LSTM

回归教程系列的概览：

-  第一部分（本教程）将训练 LIF 神经元的膜电位随时间跟随给定轨迹。
-  第二部分将使用带有递归反馈的 LIF 神经元，使用基于回归的损失函数执行分类
-  第三部分将使用更复杂的脉冲 LSTM 网络来训练神经元的发射时间。



::

    !pip install snntorch --quiet

::

    # imports
    import snntorch as snn
    from snntorch import functional as SF
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torch.nn.functional as F
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    import tqdm

1. 将分类视为回归
--------------------------------------------

在传统的深度学习中，我们经常计算交叉熵损失来训练网络进行分类。激活度最高的输出神经元被认为是预测的类别。

在脉冲神经网络中，这可能被解释为发射最多脉冲的类别。即，将交叉熵应用于总脉冲计数（或发射频率）。这样做的效果是，预测的类别将被最大化，而其他类别则被抑制。

大脑并不完全像这样工作。SNN 是稀疏激活的，虽然用这种深度学习的态度来处理 SNN 可能会获得最优的准确率，但重要的是不要过度迎合深度学习界的做法。

毕竟，我们使用脉冲来实现更好的功耗效率。良好的功耗效率依赖于稀疏的脉冲活动。

换句话说，使用深度学习技巧训练生物启发的 SNN 并不会导致类似大脑的活动。

那么我们能做些什么呢？

我们将专注于将分类问题转化为回归任务。这是通过训练预测的神经元发射给定次数的脉冲，而错误的神经元也被训练发射给定次数的脉冲，尽管频率较低。

这与交叉熵形成对比，后者会试图驱使正确的类别在 *所有* 时间步中发射脉冲，而错误的类别则完全不发射脉冲。

与前一个教程一样，我们可以使用均方误差来实现这一点。回顾一下均方误差损失的形式：

.. math:: \mathcal{L}_{MSE} = \frac{1}{n}\sum_{i=1}^n(y_i-\hat{y_i})^2

其中 :math:`y` 是目标，:math:`\hat{y}` 是预测值。

为了将 MSE 应用于脉冲计数，假设我们在分类问题中有 :math:`n` 个输出神经元，其中 :math:`n` 是可能的类别数量。:math:`\hat{y}_i` 现在是 :math:`i^{th}` 输出神经元在整个仿真运行时间内发射的总脉冲数。

鉴于我们有 :math:`n` 个神经元，这意味着 :math:`y` 和 :math:`\hat{y}` 必须是带有 :math:`n` 个元素的向量，我们的损失将总结每个神经元的独立 MSE 损失。

1.1 理论示例
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

考虑一个仿真的 10 个时间步。假设我们希望正确的神经元类别发射 8 次，错误的类别发射 2 次。假设 :math:`y_1` 是正确的类别：

.. math::  y = \begin{bmatrix} 8 \\ 2 \\ \vdots \\ 2 \end{bmatrix},  \hat{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}

对每个元素单独计算 MSE 以产生 :math:`n` 个损失分量，所有这些都加在一起生成最终损失。


2. 递归LIF神经元
------------------------------------------------

大脑中的神经元有大量的反馈连接。因此，SNN 社区一直在探索将输出脉冲反馈到输入的网络动态。这是除了膜电位的递归动态之外的。

在 snnTorch 中有几种构建RLIF（ ``RLeaky`` ）神经元的方法。

请参阅 `此文档 <https://snntorch.readthedocs.io/en/latest/snn.neurons_rleaky.html>`__ 以了解神经元的超参数的详尽描述。让我们看几个例子。

2.1 一对一连接的 RLIF 神经元
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/regression2/reg2-1.jpg?raw=true
        :align: center
        :width: 400

这假设每个神经元将其输出脉冲反馈到自己，并且只反馈到自己。同一层中的神经元之间没有交叉耦合连接。

::

    beta = 0.9 # 膜电位衰减率
    num_steps = 10 # 10 个时间步
    
    rlif = snn.RLeaky(beta=beta, all_to_all=False) # 初始化 RLeaky 神经元
    spk, mem = rlif.init_rleaky() # 初始化状态变量
    x = torch.rand(1) # 生成随机输入
    
    spk_recording = []
    mem_recording = []
    
    # 运行仿真
    for step in range(num_steps):
      spk, mem = rlif(x, spk, mem)
      spk_recording.append(spk)
      mem_recording.append(mem)

默认情况下， ``V`` 是一个可学习的参数，初始化为 :math:`1` 并在训练过程中更新。如果您希望禁用学习或使用自己的初始化变量，可以这样做：

::

    rlif = snn.RLeaky(beta=beta, all_to_all=False, learn_recurrent=False) # 禁用递归连接的学习
    rlif.V = torch.rand(1) # 设置为层大小
    print(f"递归权重是: {rlif.V.item()}")

2.2 所有连接的 RLIF 神经元
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.2.1 线性反馈
............................

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/regression2/reg2-2.jpg?raw=true
        :align: center
        :width: 400

默认情况下， ``RLeaky`` 假设反馈连接，其中给定层的所有脉冲首先由反馈层加权，然后传递到所有神经元的输入。这引入了更多的参数，但据认为这有助于学习数据中的时变特征。

::

    beta = 0.9 # 膜电位衰减率
    num_steps = 10 # 10 个时间步
    
    rlif = snn.RLeaky(beta=beta, linear_features=10)  # 初始化 RLeaky 神经元
    spk, mem = rlif.init_rleaky() # 初始化状态变量
    x = torch.rand(10) # 生成随机输入
    
    spk_recording = []
    mem_recording = []
    
    # 运行仿真
    for step in range(num_steps):
      spk, mem = rlif(x, spk, mem)
      spk_recording.append(spk)
      mem_recording.append(mem)

您可以使用 ``learn_recurrent=False`` 在反馈层中禁用学习。

2.2.2 卷积反馈
..............................................................

如果您使用的是卷积层，这将引发错误，因为不适合将输出脉冲（3 维）通过 ``nn.Linear`` 反馈层投影到 1 维。

为了解决这个问题，您必须指定您正在使用卷积反馈层：

::

    beta = 0.9 # 膜电位衰减率
    num_steps = 10 # 10 个时间步
    
    rlif = snn.RLeaky(beta=beta, conv2d_channels=3, kernel_size=(5,5))  # 初始化 RLeaky 神经元
    spk, mem = rlif.init_rleaky() # 初始化状态变量
    x = torch.rand(3, 32, 32) # 生成随机 3D 输入
    
    spk_recording = []
    mem_recording = []
    
    # 运行仿真
    for step in range(num_steps):
      spk, mem = rlif(x, spk, mem)
      spk_recording.append(spk)
      mem_recording.append(mem)

为确保输出脉冲尺寸与输入尺寸匹配，自动应用填充。

如果您有异常形状的数据，您需要手动构建自己的反馈层。

3. 构建模型
------------------------

让我们使用 ``RLeaky`` 层训练几个模型。为了加快速度，我们将训练一个具有线性反馈的模型。

::

    class Net(torch.nn.Module):
        """snnTorch 中的简单脉冲神经网络。"""
    
        def __init__(self, timesteps, hidden, beta):
            super().__init__()
            
            self.timesteps = timesteps
            self.hidden = hidden
            self.beta = beta
    
            # 第 1 层
            self.fc1 = torch.nn.Linear(in_features=784, out_features=self.hidden)
            self.rlif1 = snn.RLeaky(beta=self.beta, linear_features=self.hidden)
    
            # 第 2 层
            self.fc2 = torch.nn.Linear(in_features=self.hidden, out_features=10)
            self.rlif2 = snn.RLeaky(beta=self.beta, linear_features=10)
    
        def forward(self, x):
            """多个时间步的前向传递。"""
    
            # 初始化膜电位
            spk1, mem1 = self.rlif1.init_rleaky()
            spk2, mem2 = self.rlif2.init_rleaky()
    
            # 用于记录输出的空列表
            spk_recording = []
    
            for step in range(self.timesteps):
                spk1, mem1 = self.rlif1(self.fc1(x), spk1, mem1)
                spk2, mem2 = self.rlif2(self.fc2(spk1), spk2, mem2)
                spk_recording.append(spk2)
    
            return torch.stack(spk_recording)

在下面实例化网络：

::

    hidden = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = Net(timesteps=num_steps, hidden=hidden, beta=0.9).to(device)

4. 构建训练循环
--------------------------------------------

4.1 ``snntorch.functional`` 中的均方误差损失
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

从 ``snntorch.functional`` 中，我们调用 ``mse_count_loss`` 来设置目标神经元以 80% 的时间发射脉冲，错误神经元以 20% 的时间发射脉冲。花了 10 段话解释的内容，在一行代码中实现：

::

    loss_function = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

4.2 DataLoader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DataLoader 的样板代码。让我们只做 MNIST，将这个测试应用于时间数据是留给读者/编码者的练习。

::

    batch_size = 128
    data_path='/tmp/data/mnist'
    
    # 定义转换
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
    
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    
    # 创建 DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

4.3 训练网络
-----------------

::

    num_epochs = 5
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    loss_hist = []
    
    with tqdm.trange(num_epochs) as pbar:
        for _ in pbar:
            train_batch = iter(train_loader)
            minibatch_counter = 0
            loss_epoch = []
    
            for feature, label in train_batch:
                feature = feature.to(device)
                label = label.to(device)
    
                spk = model(feature.flatten(1)) # 前向传递
                loss_val = loss_function(spk, label) # 应用损失
                optimizer.zero_grad() # 清零梯度
                loss_val.backward() # 计算梯度
                optimizer.step() # 更新权重
    
                loss_hist.append(loss_val.item())
                minibatch_counter += 1
    
                avg_batch_loss = sum(loss_hist) / minibatch_counter
                pbar.set_postfix(loss="%.3e" % avg_batch_loss)

5. 评估
----------------------

::

    test_batch = iter(test_loader)
    minibatch_counter = 0
    loss_epoch = []
    
    model.eval()
    with torch.no_grad():
      total = 0
      acc = 0
      for feature, label in test_batch:
          feature = feature.to(device)
          label = label.to(device)
    
          spk = model(feature.flatten(1)) # 前向传递
          acc += SF.accuracy_rate(spk, label) * spk.size(1)
          total += spk.size(1)
    
    print(f"测试集上的总准确率是: {(acc/total) * 100:.2f}%")

6. 替代损失度量
==========================

在上一个教程中，我们测试了膜电位学习。我们可以在这里做同样的事情，通过设置目标神经元达到高于发射阈值的膜电位，而错误神经元达到低于发射阈值的膜电位：

::

    loss_function = SF.mse_membrane_loss(on_target=1.05, off_target=0.2)

在上述情况下，我们尝试让正确的神经元不断保持在发射阈值之上。

尝试更新网络和训练循环以使其工作。

提示：

- 您需要返回输出膜电位而不是脉冲。

- 将膜电位而不是脉冲传递给损失函数

结论
------------------------

下一个回归教程将引入脉冲 LSTM 以实现精确的脉冲时间学习。

如果您喜欢这个项目，请考虑在 GitHub 上给仓库点赞⭐，这是支持它的最简单也是最好的方式。

额外资源
------------------------

-  `在这里查看 snntorch GitHub 项目。 <https://github.com/jeshraghian/snntorch>`__
-  有关 SNN 进行非线性回归的更多细节可以在我们的相应预印本中找到： `Henkes, A.; Eshraghian, J. K.; 和
   Wessels, H. “脉冲神经网络用于非线性回归”，arXiv 预印本 arXiv:2210.03515,
   2022年10月。 <https://arxiv.org/abs/2210.03515>`__
