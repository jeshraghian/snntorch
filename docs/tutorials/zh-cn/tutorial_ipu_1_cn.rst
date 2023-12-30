===================================================
在 IPU 上加速 snnTorch
===================================================

教程由 `Jason K. Eshraghian <https://www.jasoneshraghian.com>`_ 和 Vincent Sun 编写

 `English <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_ipu_1.html#>`_ 

snnTorch 教程系列基于以下论文。如果您在工作中发现这些资源或代码有用，请考虑引用以下来源：

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  本教程是不可编辑的静态版本。交互式可编辑版本可通过以下链接获取：
    * `Python 脚本（通过 GitHub 下载）<https://github.com/jeshraghian/snntorch/tree/master/examples/tutorial_ipu_1.py>`_


简介
============

脉冲神经网络（SNN）在执行深度学习工作负载的推理时，在能耗和延迟方面实现了数量级的改善。
但讽刺的是，使用误差反向传播训练 SNN 在 CPU 和 GPU 上比非脉冲网络更昂贵。
必须考虑到额外的时间维度，并且当使用时域反向传播算法训练网络时，内存复杂性会随时间线性增长。

snnTorch 的一个替代构建已经为 `Graphcore 的智能处理单元（IPU） <https://www.graphcore.ai/>`_ 优化。
IPU 是为深度学习工作负载量身定制的加速器，通过在更小的数据块上运行单独的处理线程来采用多指令多数据（MIMD）并行性。
这非常适合必须顺序处理且不能向量化的脉冲神经元动力学状态方程的分区。


在本教程中，您将：

    * 学习如何训练使用 IPU 加速的 SNN。


确保安装了最新版本的 :code:`poptorch` 和 Poplar SDK。有关安装说明，请参阅 `Graphcore 的文档 <https://github.com/graphcore/poptorch>`_。

在没有预先安装 :code:`snntorch` 的环境中安装 :code:`snntorch-ipu` 以避免包冲突：

::

    !pip install snntorch-ipu

导入所需的 Python 包：

::

    import torch, torch.nn as nn
    import popart, poptorch
    import snntorch as snn
    import snntorch.functional as SF

数据加载
===========

加载 MNIST 数据集。

::

    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

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
    
    # 使用全精度 32-float 训练
    opts = poptorch.Options()
    opts.Precision.halfFloatCasting(poptorch.HalfFloatCastingBehavior.HalfUpcastToFloat)

    # 创建 DataLoader
    train_loader = poptorch.DataLoader(options=opts, dataset=mnist_train, batch_size=batch_size, shuffle=True, num_workers=20)
    test_loader = poptorch.DataLoader(options=opts, dataset=mnist_test, batch_size=batch_size, shuffle=True, num_workers=20)


定义网络
==============

让我们用缓慢的状态衰减率模拟我们的网络 25 个时间步：

::

    num_steps = 25
    beta = 0.9


我们现在将构建一个普通的 SNN 模型。
在 IPU 上训练时，请注意损失函数必须包装在模型类中。
完整的代码如下：

::

    class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_inputs = 784
        num_hidden = 1000
        num_outputs = 10

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

        # 交叉熵脉冲计数损失
        self.loss_fn = SF.ce_count_loss()

    def forward(self, x, labels=None):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []
       
        for step in range(num_steps):
            cur1 = self.fc1(x.view(batch_size,-1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        spk2_rec = torch.stack(spk2_rec)
        mem2_rec = torch.stack(mem2_rec)

        if self.training:
            return spk2_rec, poptorch.identity_loss(self.loss_fn(mem2_rec, labels), "none")
        return spk2_rec


让我们快速梳理一下。

构建模型与所有之前的教程相同。我们在每个密集层的末尾应用脉冲神经元节点：

::

    self.fc1 = nn.Linear(num_inputs, num_hidden)
    self.lif1 = snn.Leaky(beta=beta)
    self.fc2 = nn.Linear(num_hidden, num_outputs)
    self.lif2 = snn.Leaky(beta=beta)

默认情况下，脉冲神经元的替代梯度将是一个直通估计器。
如果您更喜欢使用快速 Sigmoid 或 Sigmoid 选项，也可以选择它们：

::

    from snntorch import surrogate

    self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())


损失函数将计算每个输出神经元的总脉冲数量并应用交叉熵损失：

::

    self.loss_fn = SF.ce_count_loss()

现在我们定义前向传递。通过调用以下函数初始化每个脉冲神经元的隐藏状态：

::

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

接下来，运行 for 循环以在 25 个时间步上模拟 SNN。
输入数据使用 :code:`.view(batch_size, -1)` 展平，使其与密集输入层兼容。

::

    for step in range(num_steps):
        cur1 = self.fc1(x.view(batch_size,-1))
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)

使用函数 :code:`poptorch.identity_loss(self.loss_fn(mem2_rec, labels), "none")` 应用损失。


在 IPUs 上训练
=================

现在，完整的训练循环将在 10 个轮次中运行。
注意优化器是从 :code:`poptorch` 调用的。否则，训练过程与 snnTorch 的典型使用大致相同。

::

    net = Model()
    optimizer = poptorch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

    poptorch_model = poptorch.trainingModel(net, options=opts, optimizer=optimizer)

    epochs = 10
    for epoch in tqdm(range(epochs), desc="epochs"):
        correct = 0.0

        for i, (data, labels) in enumerate(train_loader):
            output, loss = poptorch_model(data, labels)

            if i % 250 == 0:
                _, pred = output.sum(dim=0).max(1)
                correct = (labels == pred).sum().item()/len(labels)

                # 单个批次的准确率
                print("准确率: ", correct)

模型首先会被编译，之后，训练过程将开始。
为了保持这个教程简洁快速，训练集上的单个小批量的准确率将被打印出来。


结论
==========

我们的初步基准测试显示，在各种神经元模型的混合精度训练吞吐量上，与 CUDA 加速 SNN 相比，可以提高多达 10 倍的改善。
目前正在制作一个详细的基准测试和博客，突出显示额外的功能。

-  关于脉冲神经元、神经网络、编码和使用神经形态数据集训练的详细教程，请查看 `snnTorch
   教程系列 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__。
-  有关 snnTorch 功能的更多信息，请查看
   `此链接的文档 <https://snntorch.readthedocs.io/en/latest/>`__。
-  如果您有想法、建议或希望找到参与的方式，请 `查看 snnTorch GitHub 项目。 <https://github.com/jeshraghian/snntorch>`__
