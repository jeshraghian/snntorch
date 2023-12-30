===================================================
脉冲神经网络中的种群编码
===================================================

本教程出自 Jason K. Eshraghian (`www.ncg.ucsc.edu <https://www.ncg.ucsc.edu>`_)

 `English <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_pop.html#>`_ 

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_pop.ipynb

snnTorch 教程系列基于以下论文。如果您发现这些资源或代码对您的工作有用，请考虑引用以下来源：

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  本教程是不可编辑的静态版本。交互式可编辑版本可通过以下链接获取：
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_pop.ipynb>`_
    * `Local Notebook (download via GitHub) <https://github.com/jeshraghian/snntorch/tree/master/examples>`_


简介
============

据认为，脉冲率编码机制不太可能是初级皮层的主要编码机制。其中一个原因是平均神经元脉冲率大约是 :math:`0.1-1` Hz，这比动物和人类的反应响应时间慢得多。

但如果我们将多个神经元汇集在一起，并一起计算它们的脉冲，那么就有可能在非常短的时间窗口内测量一组神经元的发放率。种群编码增加了脉冲率编码机制的可信度。

   .. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/tutorial_pop/pop.png?raw=true
            :align: center
            :width: 300

在本教程中，您将：

    * 学习如何训练一个种群编码网络。我们将不再为每个类别指定一个神经元，而是扩展到每个类别有多个神经元，并将它们的脉冲聚合在一起。

::

    !pip install snntorch

::

    import torch, torch.nn as nn
    import snntorch as snn

数据加载
===========

定义数据加载的变量。

::

    batch_size = 128
    data_path='/tmp/data/fmnist'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

加载 FashionMNIST 数据集。

::

    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    # 定义转换
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
    
    fmnist_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
    
    # 创建 DataLoader
    train_loader = DataLoader(fmnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(fmnist_test, batch_size=batch_size, shuffle=True)


定义网络
==============

让我们比较一下使用和不使用种群编码的两个网络的性能，并训练它们 *一个时间步*。

::

    from snntorch import surrogate
    
    # 网络参数
    num_inputs = 28*28
    num_hidden = 128
    num_outputs = 10
    num_steps = 1
    
    # 脉冲神经元参数
    beta = 0.9  # 神经元衰减率 
    grad = surrogate.fast_sigmoid()

不使用种群编码
-------------------------

让我们只使用一个简单的两层密集脉冲网络。

::

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(num_inputs, num_hidden),
                        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
                        nn.Linear(num_hidden, num_outputs),
                        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
                        ).to(device)

使用种群编码
----------------------

不是使用 10 个输出神经元对应于 10 个输出类别，我们将使用 500 个输出神经元。这意味着每个输出类别随机分配了 50 个神经元。

::

    pop_outputs = 500
    
    net_pop = nn.Sequential(nn.Flatten(),
                            nn.Linear(num_inputs, num_hidden),
                            snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
                            nn.Linear(num_hidden, pop_outputs),
                            snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
                            ).to(device)

训练
========

不使用种群编码
-------------------------

定义优化器和损失函数。这里，我们使用 MSE 脉冲计数损失，它在仿真运行结束时计算输出脉冲的总数量。

正确类别的目标发放概率设置为 100%，错误类别设置为 0%。

::

    import snntorch.functional as SF
    
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0)

我们还将定义一个简单的测试准确率函数，它根据脉冲计数最高的神经元预测正确的类别。

::

    from snntorch import utils
    
    def test_accuracy(data_loader, net, num_steps, population_code=False, num_classes=False):
      with torch.no_grad():
        total = 0
        acc = 0
        net.eval()
    
        data_loader = iter(data_loader)
        for data, targets in data_loader:
          data = data.to(device)
          targets = targets.to(device)
          utils.reset(net)
          spk_rec, _ = net(data)
    
          if population_code:
            acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets, population_code=True, num_classes=10) * spk_rec.size(1)
          else:
            acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets) * spk_rec.size(1)
            
          total += spk_rec.size(1)
    
      return acc/total

让我们运行训练循环。注意我们只训练 :math:`1` 个时间步。也就是说，每个神经元只有一次发射的机会。因此，我们可能不会期望网络在这里表现得太好。

::

    from snntorch import backprop
    
    num_epochs = 5
    
    # 训练循环
    for epoch in range(num_epochs):
    
        avg_loss = backprop.BPTT(net, train_loader, num_steps=num_steps,
                              optimizer=optimizer, criterion=loss_fn, time_var=False, device=device)
        
        print(f"轮次: {epoch}")
        print(f"测试集准确率: {test_accuracy(test_loader, net, num_steps)*100:.3f}%\n")

        >> 轮次: 0
        >> 测试集准确率: 59.421%

        >> 轮次: 1
        >> 测试集准确率: 61.889%

虽然有一些方法可以改善单个时间步的性能，例如通过将损失应用到膜电位上，但使用速率编码在一个时间步上训练网络是极具挑战性的。

使用种群编码
----------------------

让我们修改损失函数以指定应启用种群编码。我们还必须指定类别数量。这意味着将有总共
:math:`50~神经元~每个类别~=~500~神经元~/~10~类别`。

::

    loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=10)
    optimizer = torch.optim.Adam(net_pop.parameters(), lr=2e-3, betas=(0.9, 0.999))

::

    num_epochs = 5
    
    # 训练循环
    for epoch in range(num_epochs):
    
        avg_loss = backprop.BPTT(net_pop, train_loader, num_steps=num_steps,
                                optimizer=optimizer, criterion=loss_fn, time_var=False, device=device)
    
        print(f"轮次: {epoch}")
        print(f"测试集准确率: {test_accuracy(test_loader, net_pop, num_steps, population_code=True, num_classes=10)*100:.3f}%\n")

        >> 轮次: 0
        >> 测试集准确率: 80.501%

        >> 轮次: 1
        >> 测试集准确率: 82.690%

即使我们只在一个时间步上训练，引入额外的输出神经元立即使性能得到了提升。

结论
==========

随着时间步的增加，种群编码的性能提升可能开始减弱。但它也可能比增加时间步更可取，因为 PyTorch 优化了处理矩阵-向量乘积，而不是随时间的顺序、逐步操作。

-  关于脉冲神经元、神经网络、编码和使用神经形态数据集训练的详细教程，请查看 `snnTorch
   教程系列 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__。
-  有关 snnTorch 功能的更多信息，请查看
   `此链接的文档 <https://snntorch.readthedocs.io/en/latest/>`__。
-  如果您有想法、建议或希望找到参与的方式，请 `查看 snnTorch GitHub 项目。 <https://github.com/jeshraghian/snntorch>`__
