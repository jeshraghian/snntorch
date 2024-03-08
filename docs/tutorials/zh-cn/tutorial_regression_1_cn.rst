============================
SNNs回归（一）
============================

用 LIF 神经元学习膜电位
---------------------------------------------

本教程出自 Alexander Henkes (`ORCID <https://orcid.org/0000-0003-4615-9271>`_) 与 Jason K. Eshraghian (`ncg.ucsc.edu <https://ncg.ucsc.edu>`_)

 `English <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_1.html#>`_ 

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_1.ipynb


snnTorch 教程系列基于以下论文。如果您发现这些资源或代码对您的工作有用，请考虑引用以下来源：

   `Alexander Henkes, Jason K. Eshraghian, and Henning Wessels. “Spiking
   neural networks for nonlinear regression”, arXiv preprint
   arXiv:2210.03515, October 2022. <https://arxiv.org/abs/2210.03515>`_

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. “Training
    Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. note::
  本教程是不可编辑的静态版本。交互式可编辑版本可通过以下链接获取：
    * `Google Colab <https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_1.ipynb>`_
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

    # 导入
    import snntorch as snn
    from snntorch import surrogate
    from snntorch import functional as SF
    from snntorch import utils
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torch.nn.functional as F
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    import random
    import statistics
    import tqdm

固定随机种子：

::

    # 种子
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

1. 脉冲回归
----------------------

1.1 线性和非线性回归的快速背景
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

到目前为止的教程都集中在多类分类问题上。但如果你已经走到这一步，那么可以肯定地说，你的大脑不仅仅能区分猫和狗。你很棒，我们相信你。

另一个问题是回归，其中多个输入特征 :math:`x_i` 被用来估计连续数值线 :math:`y \in \mathbb{R}` 上的输出。一个经典的例子是根据一系列输入，如土地大小、房间数量和对鳄梨吐司的当地需求，来估计房子的价格。

回归问题的目标通常是均方误差：

.. math:: \mathcal{L}_{MSE} = \frac{1}{n}\sum_{i=1}^n(y_i-\hat{y_i})^2

或者是平均绝对误差：

.. math:: \mathcal{L}_{L1} = \frac{1}{n}\sum_{i=1}^n|y_i-\hat{y_i}|

其中 :math:`y` 是目标，:math:`\hat{y}` 是预测值。

线性回归的一个挑战是它只能在预测输出时使用输入特征的线性加权。使用均方误差作为成本函数训练的神经网络允许我们对更复杂的数据进行非线性回归。

1.2 在回归中使用脉冲神经元
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

脉冲是一种非线性，也可以用来学习更复杂的回归任务。但如果脉冲神经元只发射代表 1 和 0 的脉冲，那么我们如何进行回归呢？我很高兴你问了！以下是一些想法：

-  使用脉冲总数（基于速率的编码）
-  使用脉冲发生的时间（基于时间/潜伏期的编码）
-  使用脉冲对之间的距离（即使用脉冲间隔）

或者你也可以用电探针穿透神经元膜，决定使用膜电位，这是一个连续值。

   注意：直接访问膜电位是否作弊，即直接访问一个本应是“隐藏状态”的东西？目前，在神经形态社区中还没有太多共识。尽管在许多模型中膜电位是一个高精度变量（因此计算成本高），但它通常用于损失函数，因为它比离散的时间步或脉冲计数更“连续”。尽管在高精度值上操作的功耗和延迟成本更高，但如果你有一个小的输出层，或者输出不需要通过权重缩放，那么影响可能微不足道。这真的是一个特定于任务和特定于硬件的问题。

2. 设置回归问题
------------------------------------------------

2.1 创建数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

让我们构建一个简单的玩具问题。以下类返回我们希望学习的函数。如果 ``mode = "linear"``，生成一个随机斜率的直线。如果 ``mode = "sqrt"``，则取这条直线的平方根。

我们的目标：训练一个漏积分-触发神经元，使其膜电位随时间跟随样本。

::

    class RegressionDataset(torch.utils.data.Dataset):
        """简单的回归数据集。"""
    
        def __init__(self, timesteps, num_samples, mode):
            """输入和输出之间的线性关系"""
            self.num_samples = num_samples # 生成的样本数量
            feature_lst = [] # 在列表中存储每个生成的样本
    
            # 一个接一个地生成线性函数
            for idx in range(num_samples):
                end = float(torch.rand(1)) # 随机最终点
                lin_vec = torch.linspace(start=0.0, end=end, steps=timesteps) # 从 0 生成到 end 的线性函数
                feature = lin_vec.view(timesteps, 1)
                feature_lst.append(feature) # 将样本添加到列表
    
            self.features = torch.stack(feature_lst, dim=1) # 将列表转换为张量
    
            # 生成线性函数或平方根函数的选项
            if mode == "linear":
                self.labels = self.features * 1
    
            elif mode == "sqrt":
                slope = float(torch.rand(1))
                self.labels = torch.sqrt(self.features * slope)
    
            else:
                raise NotImplementedError("'linear', 'sqrt'")
    
        def __len__(self):
            """样本数量。"""
            return self.num_samples
    
        def __getitem__(self, idx):
            """通用实现，但我们只有一个样本。"""
            return self.features[:, idx, :], self.labels[:, idx, :]


要查看随机样本的样子，请运行以下代码块：

::

    num_steps = 50
    num_samples = 1
    mode = "sqrt" # 'linear' 或 'sqrt'
    
    # 生成单个数据样本
    dataset = RegressionDataset(timesteps=num_steps, num_samples=num_samples, mode=mode)
    
    # 绘图
    sample = dataset.labels[:, 0, 0]
    plt.plot(sample)
    plt.title("教给网络的目标函数")
    plt.xlabel("时间")
    plt.ylabel("膜电位")
    plt.show()


.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/regression1/reg_1-1.png?raw=true
        :align: center
        :width: 450


2.2 创建 DataLoader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

上面创建的 Dataset 对象将数据加载到内存中，DataLoader 将以批次形式提供数据。
PyTorch 中的 DataLoader 是将数据传入网络的便捷接口。它们返回一个划分为大小为 ``batch_size`` 的小批量的迭代器。

::

    batch_size = 1 # 只有一个样本要学习
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True)

3. 构建模型
------------------------

让我们尝试一个简单的网络，仅使用无递归的漏积分-触发层。后续教程将展示如何使用具有更高阶递归的更复杂神经元类型。
如果数据中没有强烈的时间依赖性，即下一个时间步与前一个时间步的依赖性较弱，这些架构应该运行良好。

以下架构的一些说明：

-  设置 ``learn_beta=True`` 使衰减率 ``beta`` 成为一个可学习参数
-  每个神经元具有独特的、随机初始化的阈值和衰减率
-  通过设置 ``reset_mechanism="none"`` 禁用输出层的重置机制，因为我们不会使用任何输出脉冲

::

    class Net(torch.nn.Module):
        """snnTorch 中的简单脉冲神经网络。"""
    
        def __init__(self, timesteps, hidden):
            super().__init__()
            
            self.timesteps = timesteps # 模拟网络的时间步数
            self.hidden = hidden # 隐藏神经元的数量 
            spike_grad = surrogate.fast_sigmoid() # 替代梯度函数
            
            # 随机初始化第 1 层的衰减率和阈值
            beta_in = torch.rand(self.hidden)
            thr_in = torch.rand(self.hidden)
    
            # 第 1 层
            self.fc_in = torch.nn.Linear(in_features=1, out_features=self.hidden)
            self.lif_in = snn.Leaky(beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=spike_grad)
            
            # 随机初始化第 2 层的衰减率和阈值
            beta_hidden = torch.rand(self.hidden)
            thr_hidden = torch.rand(self.hidden)
    
            # 第 2 层
            self.fc_hidden = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
            self.lif_hidden = snn.Leaky(beta=beta_hidden, threshold=thr_hidden, learn_beta=True, spike_grad=spike_grad)
    
            # 随机初始化输出神经元的衰减率
            beta_out = torch.rand(1)
            
            # 第 3 层：漏积分神经元。注意重置机制被禁用，我们将忽略输出脉冲。
            self.fc_out = torch.nn.Linear(in_features=self.hidden, out_features=1)
            self.li_out = snn.Leaky(beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")
    
        def forward(self, x):
            """多个时间步的前向传递。"""
    
            # 初始化膜电位
            mem_1 = self.lif_in.init_leaky()
            mem_2 = self.lif_hidden.init_leaky()
            mem_3 = self.li_out.init_leaky()
    
            # 用于记录输出的空列表
            mem_3_rec = []
    
            # 循环
            for step in range(self.timesteps):
                x_timestep = x[step, :, :]
    
                cur_in = self.fc_in(x_timestep)
                spk_in, mem_1 = self.lif_in(cur_in, mem_1)
                
                cur_hidden = self.fc_hidden(spk_in)
                spk_hidden, mem_2 = self.li_out(cur_hidden, mem_2)
    
                cur_out = self.fc_out(spk_hidden)
                _, mem_3 = self.li_out(cur_out, mem_3)
    
                mem_3_rec.append(mem_3)
    
            return torch.stack(mem_3_rec)

在下面实例化网络：


::

    hidden = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = Net(timesteps=num_steps, hidden=hidden).to(device)


让我们观察输出神经元在未经训练之前的行为，以及它与目标函数的比较：

::

    train_batch = iter(dataloader)
    
    # 运行单次前向传递
    with torch.no_grad():
        for feature, label in train_batch:
            feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
            label = torch.swapaxes(input=label, axis0=0, axis1=1)
            feature = feature.to(device)
            label = label.to(device)
            mem = model(feature)
    
    # 绘图
    plt.plot(mem[:, 0, 0].cpu(), label="输出")
    plt.plot(label[:, 0, 0].cpu(), '--', label="目标")
    plt.title("未训练的输出神经元")
    plt.xlabel("时间")
    plt.ylabel("膜电位")
    plt.legend(loc='best')
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/regression1/reg_1-2.png?raw=true
        :align: center
        :width: 450

由于网络尚未经过训练，因此膜电位遵循无意义的演变并不奇怪。

4. 构建训练循环
------------------------------------------------

我们调用 ``torch.nn.MSELoss()`` 来最小化膜电位和目标演变之间的均方误差。

我们在同一数据样本上迭代。

::

    num_iter = 100 # 训练 100 次迭代
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    loss_function = torch.nn.MSELoss()
    
    loss_hist = [] # 记录损失
    
    # 训练循环
    with tqdm.trange(num_iter) as pbar:
        for _ in pbar:
            train_batch = iter(dataloader)
            minibatch_counter = 0
            loss_epoch = []
            
            for feature, label in train_batch:
                # 准备数据
                feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
                label = torch.swapaxes(input=label, axis0=0, axis1=1)
                feature = feature.to(device)
                label = label.to(device)
    
                # 前向传递
                mem = model(feature)
                loss_val = loss_function(mem, label) # 计算损失
                optimizer.zero_grad() # 清零梯度
                loss_val.backward() # 计算梯度
                optimizer.step() # 更新权重
    
                # 存储损失
                loss_hist.append(loss_val.item())
                loss_epoch.append(loss_val.item())
                minibatch_counter += 1
    
                avg_batch_loss = sum(loss_epoch) / minibatch_counter # 计算每轮的平均损失
                pbar.set_postfix(loss="%.3e" % avg_batch_loss) # 打印每批次的损失


5. 评估
------------------------

::

    loss_function = torch.nn.L1Loss() # 使用 L1 损失
    
    # 在评估期间暂停梯度计算
    with torch.no_grad():
        model.eval()
    
        test_batch = iter(dataloader)
        minibatch_counter = 0
        rel_err_lst = []
    
        # 循环遍历数据样本
        for feature, label in test_batch:
    
            # 准备数据
            feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
            label = torch.swapaxes(input=label, axis0=0, axis1=1)
            feature = feature.to(device)
            label = label.to(device)
    
            # 前向传递
            mem = model(feature)
    
            # 计算相对误差
            rel_err = torch.linalg.norm(
                (mem - label), dim=-1
            ) / torch.linalg.norm(label, dim=-1)
            rel_err = torch.mean(rel_err[1:, :])
    
            # 计算损失
            loss_val = loss_function(mem, label)
    
            # 存储损失
            loss_hist.append(loss_val.item())
            rel_err_lst.append(rel_err.item())
            minibatch_counter += 1
    
        mean_L1 = statistics.mean(loss_hist)
        mean_rel = statistics.mean(rel_err_lst)
    
    print(f"{'平均 L1-损失:':<{20}}{mean_L1:1.2e}")
    print(f"{'平均相对误差:':<{20}}{mean_rel:1.2e}")


::

    >> 平均 L1-损失:       1.22e-02
    >> 平均相对误差:     2.84e-02

让我们绘制结果以获得一些直观感受：

::

    mem = mem.cpu()
    label = label.cpu()
    
    plt.title("训练后的输出神经元")
    plt.xlabel("时间")
    plt.ylabel("膜电位")
    for i in range(batch_size):
        plt.plot(mem[:, i, :].cpu(), label="输出")
        plt.plot(label[:, i, :].cpu(), label="目标")
    plt.legend(loc='best')
    plt.show()

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/examples/regression1/reg_1-3.png?raw=true
        :align: center
        :width: 450

虽然有点参差不齐，但看起来还不错。

您可以通过扩大隐藏层的规模、增加迭代次数、增加额外的时间步、调整超参数或使用完全不同的神经元类型来尝试改善曲线拟合。

结论
------------------------

下一个回归教程将测试更强大的脉冲神经元，例如递归 LIF 神经元和脉冲 LSTM，看看它们的对比情况。

如果你喜欢这个项目，请考虑在 GitHub 上给仓库点赞⭐，这是支持它的最简单也是最好的方式。

额外资源
------------------------

-  `在这里查看 snntorch GitHub 项目。 <https://github.com/jeshraghian/snntorch>`__
-  有关 SNN 进行非线性回归的更多细节可以在我们的相应预印本中找到： `Henkes, A.; Eshraghian, J. K.; 和
   Wessels, H. “脉冲神经网络用于非线性回归”，arXiv 预印本 arXiv:2210.03515,
   2022年10月。 <https://arxiv.org/abs/2210.03515>`__
