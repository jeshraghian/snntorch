================
简介
================


.. image:: https://github.com/jeshraghian/snntorch/actions/workflows/build.yml/badge.svg
        :target: https://snntorch.readthedocs.io/en/latest/?badge=latest

.. image:: https://readthedocs.org/projects/snntorch/badge/?version=latest
        :target: https://snntorch.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/discord/906036932725841941
        :target: https://discord.gg/cdZb5brajb
        :alt: Discord

.. image:: https://img.shields.io/pypi/v/snntorch.svg
         :target: https://pypi.python.org/pypi/snntorch

.. image:: https://img.shields.io/conda/vn/conda-forge/snntorch.svg
        :target: https://anaconda.org/conda-forge/snntorch

.. image:: https://static.pepy.tech/personalized-badge/snntorch?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads
        :target: https://pepy.tech/project/snntorch

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/snntorch_alpha_scaled.png?raw=true
        :align: center
        :width: 700


* `English <https://snntorch.readthedocs.io/en/latest/readme.html>`_


想要开发更高效的神经网络, 大脑是寻找灵感的绝佳场所。snnTorch 是一个 Python 软件包, 用于利用脉冲神经网络执行基于梯度的学习。
它扩展了 PyTorch 的功能，利用其 GPU 加速张量计算的优势，并将其应用于脉冲神经元网络。
预先设计的脉冲神经元模型可无缝集成到 PyTorch 框架中，并可被视为递归激活单元。

.. image:: https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/spike_excite_alpha_ps2.gif?raw=true
        :align: center
        :width: 800

如果你喜欢这个项目, 请考虑在Github上点亮星星⭐。因为这是最简单、最好的支持方式。

如果您有关于脉冲神经网络训练的问题、意见或建议，可以在我们的 `discord <https://discord.gg/cdZb5brajb>`_ 中讨论.

snnTorch的结构
^^^^^^^^^^^^^^^^^^^^^^^^
snnTorch包含以下组件: 

.. list-table::
   :widths: 20 60
   :header-rows: 1

   * - 组件
     - 描述
   * - `snntorch <https://snntorch.readthedocs.io/en/latest/snntorch.html>`_
     - 类似 torch.nn 的脉冲神经元库，与 autograd 深度集成
   * - `snntorch.export <https://snntorch.readthedocs.io/en/latest/snntorch.export.html>`_
     - 通过 `NIR <https://nnir.readthedocs.io/en/latest/>`_ 实现与其他 SNN 库的交叉兼容
   * - `snntorch.functional <https://snntorch.readthedocs.io/en/latest/snntorch.functional.html>`_
     - 对脉冲进行常见的算术运算，如损耗、正则化等。
   * - `snntorch.spikegen <https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html>`_
     - 脉冲生成和数据转换库
   * - `snntorch.spikeplot <https://snntorch.readthedocs.io/en/latest/snntorch.spikeplot.html>`_
     - 使用 matplotlib 和 Celluloid 实现基于脉冲数据的可视化工具
   * - `snntorch.surrogate <https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html>`_
     - 可选的梯度替代函数
   * - `snntorch.utils <https://snntorch.readthedocs.io/en/latest/snntorch.utils.html>`_
     - 数据集效用函数

snnTorch 的设计旨在与 PyTorch 配合使用，就好像每个脉冲神经元只是层序列中的另一个激活。
因此，它与全连接层、卷积层、残差连接等无关。

目前，神经元模型由递归函数表示，因此无需存储系统中所有神经元的膜电位轨迹来计算梯度。
snnTorch 的精简要求使小型和大型网络都能根据需要在 CPU 上进行可行的训练。
只要将网络模型和张量加载到 CUDA 上, snnTorch 就能像 PyTorch 一样利用 GPU 加速。


引用 
^^^^^^^^^^^^^^^^^^^^^^^^
如果您发现这些资源或代码对您的工作有用，请考虑引用以下文献:

`Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu “Training
Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9)
September 2023. <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. code-block:: bash

  @article{eshraghian2021training,
          title   =  {Training spiking neural networks using lessons from deep learning},
          author  =  {Eshraghian, Jason K and Ward, Max and Neftci, Emre and Wang, Xinxin 
                      and Lenz, Gregor and Dwivedi, Girish and Bennamoun, Mohammed and 
                     Jeong, Doo Seok and Lu, Wei D},
          journal = {Proceedings of the IEEE},
          volume  = {111},
          number  = {9},
          pages   = {1016--1054},
          year    = {2023}
  }

如果您在任何有趣的工作、研究或博客中使用了 snnTorch, 请告诉我们, 我们很乐意听到更多相关信息！请发送电子邮件至 snntorch@gmail.com。

环境配置 
^^^^^^^^^^^^^^^^^^^^^^^^
需要安装以下库来使用snnTorch:

* torch >= 1.1.0
* numpy >= 1.17
* pandas
* matplotlib
* math
* nir
* nirtorch

如果使用 pip 命令安装了 snnTorch, 它们会自动安装。请确保为系统安装了正确版本的 torch, 以实现 CUDA 兼容性。

安装
^^^^^^^^^^^^^^^^^^^^^^^^

运行以下pip代码来安装:

.. code-block:: bash

  $ python
  $ pip install snntorch

要从源代码安装 snnTorch, 请运行::

  $ git clone https://github.com/jeshraghian/snnTorch
  $ cd snntorch
  $ python setup.py install


使用Conda安装::

    $ conda install -c conda-forge snntorch

使用 Graphcore 的加速器安装基于智能处理单元 (IPU) 的构建::

  $ pip install snntorch-ipu
    

API & 案例 
^^^^^^^^^^^^^^^^^^^^^^^^
一个完整的API在 `这里 <https://snntorch.readthedocs.io/>`_ 获取。其中也有提供案例、教程和 Colab 笔记本。



Quickstart 
^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/quickstart.ipynb


以下是开始使用 snnTorch 的几种方法：


* `快速入门笔记 (Colab)`_

* `API参考`_ 

* `案例`_

* `教程`_

.. _快速入门笔记 (Colab): https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/quickstart.ipynb
.. _API参考: https://snntorch.readthedocs.io/
.. _案例: https://snntorch.readthedocs.io/en/latest/examples.html
.. _教程: https://snntorch.readthedocs.io/en/latest/tutorials/index.html


有关运行 snnTorch 的快速示例，请参阅以下代码，或测试 快速入门笔记：


.. code-block:: python

  import torch, torch.nn as nn
  import snntorch as snn
  from snntorch import surrogate
  from snntorch import utils

  num_steps = 25 # number of time steps
  batch_size = 1 
  beta = 0.5  # neuron decay rate 
  spike_grad = surrogate.fast_sigmoid() # surrogate gradient

  net = nn.Sequential(
        nn.Conv2d(1, 8, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
        nn.Conv2d(8, 16, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 10),
        snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True)
        )

  data_in = torch.rand(num_steps, batch_size, 1, 28, 28) # random input data
  spike_recording = [] # record spikes over time
  utils.reset(net) # reset/initialize hidden states for all neurons

  for step in range(num_steps): # loop over time
      spike, state = net(data_in[step]) # one time step of forward-pass
      spike_recording.append(spike) # record spikes in list


深入了解 SNN
^^^^^^^^^^^^^^^^^^^^^^^^^^^
如果您想学习训练脉冲神经网络的所有基础知识, 从神经元模型到神经代码, 直至反向传播, snnTorch 系列教程是您开始学习的好地方。
它由交互式笔记本组成，配有完整的解释，可以让你快速掌握。


.. list-table::
   :widths: 20 60 30
   :header-rows: 1

   * - Tutorial
     - Title
     - Colab Link
   * - `Tutorial 1 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html>`_
     - Spike Encoding with snnTorch
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb

   * - `Tutorial 2 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html>`_
     - The Leaky Integrate and Fire Neuron
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_2_lif_neuron.ipynb

   * - `Tutorial 3 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html>`_
     -  A Feedforward Spiking Neural Network
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_3_feedforward_snn.ipynb


   * - `Tutorial 4 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_4.html>`_
     -  2nd Order Spiking Neuron Models (Optional)
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_4_advanced_neurons.ipynb

  
   * - `Tutorial 5 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html>`_
     -  Training Spiking Neural Networks with snnTorch
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_5_FCN.ipynb
   

   * - `Tutorial 6 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html>`_
     - Surrogate Gradient Descent in a Convolutional SNN
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_6_CNN.ipynb

   * - `Tutorial 7 <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html>`_
     - Neuromorphic Datasets with Tonic + snnTorch
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_7_neuromorphic_datasets.ipynb

.. list-table::
   :widths: 70 40
   :header-rows: 1

   * - Advanced Tutorials
     - Colab Link

   * - `Population Coding <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_pop.html>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_pop.ipynb

   * - `Regression: Part I - Membrane Potential Learning with LIF Neurons <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_1.html>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_1.ipynb

   * - `Regression: Part II - Regression-based Classification with Recurrent LIF Neurons <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_regression_2.html>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :alt: Open In Colab
        :target: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_regression_2.ipynb

   * - `Accelerating snnTorch on IPUs <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_ipu_1.html>`_
     -       —

智能处理单元 (IPU) 加速
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

snnTorch已经针对 `Graphcore's IPU 加速器 <https://www.graphcore.ai/>`_进行了优化 

安装基于IPU的snnTorch::

  $ pip install snntorch-ipu

首次调用 :code:`import snntorch` 时, 将自动编译与 IPU 兼容的低级自定义操作。


更新 Poplar SDK 时，这些操作可能需要重新编译。
这可以通过重新安装 :code:`snntorch-ipu`，或删除基本目录中扩展名为 .so 的文件来实现。

:code:`snntorch.backprop` 模块以及 :code:`snntorch.functional` 和 :code:`snntorch.surrogate` 中的几个函数与 IPU 不兼容，但可以使用 PyTorch 基元重新创建。

更多要求包括:

* poptorch 
* The Poplar SDK 

参考 `Graphcore的文档 <https://github.com/graphcore/poptorch>`_ 以获取 poptorch 和 Poplar SDK 的安装说明。

The homepage for the snnTorch IPU project can be found `here <https://github.com/vinniesun/snntorch-ipu>`__.
A tutorial for training SNNs is provided `here <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_ipu_1.html>`__.


发电
^^^^^^^^^^^^^^^^^^^^^^^^
如果你想对snnTorch动动手脚, 可以在 `这里`_ 找到一些指导。

.. _这里: https://snntorch.readthedocs.io/en/latest/contributing.html

项目信息
^^^^^^^^^^^^^^^^^^^^^^^^
snnTorch目前由 `UCSC Neuromorphic Computing Group <https://ncg.ucsc.edu>`_ 维护。它最初是由 `Jason K. Eshraghian`_ 于 `Lu Group (University of Michigan)`_开发的

其他贡献者包括 `Vincent Sun <https://github.com/vinniesun>`_, `Peng Zhou <https://github.com/pengzhouzp>`_, `Ridger Zhu <https://github.com/ridgerchu>`_, `Alexander Henkes <https://github.com/ahenkes1>`_, Xinxin Wang, Sreyes Venkatesh, and Emre Neftci.

.. _Jason K. Eshraghian: https://jasoneshraghian.com
.. _Lu Group (University of Michigan): https://lugroup.engin.umich.edu/


许可和版权
^^^^^^^^^^^^^^^^^^^^^^^^
snnTorch 源代码根据 MIT 许可条款发布。
snnTorch 的文档采用 署名-非商业性使用-相同方式共享 协议文本进行许可。 (`CC BY-SA 3.0 <https://creativecommons.org/licenses/by-sa/3.0/>`_).
