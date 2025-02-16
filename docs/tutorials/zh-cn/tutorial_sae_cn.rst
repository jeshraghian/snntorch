.. container:: cell markdown

   ` <https://github.com/jeshraghian/snntorch/>`__

   .. rubric:: snnTorch - 使用卷积脉冲神经网络的脉冲自编码器（SAE）
      :name: snntorch---spiking-autoencoder-sae-using-convolutional-spiking-neural-networks

   .. rubric:: 教程作者 Alon Loeffler (www.alonloeffler.com)
      :name: tutorial-by-alon-loeffler-wwwalonloefflercom

   \*本教程改编自我在 Medium.com 上发布的原创文章

   ` <https://github.com/jeshraghian/snntorch/>`__
   ` <https://github.com/jeshraghian/snntorch/>`__

.. container:: cell markdown

   如果您想了解 SNN 是如何工作的，以及底层发生了什么, `您可能会对这里提供的 snnTorch 教程系列感兴趣。 <https://snntorch.readthedocs.io/en/latest/tutorials/index.html>`__ 
   
   snnTorch 教程系列基于以下论文。如果您在工作中发现这些资源或代码有用，请考虑引用以下来源：

    `Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish
    Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, 和 Wei D. Lu. “使用深度学习经验训练脉冲神经网络”。IEEE 会议记录，111(9) 2023年9月。 <https://ieeexplore.ieee.org/abstract/document/10242251>`_

.. container:: cell markdown

   在本教程中，您将学习如何使用 snnTorch 来：

   -  创建一个脉冲自编码器
   -  重构 MNIST 图像

   如果在 Google Colab 中运行：

   -  您可以通过检查 ``Runtime`` >
      ``Change runtime type`` > ``Hardware accelerator: GPU`` 连接到 GPU

.. container:: cell markdown

   .. rubric:: 1. 自编码器
      :name: 1-autoencoders

   | 自编码器是一种被训练用来重构其输入数据的神经网络。它包括两个主要部分：1）编码器
   | 2）解码器

   编码器接收输入数据（例如图像），并将其映射到较低维度的潜在空间。
   
   例如，编码器可能将 28 x 28 像素的 MNIST 图像（总共 784 像素）作为输入，并在将其压缩为较小维度（例如 32 个特征）的同时提取图像的重要特征。这种压缩的图像表示称为 *潜在表示(latent representation)*。

   解码器将潜在表示映射回原始输入空间（即从 32 个特征回到 784 个像素），并尝试从少量关键特征重构原始图像。

   .. raw:: html

      <center>
         <figure>
      <img src='https://miro.medium.com/max/828/0*dHxZ5LCq5kEPltWH.webp' width="600">
      <figcaption> 一个简单的自编码器示例，其中 x 是输入数据，z 是编码的潜在空间，x' 是解码后的重构输入（来源：Wikipedia）。 </figcaption>
                  </figure>
      </center>

   自编码器的目标是最小化输入数据和解码器输出之间的重构误差。

   这是通过训练模型来最小化重构损失来实现的，这通常被定义为输入和重构输出之间的均方误差（MSE）。

   .. raw:: html

      <center>
         <figure>
      <img src='https://miro.medium.com/max/640/1*kjfms6RCnHVMLRSq75AD0Q.webp'>
      <figcaption> MSE 损失方程。在这里，$y$ 将代表原始图像（y true），$\hat{y}$ 将代表重构输出（y pred）（来源：Towards Data Science）。 </figcaption>
                  </figure>
      </center>

   自编码器是通过在重构过程中只找到数据的重要部分并丢弃其他所有内容，从而减少数据中的噪声的绝佳工具。这实际上是一种降维工具。

.. container:: cell markdown

   在本教程中（类似于教程 1），我们将假设我们有一些非脉冲输入数据（即 MNIST 数据集），我们希望对其进行编码和重构。
   
   那么，让我们开始吧！

.. container:: cell markdown

   .. rubric:: 2. 设置
      :name: 2-setting-up

.. container:: cell markdown

   .. rubric:: 2.1 安装/导入包并设置环境
      :name: 21-installimport-packages-and-set-up-environment

.. container:: cell markdown

   首先，我们需要安装 snnTorch 及其依赖项（请注意，本教程假设您已经安装了 pytorch 和 torchvision - 这些在 Colab 中预先安装）。您可以通过运行以下命令来实现：

.. container:: cell code

   .. code:: python

      !pip install snntorch

.. container:: cell markdown

   接下来，让我们导入必要的模块并设置 SAE 模型。

   我们可以使用 pyTorch 定义编码器和解码器网络，并使用 snnTorch 将网络中的神经元转换为漏积分触发（LIF）神经元，它们可以读取和输出脉冲。

   我们将使用卷积神经网络（CNN），在教程 6 中介绍过，作为编码器和解码器的基础。

.. container:: cell code

   .. code:: python

      import os

      import torch
      import torch.nn as nn
      import torch.nn.functional as F

      from torchvision import datasets, transforms
      from torch.utils.data import DataLoader
      from torchvision import utils as utls

      import snntorch as snn
      from snntorch import utils
      from snntorch import surrogate

      import numpy as np

      #定义 SAE 模型：
      class SAE(nn.Module):
          def __init__(self,latent_dim):
              super().__init__()
              self.latent_dim = latent_dim #编码的 z 空间数据的维度

.. container:: cell markdown

   .. rubric:: 3. 构建自编码器
      :name: 3-building-the-autoencoder

.. container:: cell markdown

   .. rubric:: 3.1 DataLoaders
      :name: 31-dataloaders

   我们将使用 MNIST 数据集

.. container:: cell code

   .. code:: python

      # dataloader 参数
      batch_size = 250
      data_path='/tmp/data/mnist'

      dtype = torch.float
      device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

.. container:: cell code

   .. code:: python

      # 定义转换
      input_size = 32 #为了这个教程，我们将把原始的 MNIST 从 28 调整到 32

      transform = transforms.Compose([
                  transforms.Resize((input_size, input_size)),
                  transforms.Grayscale(),
                  transforms.ToTensor(),
                  transforms.Normalize((0,), (1,))])

      # 加载 MNIST

      # 训练数据
      train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transform, download=True)
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

      # 测试数据
      test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transform, download=True)
      test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

.. container:: cell markdown

   .. rubric:: 3.2 编码器
      :name: 32-the-encoder

   让我们开始构建我们逐渐组合在一起的自编码器的部分：

.. container:: cell markdown

   首先，让我们添加一个带有三个卷积层（``nn.Conv2d``）和一个全连接线性输出层的编码器。

   -  我们将使用大小为 3 的内核，填充为 1，步长为 2 的 CNN 超参数。

   -  我们还在卷积层之间添加了一个批量归一化层。由于我们将使用神经元膜电位作为每个神经元的输出，归一化将有助于我们的训练过程。


.. container:: cell code

   .. code:: python

      #定义 SAE 模型：
      class SAE(nn.Module):
          def __init__(self):
              super().__init__()
              self.latent_dim = latent_dim #编码的 z 空间数据的维度
              
              # 编码器
              self.encoder = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, stride=2), # 卷积层 1
                                  nn.BatchNorm2d(32),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh), #SNN TORCH LIF 神经元
                                  nn.Conv2d(32, 64, 3, padding=1, stride=2), # 卷积层 2
                                  nn.BatchNorm2d(64),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                  nn.Conv2d(64, 128, 3, padding=1, stride=2), # 卷积层 3
                                  nn.BatchNorm2d(128),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                  nn.Flatten(start_dim=1, end_dim=3), # 展平卷积输出
                                  nn.Linear(128*4*4, latent_dim), # 全连接线性层
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, threshold=thresh)
                                  )

.. container:: cell markdown

   .. rubric:: 3.3 解码器
      :name: 33-the-decoder

.. container:: cell markdown

   在编写解码器之前，还需要一个小步骤。
   当解码 z 空间数据时，我们需要将平面的编码表示（latent_dim）转换回用于反卷积的张量表示。

   为此，我们需要运行一个额外的全连接线性层，将数据转换回 128 x 4 x 4 的张量。

.. container:: cell code

   .. code:: python

      #定义 SAE 模型：
      class SAE(nn.Module):
          def __init__(self, latent_dim):
              super().__init__()
              self.latent_dim = latent_dim #编码的 z 空间数据的维度
              
              # 编码器
              self.encoder = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, stride=2), # 卷积层 1
                                  nn.BatchNorm2d(32),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh), #SNN TORCH LIF 神经元
                                  nn.Conv2d(32, 64, 3, padding=1, stride=2), # 卷积层 2
                                  nn.BatchNorm2d(64),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                  nn.Conv2d(64, 128, 3, padding=1, stride=2), # 卷积层 3
                                  nn.BatchNorm2d(128),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                  nn.Flatten(start_dim=1, end_dim=3), # 展平卷积输出
                                  nn.Linear(128*4*4, latent_dim), # 全连接线性层
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, threshold=thresh)
                                  )

              # 从潜在空间转换回张量用于卷积
              self.linearNet = nn.Sequential(nn.Linear(latent_dim, 128*4*4),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, threshold=thresh))
              # 解码器
              self.decoder = nn.Sequential(nn.Unflatten(1, (128, 4, 4)), # 将 1 维数据解平到 128 x 4 x 4 的张量
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                  nn.ConvTranspose2d(128, 64, 3, padding=1, stride=(2, 2), output_padding=1),
                                  nn.BatchNorm2d(64),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                  nn.ConvTranspose2d(64, 32, 3, padding=1, stride=(2, 2), output_padding=1),
                                  nn.BatchNorm2d(32),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                  nn.ConvTranspose2d(32, 1, 3, padding=1, stride=(2, 2), output_padding=1),
                                  snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, threshold=20000) #设置较高以便可以训练膜电位
                                  )

.. container:: cell markdown

   需要注意的重要一点是，在最后一个 Leaky 层中，我们的 spiking 阈值（``thresh``）被设置得非常高。这是 snnTorch 中的一个巧妙技巧，允许最后一层的神经元膜持续更新，而从不达到 spiking 阈值。

   每个 Leaky 神经元的输出将包括一个脉冲（0或1）的张量和一个神经元膜电位（负或正实数）的张量。snnTorch 允许我们在训练中使用每个神经元的脉冲或膜电位输出。我们将使用最后一层的膜电位输出来进行图像重建。

.. container:: cell markdown

   .. rubric:: 3.4 前向函数
      :name: 34-forward-function

   最后，让我们编写前向、编码和解码函数，然后将它们整合在一起。

.. container:: cell code

   .. code:: python

      def forward(self, x): 
          utils.reset(self.encoder) #需要重置 LIF 的隐藏状态
          utils.reset(self.decoder)
          utils.reset(self.linearNet)
          
          #编码
          spk_mem=[];spk_rec=[];encoded_x=[]
          for step in range(num_steps): #对于时间 t
              spk_x,mem_x=self.encode(x) #输出脉冲列和神经元膜状态
              spk_rec.append(spk_x)
              spk_mem.append(mem_x)
          spk_rec=torch.stack(spk_rec,dim=2) #在第二个张量维度上堆叠脉冲
          spk_mem=torch.stack(spk_mem,dim=2) #在第二个张量维度上堆叠膜电位
          
          #解码
          spk_mem2=[];spk_rec2=[];decoded_x=[]
          for step in range(num_steps): #对于时间 t
              x_recon,x_mem_recon=self.decode(spk_rec[...,step])
              spk_rec2.append(x_recon)
              spk_mem2.append(x_mem_recon)
          spk_rec2=torch.stack(spk_rec2,dim=4)
          spk_mem2=torch.stack(spk_mem2,dim=4)  
          out = spk_mem2[:,:,:,:,-1] #返回最后时间点 t=-1 的输出神经元膜电位
          return out

      def encode(self,x):
          spk_latent_x,mem_latent_x=self.encoder(x)
          return spk_latent_x,mem_latent_x

      def decode(self,x):
          spk_x,mem_x = self.linearNet(x) #将潜在维度转换回编码器最后一层的总特征大小
          spk_x2,mem_x2=self.decoder(spk_x)
          return spk_x2,mem_x2

.. container:: cell markdown

   这里有几点需要注意：

   1) 在我们的前向函数的每次调用开始时，我们需要重置每个 LIF 神经元的隐藏权重。如果不这样做，我们将在尝试反向传播时遇到奇怪的梯度错误。我们使用 ``utils.reset`` 来完成这个操作。

   2) 在前向函数中，当我们调用编码和解码函数时，我们需要在循环中进行。这是因为我们正在将静态图像转换成脉冲列，正如前面所解释的。脉冲列需要一个时间 t，脉冲可以在这个时间内发生或不发生。因此，我们对原始图像进行了 :math:`t`（或 ``num_steps``）次编码和解码，以创建一个潜在表示 :math:`z`。

.. container:: cell markdown

   例如，将 MNIST 数据集中的样本数字 7 转换为具有 32 维潜在空间和 t=50 的脉冲列，可能看起来像这样：编码后的样本 MNIST 数字 7 的脉冲列。其他实例的 7 将有略微不同的脉冲列，不同数字将有更多不同的脉冲列。

.. container:: cell markdown

   .. rubric:: 3.5 整合所有内容：
      :name: 35-putting-it-all-together

   我们最终的、完整的 SAE 类应该看起来像这样：

.. container:: cell code

   .. code:: python

      class SAE(nn.Module):
          def __init__(self):
              super().__init__()
              #编码器
              self.encoder = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, stride=2),
                                nn.BatchNorm2d(32),
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                nn.Conv2d(32, 64, 3, padding=1, stride=2),
                                nn.BatchNorm2d(64),
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                nn.Conv2d(64, 128, 3, padding=1, stride=2),
                                nn.BatchNorm2d(128),
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                nn.Flatten(start_dim=1, end_dim=3),
                                nn.Linear(2048, latent_dim), #这应该是最后一层的输出大小（通道 * 像素 * 像素）
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, threshold=thresh)
                                )
              # 从潜在空间转换回张量用于卷积
              self.linearNet = nn.Sequential(nn.Linear(latent_dim, 128*4*4),
                                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, threshold=thresh))        
              #解码器
              
              self.decoder = nn.Sequential(nn.Unflatten(1, (128, 4, 4)), 
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                nn.ConvTranspose2d(128, 64, 3, padding=1, stride=(2,2), output_padding=1),
                                nn.BatchNorm2d(64),
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                nn.ConvTranspose2d(64, 32, 3, padding=1, stride=(2,2), output_padding=1),
                                nn.BatchNorm2d(32),
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=thresh),
                                nn.ConvTranspose2d(32, 1, 3, padding=1, stride=(2,2), output_padding=1),
                                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, threshold=20000) #设置阈值很高，使膜电位可以被训练
                                )
              
          def forward(self, x): #尺寸：[批量，通道，宽度，长度]
              utils.reset(self.encoder) #需要重置 LIF 的隐藏状态
              utils.reset(self.decoder)
              utils.reset(self.linearNet) 
              
              #编码
              spk_mem=[];spk_rec=[];encoded_x=[]
              for step in range(num_steps): #对于时间 t
                  spk_x,mem_x=self.encode(x) #输出脉冲列和神经元膜状态
                  spk_rec.append(spk_x)
                  spk_mem.append(mem_x)
              spk_rec=torch.stack(spk_rec,dim=2)
              spk_mem=torch.stack(spk_mem,dim=2) #尺寸：[批量，通道，宽度，长度，时间]
              
              #解码
              spk_mem2=[];spk_rec2=[];decoded_x=[]
              for step in range(num_steps): #对于时间 t
                  x_recon,x_mem_recon=self.decode(spk_rec[...,step])
                  spk_rec2.append(x_recon)
                  spk_mem2.append(x_mem_recon)
              spk_rec2=torch.stack(spk_rec2,dim=4)
              spk_mem2=torch.stack(spk_mem2,dim=4) #尺寸：[批量，通道，宽度，长度，时间]  
              out = spk_mem2[:,:,:,:,-1] #返回最后时间点 t=-1 的输出神经元膜电位
              return out #尺寸：[批量，通道，宽度，长度]

          def encode(self,x):
              spk_latent_x,mem_latent_x=self.encoder(x)
              return spk_latent_x,mem_latent_x

          def decode(self,x):
              spk_x,mem_x = self.linearNet(x) #将潜在维度转换回编码器最后一层的总特征大小
              spk_x2,mem_x2=self.decoder(spk_x)
              return spk_x2,mem_x2

.. container:: cell markdown

   .. rubric:: 4. 训练和测试
      :name: 4-training-and-testing

   最后，我们可以开始训练我们的 SAE，并测试其有效性。我们已经加载了 MNIST 数据集，并将其分为训练和测试类别。

.. container:: cell markdown

   .. rubric:: 4.1 训练函数
      :name: 41-training-function

   我们定义了训练函数，该函数接收网络模型、训练数据集、优化器和轮数作为输入，并在完成当前轮次的所有批次后返回损失值。

   如开头所述，我们将使用 MSE 损失来比较重建的图像(``x_recon``)和原始图像(``real_img``)。

   与往常一样，为了为反向传播设置梯度，我们使用 ``opti.zero_grad()``，然后调用 ``loss_val.backward()`` 和 ``opti.step()`` 来执行反向传播。

.. container:: cell code

   .. code:: python

      # 训练
      def train(network, trainloader, opti, epoch): 
          
          network=network.train()
          train_loss_hist=[]
          for batch_idx, (real_img, labels) in enumerate(trainloader):   
              opti.zero_grad()
              real_img = real_img.to(device)
              labels = labels.to(device)
              
              # 将数据传入网络，并从 t=-1 时刻的膜电位返回重建的图像
              x_recon = network(real_img) # 传入的尺寸：[批量大小，输入大小，图像宽度，图像长度]
              
              # 计算损失        
              loss_val = F.mse_loss(x_recon, real_img)
                      
              print(f'训练[{epoch}/{max_epoch}][{batch_idx}/{len(trainloader)}] 损失: {loss_val.item()}')

              loss_val.backward()
              opti.step()

              # 在每轮结束时保存重建的图像
              if batch_idx == len(trainloader)-1:
                  # 注意：您需要在选择的路径中创建 training/ 和 testing/ 文件夹
                  utls.save_image((real_img+1)/2, f'figures/training/epoch{epoch}_finalbatch_inputs.png') 
                  utls.save_image((x_recon+1)/2, f'figures/training/epoch{epoch}_finalbatch_recon.png')
          return loss_val

.. container:: cell markdown

   .. rubric:: 4.2 测试函数
      :name: 42-testing-function

   测试函数与训练函数几乎相同，区别在于我们不进行反向传播，因此不需要梯度，我们使用 ``torch.no_grad()``。

.. container:: cell code

   .. code:: python

      # 测试
      def test(network, testloader, opti, epoch):
          network=network.eval()
          test_loss_hist=[]
          with torch.no_grad(): # 这次不需要梯度
              for batch_idx, (real_img, labels) in enumerate(testloader):   
                  real_img = real_img.to(device)
                  labels = labels.to(device)
                  x_recon = network(real_img)

                  loss_val = F.mse_loss(x_recon, real_img)

                  print(f'测试[{epoch}/{max_epoch}][{batch_idx}/{len(testloader)}]  损失: {loss_val.item()}')#, RECONS: {recons_meter.avg}, DISTANCE: {dist_meter.avg}')
                      
                  if batch_idx == len(testloader)-1:
                      utls.save_image((real_img+1)/2, f'figures/testing/epoch{epoch}_finalbatch_inputs.png')
                      utls.save_image((x_recon+1)/2, f'figures/testing/epoch{epoch}_finalbatch_recons.png')
          return loss_val

.. container:: cell markdown

   在脉冲神经网络中，计算损失有几种方法。在这里，我们只是取最后的全连接层神经元在最后一个时间步（:math:`t = 5`）的膜电位。

   因此，我们只需每轮对每个原始图像及其对应的解码重建图像进行一次比较。我们也可以返回每个时间步的膜电位，并创建 t 个不同版本的重建图像，然后将每个图像与原始图像进行比较并取平均损失。对此感兴趣的朋友可以用下面的方法替换上面的损失函数：

   (*注意这将无法运行，因为我们还没有定义任何变量，这里仅供示例参考*)

.. container:: cell code

   .. code:: python

      train_loss_hist=[]
      loss_val = torch.zeros((1), dtype=dtype, device=device)
      for step in range(num_steps):
          loss_val += F.mse_loss(x_recon, real_img)
      train_loss_hist.append(loss_val.item())
      avg_loss=loss_val/num_steps

   .. container:: output error

      ::

         ---------------------------------------------------------------------------
         NameError                                 Traceback (most recent call last)
         Cell In[72], line 4
               2 loss_val = torch.zeros((1), dtype=dtype, device=device)
               3 for step in range(num_steps):
         ----> 4     loss_val += F.mse_loss(x_recon, real_img)
               5 train_loss_hist.append(loss_val.item())
               6 avg_loss=loss_val/num_steps

         NameError: name 'x_recon' is not defined

.. container:: cell markdown

   .. rubric:: 5. 结论：运行 SAE
      :name: 5-conclusion-running-the-sae

   现在，我们终于可以运行我们的 SAE 模型了。让我们定义一些参数，并进行训练和测试

.. container:: cell markdown

   让我们创建目录，以便保存训练和测试的原始图像和重建图像：

.. container:: cell code

   .. code:: python

      # 在您选择的路径中创建 training/ 和 testing/ 文件夹
      if not os.path.isdir('figures/training'):
          os.makedirs('figures/training')
          
      if not os.path.isdir('figures/testing'):
          os.makedirs('figures/testing')

.. container:: cell code

   .. code:: python

      # dataloader 参数
      batch_size = 250
      input_size = 32 #调整 mnist 数据大小（可选）

      # 设置 GPU
      dtype = torch.float
      device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

      # 神经元和模拟参数
      spike_grad = surrogate.atan(alpha=2.0)# 替代梯度 fast_sigmoid(slope=25) 
      beta = 0.5 # 神经元衰减率
      num_steps=5
      latent_dim = 32 # 潜在层维度（我们希望的信息压缩程度）
      thresh=1 # 发放阈值（越低，通过的脉冲越多）
      epochs=10 
      max_epoch=epochs

      # 定义网络和优化器
      net=SAE()
      net = net.to(device)

      optimizer = torch.optim.AdamW(net.parameters(), 
                                  lr=0.0001,
                                  betas=(0.9, 0.999), 
                                  weight_decay=0.001)

      # 运行训练和测试        
      for e in range(epochs): 
          train_loss = train(net, train_loader, optimizer, e)
          test_loss = test(net,test_loader,optimizer,e)

   .. container:: output stream stdout

      ::

         训练[0/10][0/240] 损失: 0.10109379142522812
         训练[0/10][1/240] 损失: 0.10465191304683685

   .. container:: output stream stderr

      ::


         KeyboardInterrupt

.. container:: cell markdown

   经过仅仅10个周期的训练和测试，我们的重建损失应该在0.05左右，重建的图像应该看起来如下：

.. container:: cell markdown

.. container:: cell markdown

   是的，重建的图像有些模糊，损失也不是完美的，但从只有10个周期，且仅使用 :math:`t = 5` 时刻的最终膜电位来计算重建损失来看，这已是一个相当不错的开始！

.. container:: cell markdown

   尝试增加训练周期数，或者调整 `thresh`、`num_steps` 和 `batch_size` 的值，看看你是否能获得更好的损失！

