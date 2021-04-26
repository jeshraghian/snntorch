from dcll.pytorch_libdcll import *
from dcll.npamlib import spiketrains
from dcll.experiment_tools import *
from dcll.load_dvsgestures_sparse import *
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='DCLL for DVS gestures')
parser.add_argument('--batchsize', type=int, default=1, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--no_save', action='store_true', default=False, help='disables saving into Results directory')
parser.add_argument('--spiking', type=bool, default=True, metavar='N', help='Spiking')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--testinterval', type=int, default=20, metavar='N', help='how epochs to run before testing')
parser.add_argument('--lr', type=float, default=1e-7, metavar='N', help='learning rate (Adamax)')
parser.add_argument('--alpha', type=float, default=.9, metavar='N', help='Time constant for neuron')
parser.add_argument('--netscale', type=float, default=1.0, metavar='N', help='scale network size')
parser.add_argument('--alphas', type=float, default=.87, metavar='N', help='Time constant for synapse')
parser.add_argument('--alpharp', type=float, default=.9, metavar='N', help='Time constant for refractory period')
parser.add_argument('--beta', type=float, default=.95, metavar='N', help='Beta2 parameters for Adamax')
parser.add_argument('--arp', type=float, default=3, metavar='N', help='Absolute refractory period in ticks')
parser.add_argument('--lc_ampl', type=float, default=.5, metavar='N', help='magnitude of local classifier init')
parser.add_argument('--valid', action='store_true', default=False, help='Validation mode (only a portion of test cases will be used)')
parser.add_argument('--test_only', type=str, default='',  help='Only test using predefined parameters found under provided directory')
parser.add_argument('--test_offset', type=int, default=0,  help='offset test sample')
parser.add_argument('--random_tau', action='store_true', default=False,  help='randomize time constants in convolutional layers')

args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

#Method for computing classification accuracy
acc = accuracy_by_vote

n_epochs = args.epochs
n_test_interval = args.testinterval
n_tests_total = n_epochs//n_test_interval+1
batch_size = args.batchsize
n_iters = 500
n_iters_test = 1800
dt = 1000 #us
ds = 4
in_channels_1 = int(100)
out_channels_1 = int(256*args.netscale)
out_channels_2 = int(256*args.netscale)
out_channels_3 = int(256*args.netscale)
out_channels_4 = int(512*args.netscale)
out_channels_5 = int(1024*args.netscale)
out_channels_6 = int(1024*args.netscale)
target_size = 1
#act=lambda x: nn.Hardtanh(min_val=0)((x/10+.5))
act= nn.Sigmoid()

layer1 = DenseDCLLlayer(
        in_channels_1,
        out_channels = out_channels_1,
        target_size=target_size,
        alpha = args.alpha,
        alphas = args.alphas,
        alpharp = args.alpharp,
        wrp = args.arp,
        act = act,
        lc_dropout = False,
        spiking = args.spiking,
        lc_ampl = args.lc_ampl,
        random_tau = args.random_tau).to(device)

layer2 = DenseDCLLlayer(
        in_channels = layer1.out_channels,
        out_channels = out_channels_2,
        target_size=target_size,
        alpha = args.alpha,
        alphas = args.alphas,
        alpharp = args.alpharp,
        wrp = args.arp,
        act = act,
        lc_dropout = False,
        spiking = args.spiking,
        lc_ampl = args.lc_ampl,
        random_tau = args.random_tau).to(device)

layer3 = DenseDCLLlayer(
        in_channels = layer2.out_channels,
        out_channels = out_channels_3,
        target_size=target_size,
        alpha = args.alpha,
        alphas = args.alphas,
        alpharp = args.alpharp,
        wrp = args.arp,
        act = act,
        lc_dropout = False,
        spiking = args.spiking,
        lc_ampl = args.lc_ampl,
        random_tau = args.random_tau).to(device)

#Adamax parameters { 'betas' : [.0, .99]}
opt = optim.Adamax
opt_param = {'lr':args.lr, 'betas' : [.0, args.beta]}
#opt = optim.SGD
loss = torch.nn.SmoothL1Loss
#loss = torch.nn.CrossEntropyLoss
#opt_param = {'lr':3e-4}

dcll_slices = [None for i in range(3)]
dcll_slices[0] = DCLLClassification(
        dclllayer = layer1,
        name = 'conv2d_layer1',
        batch_size = batch_size,
        loss = loss,
        optimizer = opt,
        kwargs_optimizer = opt_param,
        collect_stats = True,
        burnin = 20)

dcll_slices[1] = DCLLClassification(
        dclllayer = layer2,
        name = 'conv2d_layer2',
        batch_size = batch_size,
        loss = loss,
        optimizer = opt,
        kwargs_optimizer = opt_param,
        collect_stats = True,
        burnin = 20)

dcll_slices[2] = DCLLClassification(
        dclllayer = layer3,
        name = 'conv2d_layer3',
        batch_size = batch_size,
        loss = loss,
        optimizer = opt,
        kwargs_optimizer = opt_param,
        collect_stats = True,
        burnin = 20)

acc_train = []

if __name__ == "__main__":
    [s.init(batch_size, init_states = True) for s in dcll_slices]

    input = torch.Tensor([spiketrains(in_channels_1, rates=100, T=1000)]).transpose(0,1)
    targets_tim = torch.Tensor([[ np.linspace(0,1,n_iters), ]]).transpose(0,2).transpose(1,2)
    targets_cos = torch.Tensor([[ .5*np.cos(np.linspace(0,1,n_iters)*200/2/np.pi), ]]).transpose(0,2).transpose(1,2)
    targets_sin = torch.Tensor([[ .5*np.sin(np.linspace(0,1,n_iters)*100/2/np.pi), ]]).transpose(0,2).transpose(1,2)

    input = input.repeat(1, batch_size, 1)
    targets_tim = targets_tim.repeat(1, batch_size, 1)
    targets_sin = targets_sin.repeat(1, batch_size, 1)
    targets_cos = targets_cos.repeat(1, batch_size, 1)

    for epoch in range(n_epochs):

        [s.init(batch_size, init_states = False) for s in dcll_slices]


        for iter in range(n_iters):
            input_ = input[iter].to(device)
            output1, _, _, _, _ = dcll_slices[0].train_dcll(input_,  targets_tim[iter].to(device), regularize = True)
            output2, _, _, _, _ = dcll_slices[1].train_dcll(output1, targets_cos[iter].to(device), regularize = True)
            output3, _, _, _, _ = dcll_slices[2].train_dcll(output2, targets_sin[iter].to(device), regularize = True)

        res = np.zeros([n_iters, 3, 256])
        loss_ = np.zeros([n_iters, 3])
        hist_loss = []

        if (epoch%5) == 0:
            [s.init(batch_size, init_states = False) for s in dcll_slices]
            [s.eval() for s in dcll_slices]
            for iter in range(n_iters):
                input_ = input[iter].to(device)
                output1, pvoutput1, pv1, _, loss1 = dcll_slices[0].train_dcll(input_,  targets_tim[iter].to(device), do_train=False)
                output2, pvoutput2, pv2, _, loss2 = dcll_slices[1].train_dcll(output1, targets_cos[iter].to(device), do_train=False)
                output3, pvoutput3, pv3, _, loss3 = dcll_slices[2].train_dcll(output2, targets_sin[iter].to(device), do_train=False)
                res[iter,0] = pv1.squeeze().cpu().detach().numpy()[0]
                res[iter,1] = pv2.squeeze().cpu().detach().numpy()[0]
                res[iter,2] = pv3.squeeze().cpu().detach().numpy()[0]
                loss_[iter, 0] = loss1
                loss_[iter, 1] = loss2
                loss_[iter, 2] = loss3
            [s.train() for s in dcll_slices]
            hist_loss.append(loss_.sum(axis=0))
            print('Epoch {:<5}'.format(epoch), 'Loss ',('{:,.4} '*loss_.shape[1]).format(*hist_loss[-1]), res.mean(axis=2).mean(axis=0))

    d = mksavedir()
    annotate(d, text = str(args), filename= 'args')
    save_source(d)
    save(d,args,'args.pkl')
    annotate(d, text = "", filename = "best result")
    save_dcllslices(d, dcll_slices)

    [s.init(batch_size, init_states = False) for s in dcll_slices]
    [s.eval() for s in dcll_slices]

    do1_ = []
    do2_ = []
    do3_ = []
    dwt = []
    deps = []
    for iter in range(n_iters):
        input_ = input[iter].to(device)
        do1 = dcll_slices[0].train_dcll(input_,  targets_tim[iter].to(device), do_train=False)
        do2 = dcll_slices[1].train_dcll(do1[0], targets_cos[iter].to(device), do_train=False)
        do3 = dcll_slices[2].train_dcll(do2[0], targets_sin[iter].to(device), do_train=False)

        dwt.append(dcll_slices[2].dclllayer.i2h.weight.grad.data.detach().cpu().numpy())
        deps.append(dcll_slices[2].dclllayer.i2h.state.eps1.detach().cpu().numpy())

        do1_.append([d.cpu().detach().numpy() for d in do1])
        do2_.append([d.cpu().detach().numpy() for d in do2])
        do3_.append([d.cpu().detach().numpy() for d in do3])

    np.save(d+'do1', do1_)
    np.save(d+'do2', do2_)
    np.save(d+'do3', do3_)
    np.save(d+'loss', hist_loss)
    np.save(d+'dwt', np.array(dwt)*args.lr)
    np.save(d+'deps', np.array(deps))
    np.save(d+'target', [targets_tim.numpy(), targets_cos.numpy(), targets_sin.numpy()])
