#!/usr/bin/env python
#-----------------------------------------------------------------------------
# File Name : spikeConv2d.py
# Author: Emre Neftci
#
# Creation Date : Mon 16 Jul 2018 09:56:30 PM MDT
# Last Modified : Mon 01 Oct 2018 04:48:59 PM PDT
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : Apache License, Version 2.0
#-----------------------------------------------------------------------------
from dcll.pytorch_libdcll import *
from dcll.experiment_tools import *
from dcll.load_dvsgestures_sparse import *
import argparse
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser(description='DCLL for DVS gestures')
parser.add_argument('--batchsize', type=int, default=72, metavar='N', help='input batch size for training')
parser.add_argument('--epochs', type=int, default=4500, metavar='N', help='number of epochs to train')
parser.add_argument('--no_save', type=bool, default=False, metavar='N', help='disables saving into Results directory')
parser.add_argument('--spiking', type=bool, default=True, metavar='N', help='Spiking')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--testinterval', type=int, default=20, metavar='N', help='how epochs to run before testing')
parser.add_argument('--lr', type=float, default=1e-9, metavar='N', help='learning rate for Adamax')
parser.add_argument('--alpha', type=float, default=.97, metavar='N', help='Time constant for neuron')
parser.add_argument('--netscale', type=float, default=1., metavar='N', help='scale network size')
parser.add_argument('--alphas', type=float, default=.92, metavar='N', help='Time constant for synapse')
parser.add_argument('--alpharp', type=float, default=.65, metavar='N', help='Time constant for refractory period')
parser.add_argument('--beta', type=float, default=.95, metavar='N', help='Beta2 parameters for Adamax')
parser.add_argument('--arp', type=float, default=1., metavar='N', help='Absolute refractory period in ticks')
parser.add_argument('--lc_ampl', type=float, default=.5, metavar='N', help='magnitude of local classifier init')
parser.add_argument('--valid', action='store_true', default=False, help='Validation mode (only a portion of test cases will be used)')
parser.add_argument('--test_only', type=str, default='',  help='Only test using predefined parameters found under provided directory')
parser.add_argument('--test_offset', type=int, default=0,  help='offset test sample')
parser.add_argument('--random_tau', type=bool, default=True,  help='randomize time constants in convolutional layers')
parser.add_argument('--n_iters_test', type=int, default=1800, metavar='N', help='for how many ms do we present a sample during classification')

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
n_iters_test = args.n_iters_test
dt = 1000 #us
in_channels = 2
ds = 4
im_dims = im_width, im_height = (128//ds, 128//ds)
out_channels_1 = int(64*args.netscale)
out_channels_2 = int(128*args.netscale)
out_channels_3 = int(128*args.netscale)
out_channels_4 = int(256*args.netscale)
out_channels_5 = int(512*args.netscale)
out_channels_6 = int(512*args.netscale)
target_size = 11
act=nn.Sigmoid()
device = "cpu"

layer1 = Conv2dDCLLlayer(
        in_channels,
        out_channels = out_channels_1,
        im_dims = im_dims,
        target_size=target_size,
        stride=1,
        pooling=(2,2),
        padding=2,
        kernel_size=7,
        alpha = args.alpha,
        alphas = args.alphas,
        alpharp = args.alpharp,
        wrp = args.arp,
        act = act,
        lc_dropout = .5,
        spiking = args.spiking,
        lc_ampl = args.lc_ampl,
        random_tau = args.random_tau).to(device)

layer2 = Conv2dDCLLlayer(
        in_channels = layer1.out_channels,
        out_channels = out_channels_2,
        im_dims = layer1.get_output_shape(),
        target_size=target_size,
        pooling=1,
        padding=2,
        kernel_size=7,
        alpha = args.alpha,
        alphas = args.alphas,
        alpharp = args.alpharp,
        wrp = args.arp,
        act = act,
        lc_dropout = .5,
        spiking = args.spiking,
        lc_ampl = args.lc_ampl,
        random_tau = args.random_tau).to(device)

layer3 = Conv2dDCLLlayer(
        in_channels = layer2.out_channels,
        out_channels = out_channels_3,
        im_dims = layer2.get_output_shape(),
        target_size=target_size,
        pooling=(2,2),
        padding=2,
        kernel_size=7,
        alpha = args.alpha,
        alphas = args.alphas,
        alpharp = args.alpharp,
        wrp = args.arp,
        act = act,
        lc_dropout = .5,
        spiking = args.spiking,
        lc_ampl = args.lc_ampl,
        random_tau = args.random_tau,
        output_layer=False).to(device)



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
        burnin = 50)

dcll_slices[1] = DCLLClassification(
        dclllayer = layer2,
        name = 'conv2d_layer2',
        batch_size = batch_size,
        loss = loss,
        optimizer = opt,
        kwargs_optimizer = opt_param,
        collect_stats = True,
        burnin = 50)

dcll_slices[2] = DCLLClassification(
        dclllayer = layer3,
        name = 'conv2d_layer3',
        batch_size = batch_size,
        loss = loss,
        optimizer = opt,
        kwargs_optimizer = opt_param,
        collect_stats = True,
        burnin = 50)

#Load data
gen_train, _ = create_data(
        batch_size = batch_size,
        chunk_size = n_iters,
        size = [in_channels, im_width, im_height],
        ds = ds,
        dt = dt)


_, gen_test = create_data(
        batch_size = batch_size,
        chunk_size = n_iters_test,
        size = [in_channels, im_width, im_height],
        ds = ds,
        dt = dt)

def generate_test(gen_test, n_test:int, offset=0):
    input_test, labels_test = gen_test.next(offset=offset)
    input_tests = []
    labels1h_tests = []
    n_test = min(n_test,int(np.ceil(input_test.shape[0]/batch_size)))
    for i in range(n_test):
        input_tests.append( torch.Tensor(input_test.swapaxes(0,1))[:,i*batch_size:(i+1)*batch_size].reshape(n_iters_test,-1,in_channels,im_width,im_height))
        labels1h_tests.append(torch.Tensor(labels_test[:,i*batch_size:(i+1)*batch_size]))
    return n_test, input_tests, labels1h_tests


acc_train = []

if __name__ == "__main__":
    from tensorboardX import SummaryWriter
    import datetime,socket,os
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment='' #str(args)[10:].replace(' ', '_')
    log_dir = os.path.join('runs_args/', 'pytorch_conv3L_dvsgestures_args_', current_time + '_' + socket.gethostname() +'_' + comment, )
    print(log_dir)

    if args.test_only=='':
        writer = SummaryWriter(log_dir = log_dir)

    n_test, input_tests, labels1h_tests = generate_test(gen_test, n_test=1 if args.valid else 100, offset = args.test_offset)
    print('Ntest is :')
    print(n_test)


    [s.init(batch_size, init_states = True) for s in dcll_slices]

    acc_test = np.empty([n_tests_total,n_test,len(dcll_slices)])

    if args.test_only=='':
        if not args.no_save:
            d = mksavedir(pre='Results_dvsgestures/')
            annotate(d, text = log_dir, filename= 'log_filename')
            annotate(d, text = str(args), filename= 'args')
            save_source(d)
            with open(os.path.join(d, 'args.pkl'), 'wb') as fp:
                pickle.dump(vars(args), fp)

        for epoch in range(n_epochs):
            print(epoch)
            if ((epoch+1)%1000)==0:
                dcll_slices[0].optimizer.param_groups[-1]['lr']/=5
                dcll_slices[1].optimizer.param_groups[-1]['lr']/=5
                dcll_slices[2].optimizer.param_groups[-1]['lr']/=5
                #dcll_slices[2].optimizer2.param_groups[-1]['lr']/=5
                print("Adjusted parameters")

            input, labels = gen_train.next()
            input = torch.Tensor(input.swapaxes(0,1)).reshape(n_iters,batch_size,in_channels,im_width,im_height)
            labels1h = torch.Tensor(labels)

            [s.init(batch_size, init_states = False) for s in dcll_slices]

            for iter in tqdm(range(n_iters)):
                output1 = dcll_slices[0].train_dcll(input[iter].to(device),labels1h[iter].to(device), regularize = .001)[0]
                output2 = dcll_slices[1].train_dcll(output1,    labels1h[iter].to(device)           , regularize = .001)[0]
                output3 = dcll_slices[2].train_dcll(output2,    labels1h[iter].to(device)           , regularize = .001)[0]

            if (epoch%n_test_interval)==1:

                print('TEST Epoch {0}: '.format(epoch))
                for i in range(n_test):
                    [s.init(batch_size, init_states = False) for s in dcll_slices]

                    for iter in range(n_iters_test):
                        output1 = dcll_slices[0].forward(input_tests[i][iter].to(device))[0]
                        output2 = dcll_slices[1].forward(output1   )[0]
                        output3 = dcll_slices[2].forward(output2   )[0]

                    acc_test[epoch//n_test_interval,i,:] = [ s.accuracy(labels1h_tests[i].to(device)) for s in dcll_slices]
                    acc__test_print =  ' '.join(['L{0} {1:1.3}'.format(i,v) for i,v in enumerate(acc_test[epoch//n_test_interval,i])])
                    print('TEST Epoch {0} Batch {1}:'.format(epoch, i) + acc__test_print)
                    [s.write_stats(writer, label = 'test/', epoch = epoch) for s in dcll_slices]

        if not args.no_save:
            np.save(d+'/acc_test.npy', acc_test)
            annotate(d, text = "", filename = "best result")
            save_dcllslices(d, dcll_slices)

    else:
        from tqdm import tqdm
        load_dcllslices(args.test_only,dcll_slices)

        doutput1=[]
        doutput2=[]
        doutput3=[]
        doutput4=[]

        for i in range(n_test):
            [s.init(batch_size, init_states = False) for s in dcll_slices]

            for iter in tqdm(range(n_iters_test)):
                output41,output42,output43,output44 = dcll_slices[0].forward(input_tests[i][iter].to(device))
                output2 = dcll_slices[1].forward(output41   )[0]
                output3 = dcll_slices[2].forward(output2   )[0]
                if i==0: doutput1.append(output41[0].detach().cpu().numpy())
                if i==0: doutput2.append(output42[0].detach().cpu().numpy())
                if i==0: doutput3.append(output43[0].detach().cpu().numpy())
                if i==0: doutput4.append(output44[0].detach().cpu().numpy())

            acc_test_only = [ s.accuracy(labels1h_tests[i].to(device)) for s in dcll_slices]
            acc__test_print =  ' '.join(['L{0} {1:1.3}'.format(j,v) for j,v in enumerate(acc_test_only)])
            print(acc__test_print)
            print('Aggregate: {}'.format(accuracy_by_vote(np.concatenate([np.array(d.clout) for d in dcll_slices[2:]]),labels1h_tests[i])))
        np.save(args.test_only+'doutput1.npy',np.array(doutput1))
        np.save(args.test_only+'doutput2.npy',np.array(doutput2))
        np.save(args.test_only+'doutput3.npy',np.array(doutput3))
        np.save(args.test_only+'doutput4.npy',np.array(doutput4))
        np.save(args.test_only+'clout1.npy',np.array(dcll_slices[0].clout))
        np.save(args.test_only+'clout2.npy',np.array(dcll_slices[1].clout))
        np.save(args.test_only+'clout3.npy',np.array(dcll_slices[2].clout))
        np.save(args.test_only+'testlabels.npy', labels1h_tests[i])
        np.save(args.test_only+'testinputrate.npy', input_tests[i].reshape(n_iters_test,-1).mean(1).cpu().numpy())
        np.save(args.test_only+'testinput.npy', input_tests[i][:,0:10].cpu().numpy())
