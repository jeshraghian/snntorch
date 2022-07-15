from cv2 import CALIB_FIX_S1_S2_S3_S4
import torch
import torch.nn as nn
from .neurons import *

class Izhikevich(SpikingNeuron):
    def __init__(self, threshold=30.0, spike_grad=None, init_hidden=False, inhibition=False,
                learn_threshold=False, reset_mechanism="subtract", state_quant=False, 
                output=False, a_=0.02, b_=0.2, c_=-65, d_=6, dt_=0.5, c1_=0.04,
                c2_=5, c3_=140, v_rest_=-70, u_rest_=-14):
        super(Izhikevich, self).__init__(
            threshold,
            spike_grad,
            init_hidden,
            inhibition,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
        )

        self.spk = _SpikeTensor(init_flag=False)
        self.v_rest = v_rest_
        self.u_rest = u_rest_
        self.a = a_
        self.b = b_
        self.c = c_
        self.d = d_
        self.c1 = c1_
        self.c2 = c2_
        self.c3 = c3_
        self.dt = dt_

        self.v, self.u = self.init_izhikevich(self.v_rest, self.u_rest)

    def forward(self, input_: torch.Tensor, spk=False, mem=False):
        # self.spk = torch.where(self.v >= self.threshold, 1.0, 0.0)
        # non_active = 1 - self.spk

        dv = (self.c1 * self.v + self.c2)*self.v + self.c3 - self.u
        self.v = self.v + (dv + input_) * self.dt
        du = self.a * (self.b * self.v - self.u)
        self.u = self.u + self.dt*du

        self.spk = torch.where(self.v >= self.threshold, 1.0, 0.0)
        non_active = 1 - self.spk
        
        self.v = non_active*self.v + self.spk*self.c
        self.u = non_active*self.u + self.spk*(self.u + self.d)

        return self.spk, self.v, self.u

