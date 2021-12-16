import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from new_layers._3dgcn import Model_3dgcn
from new_layers.gru import GRUModel

def get_xt_from_x(x):

    ''' 
    get it from dataset.py in 3dgcn
    '''


    return x_t

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, _3dgcn_bool=True, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self._3dgcn_bool = _3dgcn_bool         # use 3dgcn
    

        self.gru = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self._3dgconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        ''' 3dgcn has no adj matrix'''
        # self.supports_len = 0
        # if supports is not None:
        #     self.supports_len += len(supports)


        ''' 3dgcn has no adj matrix'''
        # if _3dgcn_bool and addaptadj:
        #     if aptinit is None:
        #         if supports is None:
        #             self.supports = []
        #         self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        #         self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        #         self.supports_len +=1
        #     else:
        #         if supports is None:
        #             self.supports = []
        #         m, p, n = torch.svd(aptinit)
        #         initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
        #         initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
        #         self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
        #         self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
        #         self.supports_len += 1




        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):

                # dilated convolutions
                # self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                #                                    out_channels=dilation_channels,
                #                                    kernel_size=(1,kernel_size),dilation=new_dilation))

                # self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                #                                  out_channels=dilation_channels,
                #                                  kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2

                # if self.gcn_bool:
                #     self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

                
                    
        if self._3dgcn_bool:
            self._3dgconv = Model_3dgcn()

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

        


    def forward(self, input, label_index):

        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        ''' 3dgcn has no adj matrix'''
        # # calculate the current adaptive adj matrix once per iteration
        # new_supports = None
        # if self._3dgcn_bool and self.addaptadj and self.supports is not None:
        #     adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        #     new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x

            # Now GRU(dilated convolution)
            
            gru = GRUModel(residual.size(), self.dropout)
            x = gru(residual)

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            # Now 3dgcn (GCN)
            if self._3dgcn_bool:

                x_t = get_xt_from_x(x)
                x = self._3dgconv(x, x_t, label_index)
                
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            # Batch normalization
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x





