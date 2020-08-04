import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

class RCBlock(nn.Module):
    def __init__(self, feat_dim, ks, dilation, num_groups):
        super().__init__()
        # ks = 3  # kernel size
        ksm1 = ks-1
        mfd = feat_dim
        di = dilation
        self.num_groups = num_groups

        self.relu = nn.LeakyReLU()

        self.rec = nn.GRU(mfd, mfd, num_layers=1, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(mfd, mfd, ks, 1, ksm1*di//2, dilation=di, groups=num_groups)
        self.gn = nn.GroupNorm(num_groups, mfd)

    def init_hidden(self, batch_size, hidden_size):
        num_layers = 1
        num_directions = 2
        hidden = torch.zeros(num_layers*num_directions, batch_size, hidden_size)
        hidden.normal_(0, 1)
        return hidden

    def forward(self, x):
        bs, mfd, nf = x.size()

        hidden = self.init_hidden(bs, mfd).to(x.device)

        r = x.transpose(1, 2)
        r, _ = self.rec(r, hidden)
        r = r.transpose(1, 2).view(bs, 2, mfd, nf).sum(1)
        c = self.relu(self.gn(self.conv(r)))
        x = x+r+c

        return x

class BNSNConv1dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dilation = dilation

        block = [
            spectral_norm(nn.Conv1d(input_dim, output_dim, ks,
                                    1, dilation*ksm1d2, dilation=dilation)),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        x = self.block(x)

        return x

class BNSNConv2dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, frequency_stride, time_dilation):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.time_dilation = time_dilation
        self.frequency_stride = frequency_stride

        block = [
            spectral_norm(nn.Conv2d(
                input_dim, output_dim, ks,
                (frequency_stride, 1),
                (1, time_dilation*ksm1d2),
                dilation=(1, time_dilation))),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        x = self.block(x)
        
        return x

class GBlock(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim, num_groups):
        super().__init__()

        ks = 3  # filter size
        mfd = middle_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mfd = mfd
        self.num_groups = num_groups

        # ### Main body ###
        block = [
            nn.Conv1d(input_dim, mfd, 3, 1, 1),
            nn.GroupNorm(num_groups, mfd),
            nn.LeakyReLU(),
            #RCBlock(mfd, ks, dilation=1, num_groups=num_groups),
            nn.Conv1d(mfd, output_dim, 3, 1, 1),
            
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        # ### Main ###
        x = self.block(x)

        return x



class NetG(nn.Module):
    def __init__(self, feat_dim, z_dim, z_scale_factors=[2,2,2]):
        super().__init__()

        # ks = 3  # filter size
        mfd = 512
        num_groups = 4
        self.num_groups = num_groups
        self.mfd = mfd

        self.feat_dim = feat_dim
        self.z_dim = z_dim
        self.z_scale_factors = z_scale_factors

        # ### Main body ###
        self.block0 = GBlock(z_dim, feat_dim, mfd, num_groups)

        blocks = []
        for scale_factor in z_scale_factors:
            block = GBlock(feat_dim, feat_dim, mfd, num_groups)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        # ### Head ###
        # self.head = nn.Conv1d(mfd, feat_dim, 3, 1, 1)

    def forward(self, z):

        # SBlock0
        z_scale_factors = self.z_scale_factors
        # nf = min(z.size(2), cond_.size(2))
        # zc = torch.cat([z[:, :, :nf], cond_[:, :, :nf]], dim=1)
        
        x = self.block0(z)

        # print(len(self.blocks))
        for ii, (block, scale_factor) in enumerate(zip(self.blocks, z_scale_factors)):
            x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')

            # print(total_scale_factor, x.shape, cond_.shape)
            # nf = min(x.size(2), cond_.size(2))
            # c = torch.cat([x[:, :, :nf], cond_[:, :, :nf]], dim=1)
            x = x + block(x)

        # Head
        # shape=(bs, feat_dim, nf)
        # x = torch.sigmoid(self.head(x))
        # x = torch.sigmoid(x)

        return x




class NetD(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        ks = 3  # filter size
        mfd = 512

        self.mfd = mfd
        self.input_size = input_size

        # ### Main body ###
        blocks2d = [
            BNSNConv2dDBlock(1, 4, ks, 2, 2),
            BNSNConv2dDBlock(4, 16, ks, 2, 4),
            BNSNConv2dDBlock(16, 64, ks, 2, 8),
            #BNSNConv2dDBlock(64, 128, ks, 2, 16)
        ]

        blocks1d = [
            BNSNConv1dDBlock(64*10 * input_size//80, mfd, 3, 1),
            BNSNConv1dDBlock(mfd, mfd, ks, 16),
            BNSNConv1dDBlock(mfd, mfd, ks, 32),
            BNSNConv1dDBlock(mfd, mfd, ks, 64),
            BNSNConv1dDBlock(mfd, mfd, ks, 128),
            #BNSNConv1dDBlock(mfd, mfd, ks, 256),
        ]

        self.body2d = nn.Sequential(*blocks2d)
        self.body1d = nn.Sequential(*blocks1d)

        self.head = spectral_norm(nn.Conv1d(mfd, input_size, 3, 1, 1))

    def forward(self, x):
        '''
        x.shape=(batch_size, feat_dim, num_frames)
        cond.shape=(batch_size, cond_dim, num_frames)
        '''
        bs, fd, nf = x.size()

        # ### Process generated ###
        # shape=(bs, 1, fd, nf)
        x = x.unsqueeze(1)

        # shape=(bs, 64, 10, nf_)
        x = self.body2d(x)
        # shape=(bs, 64*10, nf_)
        x = x.view(bs, -1, x.size(3))

        # ### Merging ###
        x = self.body1d(x)

        # ### Head ###
        # shape=(bs, input_size, nf)
        # out = torch.sigmoid(self.head(x))
        out = self.head(x)

        return out
class Conv2dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, frequency_stride, time_dilation):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.time_dilation = time_dilation
        self.frequency_stride = frequency_stride

        block = [
            spectral_norm(nn.Conv2d(
                input_dim, output_dim, ks,
                (frequency_stride, 1),
                (1, time_dilation*ksm1d2),
                dilation=(1, time_dilation))),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        x = self.block(x)
        
        return x
class Conv1dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride ,dilation):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dilation = dilation

        block = [
            spectral_norm(nn.Conv1d(input_dim, output_dim, ks,
                                    stride = stride, padding = dilation*ksm1d2,  dilation=dilation)),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        x = self.block(x)

        return x
class NetD_WGAN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        ks = 3  # filter size
        mfd = 512

        self.mfd = mfd
        self.input_size = input_size

        # ### Main body ###
        
        blocks2d = [
            Conv2dDBlock(1, 4, ks, 2, 1),
            Conv2dDBlock(4, 16, ks, 2, 2),
            Conv2dDBlock(16, 64, ks, 2, 4),
        ]

        blocks1d = [
            Conv1dDBlock(80, mfd, 3,1, 1),
            Conv1dDBlock(mfd, mfd, ks, 2, 4),
            Conv1dDBlock(mfd, mfd, ks, 2, 8),
            Conv1dDBlock(mfd, mfd, ks, 2, 16),
        ]

        self.body2d = nn.Sequential(*blocks2d)
        self.body1d = nn.Sequential(*blocks1d)

        self.head = spectral_norm(nn.Conv1d(mfd, 1, 3, 1, 1))
        self.head2 = nn.Linear(400//8, 1)
    def forward(self, x):
        '''
        x.shape=(batch_size, feat_dim, num_frames)
        cond.shape=(batch_size, cond_dim, num_frames)
        '''
        bs, fd, nf = x.size()

        # ### Process generated ###
        # shape=(bs, 1, fd, nf)
        #x = x.unsqueeze(1)

        # shape=(bs, 64, 10, nf_)
        #x = self.body2d(x)
        # shape=(bs, 64*10, nf_)
        #x = x.view(bs, -1, x.size(3))

        # ### Merging ###
        x = self.body1d(x)

        # ### Head ###
        # shape=(bs, input_size, nf)
        # out = torch.sigmoid(self.head(x))
        out = self.head(x)
        out = self.head2(x)
        return out

class DownSample(nn.Module):
    def __init__(self,  freq ):
        super(DownSample, self).__init__()
        mfd = 1024 
        num_groups = 4
        self.down_f0 = nn.Conv1d(freq, freq//8, 3, stride=1, padding=1)
        
        self.down_t0 = nn.Conv1d(freq, freq, 5, stride=2, padding=2)
        self.down_t1 = nn.Conv1d((freq), (freq), 5, stride=2, padding=2)
        self.down_t2 = nn.Conv1d((freq), (freq), 5, stride=2, padding=2)
        #self.down_t3 = nn.Conv1d((freq), (freq), 5, stride=2, padding=2)
        self.activ = nn.LeakyReLU()
        self.norm = nn.InstanceNorm1d((freq//8), track_running_stats=False)

    def forward(self, inp):
        C1 = self.activ( self.down_t0(inp)  )
        C2 = self.activ( self.down_t1(C1)  )
        C3 = self.activ( self.down_t2(C2)  )
        C5 = self.activ( self.down_f0(C3))
        C5 = self.norm(C5) + torch.zeros_like(C5).normal_(0, 1).float().cuda()
        return C5

class res_block(nn.Module):
    def __init__(self, in_fmap, mid_fmap=32):
        super(res_block, self).__init__()
        self.conv = nn.Conv1d(in_fmap, mid_fmap, 5, stride=1, padding=2)
        self.transp_conv = nn.ConvTranspose1d(mid_fmap, in_fmap, 5, padding=2)
        self.recurrent = nn.GRU(mid_fmap, mid_fmap, num_layers=1, batch_first=True, dropout=0.2)

    def forward(self, inp):
        mid = self.conv(inp)
        mid, hidden = self.recurrent(mid.permute(0,2,1))
        mid = mid.permute(0,2,1)
        #out = self.transp_conv(mid)
        out = inp + self.transp_conv(mid)
        return out

class NetSing2Speech(nn.Module):
    def __init__(self, freq ):
        super(NetSing2Speech, self).__init__()
        mfd = 512
        num_groups = 4
        self.down_f0 = nn.Conv1d(freq, freq//8, 3, stride=1, padding=1)
        
        self.down_t0 = nn.Conv1d(freq, freq, 5, stride=2, padding=2)
        self.down_t1 = nn.Conv1d((freq), (freq), 5, stride=2, padding=2)
        self.down_t2 = nn.Conv1d((freq), (freq), 5, stride=2, padding=2)
        self.down_t3 = nn.Conv1d((freq), (freq), 5, stride=2, padding=2)
        self.activ = nn.LeakyReLU()

        self.gb1 = GBlock(freq//8, freq, mfd, num_groups)
        
        self.gb2 = GBlock(freq, freq, mfd, num_groups)
        self.gb3 = GBlock(freq, freq, mfd, num_groups)
        self.gb4 = GBlock(freq, freq, mfd, num_groups)
        self.gb5 = GBlock(freq, freq, mfd, num_groups)
        
        self.res_block1 = res_block((freq), 128)
        self.res_block2 = res_block((freq), 128)       
        
        self.res_block3 = res_block((freq), 128)
        
        self.norm_2 = nn.InstanceNorm1d((freq), track_running_stats=False)
        
        

    def forward(self, inp):
        C1 = self.activ( self.down_t0(inp)  )
        C2 = self.activ( self.down_t1(C1)  )
        C3 = self.activ( self.down_t2(C2)  )
        C5 = self.activ( self.down_f0(C3))
        C5 = self.norm_2(C5)
        
        C6 = self.gb1(C5)
        #C6 = self.gb1_2(C6)
        #C6 = self.gb1_3(C6)
        
        C6 = F.interpolate(C6, scale_factor=2, mode='nearest')
        #print (C6.size())
        C6 = C6 + self.res_block3(C2) + self.gb2(C6)
        
        C5 = F.interpolate(C6, scale_factor=2, mode='nearest')
        C5 = C5 + self.res_block2(C1) + self.gb3(C5)

        C4 = F.interpolate(C5, scale_factor=2, mode='nearest')
        C4 = C4 + self.gb4(C4) + self.res_block1(inp)

        return C4


class NetSing2SpeechPitch(nn.Module):
    def __init__(self, freq ):
        super(NetSing2SpeechPitch, self).__init__()
        mfd = 1024 #512
        num_groups = 4*4
        pdim = 128

        self.down_f0 = nn.Conv1d(freq, freq//8, 3, stride=1, padding=1)
        
        self.down_t0 = nn.Conv1d(freq, freq//2, 5, stride=2, padding=2)
        self.down_t1 = nn.Conv1d((freq//2), (freq//4), 5, stride=2, padding=2)
        self.down_t2 = nn.Conv1d((freq//4), (freq//8), 5, stride=2, padding=2)
        self.activ = nn.LeakyReLU()

        self.gb0 = GBlock(freq//8 , freq//8, mfd, num_groups)
        self.gb1 = GBlock(freq//4 + pdim, freq//4, mfd, num_groups)
        
        self.gb2 = GBlock(freq//2 + pdim, freq//2, mfd, num_groups)
        self.gb3 = GBlock(freq + pdim, freq, mfd, num_groups)
        self.gb4 = GBlock(freq + pdim, freq, mfd, num_groups)

        self.gb1_refine = GBlock(freq//8 , freq//8, mfd, num_groups)
        
        self.gb2_refine = GBlock(freq//4 , freq//4, mfd, num_groups)
        self.gb3_refine = GBlock(freq//2 , freq//2, mfd, num_groups)
        
        self.res_block1 = res_block((freq), 128)
        self.res_block2 = res_block((freq//2), 128)      
        self.res_block3 = res_block((freq//4), 128)
        
        self.norm = nn.InstanceNorm1d((freq), track_running_stats=False)
        self.norm_2 = nn.InstanceNorm1d((freq//8), track_running_stats=False)
        
        
        self.embeds = nn.Embedding(1000, pdim)
        self.down_p_t0 = nn.Conv1d(pdim, pdim, 3, stride=1, padding=1)
        self.down_p_t1 = nn.Conv1d((pdim), (pdim), 5, stride=2, padding=2)
        self.down_p_t2 = nn.Conv1d((pdim), (pdim), 5, stride=2, padding=2)
        self.down_p_t3 = nn.Conv1d((pdim), (pdim), 5, stride=2, padding=2)

    def forward(self, inp, pitch):
        inp = self.norm(inp)
        
        C1 = self.activ( self.down_t0(inp)  )
        C2 = self.activ( self.down_t1(C1)  )
        C3 = self.activ( self.down_t2(C2)  )
        #C5 = self.activ( self.down_f0(C3))
        C5 = self.norm_2(C3)
        
        P0 = self.embeds(pitch)
        P0 = P0.transpose(1,2)
        P1 = self.activ( self.down_p_t0(P0)  )
        P2 = self.activ( self.down_p_t1(P1)  )
        P3 = self.activ( self.down_p_t2(P2)  )
        
        
        C6 = self.gb0(C5)

        C6 = F.interpolate(C6, scale_factor=2, mode='nearest')
        C6 = torch.cat([C6, self.gb1_refine(C6)], dim = 1)
        
        C6 = C6 + self.res_block3(C2) + self.gb1(torch.cat([C6, P3], dim = 1))
        
        C5 = F.interpolate(C6, scale_factor=2, mode='nearest')
        C5 = torch.cat([C5,  self.gb2_refine(C5)], dim = 1)
        C5 = C5 + self.res_block2(C1) + self.gb2(torch.cat([C5, P2], dim = 1))

        C4 = F.interpolate(C5, scale_factor=2, mode='nearest')
        C4 = torch.cat([C4, self.gb3_refine(C4)], dim = 1)
        C4 = C4 + self.res_block1(inp) + self.gb3(torch.cat([C4, P1], dim = 1))

        return C4


class NetSing2SpeechPitch_WO_SKIP(nn.Module):
    def __init__(self, freq ):
        super().__init__()
        mfd = 512
        num_groups = 4
        pdim = 128

        self.down_f0 = nn.Conv1d(freq, freq//8, 3, stride=1, padding=1)
        
        self.down_t0 = nn.Conv1d(freq, freq//2, 5, stride=2, padding=2)
        self.down_t1 = nn.Conv1d((freq//2), (freq//4), 5, stride=2, padding=2)
        self.down_t2 = nn.Conv1d((freq//4), (freq//8), 5, stride=2, padding=2)
        self.activ = nn.LeakyReLU()

        self.gb0 = GBlock(freq//8 , freq//8, mfd, num_groups)
        self.gb1 = GBlock( pdim, freq//4, mfd, num_groups)
        
        self.gb2 = GBlock(pdim, freq//2, mfd, num_groups)
        self.gb3 = GBlock(pdim, freq, mfd, num_groups)
        self.gb4 = GBlock(pdim, freq, mfd, num_groups)

        self.gb1_refine = GBlock(freq//8 , freq//8, mfd, num_groups)
        
        self.gb2_refine = GBlock(freq//4 , freq//4, mfd, num_groups)
        self.gb3_refine = GBlock(freq//2 , freq//2, mfd, num_groups)
        
        self.res_block1 = res_block((freq), 128)
        self.res_block2 = res_block((freq//2), 128)      
        self.res_block3 = res_block((freq//4), 128)
        
        self.norm = nn.InstanceNorm1d((freq), track_running_stats=False)
        self.norm_2 = nn.InstanceNorm1d((freq//8), track_running_stats=False)
        
        
        self.embeds = nn.Embedding(1000, pdim)
        self.down_p_t0 = nn.Conv1d(pdim, pdim, 3, stride=1, padding=1)
        self.down_p_t1 = nn.Conv1d((pdim), (pdim), 5, stride=2, padding=2)
        self.down_p_t2 = nn.Conv1d((pdim), (pdim), 5, stride=2, padding=2)
        self.down_p_t3 = nn.Conv1d((pdim), (pdim), 5, stride=2, padding=2)

    def forward(self, inp, pitch):
        inp = self.norm(inp)
        
        C1 = self.activ( self.down_t0(inp)  )
        C2 = self.activ( self.down_t1(C1)  )
        C3 = self.activ( self.down_t2(C2)  )
        #C5 = self.activ( self.down_f0(C3))
        C5 = self.norm_2(C3)
        
        P0 = self.embeds(pitch)
        P0 = P0.transpose(1,2)
        P1 = self.activ( self.down_p_t0(P0)  )
        P2 = self.activ( self.down_p_t1(P1)  )
        P3 = self.activ( self.down_p_t2(P2)  )
        
        
        C6 = self.gb0(C5)

        C6 = F.interpolate(C6, scale_factor=2, mode='nearest')
        C6 = torch.cat([C6, self.gb1_refine(C6)], dim = 1)
        
        C6 = C6 + self.res_block3(C2) + self.gb1(P3)
        
        C5 = F.interpolate(C6, scale_factor=2, mode='nearest')
        C5 = torch.cat([C5,  self.gb2_refine(C5)], dim = 1)
        C5 = C5 + self.res_block2(C1) + self.gb2(P2)

        C4 = F.interpolate(C5, scale_factor=2, mode='nearest')
        C4 = torch.cat([C4, self.gb3_refine(C4)], dim = 1)
        C4 = C4 + self.res_block1(inp) + self.gb3(P1)

        return C4

class NetSing2SpeechPitchExperiment(nn.Module):
    def __init__(self, freq ):
        super().__init__()
        mfd = 1024 #512
        num_groups = 4*4
        pdim = 128

        self.down_f0 = nn.Conv1d(freq, freq//8, 3, stride=1, padding=1)
        
        self.down_t0 = nn.Conv1d(freq, freq//2, 5, stride=2, padding=2)
        self.down_t1 = nn.Conv1d((freq//2), (freq//4), 5, stride=2, padding=2)
        self.down_t2 = nn.Conv1d((freq//4), (freq//8), 5, stride=2, padding=2)
        self.activ = nn.LeakyReLU()

        self.gb0 = GBlock(freq//8 , freq//8, mfd, num_groups)
        self.gb1 = GBlock(freq//4 + pdim, freq//4, mfd, num_groups)
        
        self.gb2 = GBlock(freq//2 + pdim, freq//2, mfd, num_groups)
        self.gb3 = GBlock(freq + pdim, freq, mfd, num_groups)
        self.gb4 = GBlock(freq + pdim, freq, mfd, num_groups)

        self.gb1_refine = GBlock(freq//8 , freq//8, mfd, num_groups)
        
        self.gb2_refine = GBlock(freq//4 , freq//4, mfd, num_groups)
        self.gb3_refine = GBlock(freq//2 , freq//2, mfd, num_groups)
        
        self.res_block1 = res_block((freq), 128)
        self.res_block2 = res_block((freq//2), 128)      
        self.res_block3 = res_block((freq//4), 128)
        
        self.norm = nn.InstanceNorm1d((freq), track_running_stats=False)
        self.norm_2 = nn.InstanceNorm1d((freq//8), track_running_stats=False)
        
        
        self.embeds = nn.Embedding(1000, pdim)
        self.down_p_t0 = nn.Conv1d(pdim, pdim, 3, stride=1, padding=1)
        self.down_p_t1 = nn.Conv1d((pdim), (pdim), 5, stride=2, padding=2)
        self.down_p_t2 = nn.Conv1d((pdim), (pdim), 5, stride=2, padding=2)
        self.down_p_t3 = nn.Conv1d((pdim), (pdim), 5, stride=2, padding=2)

    def forward(self, inp, pitch):
        inp = self.norm(inp)
        
        C1 = self.activ( self.down_t0(inp)  )
        C2 = self.activ( self.down_t1(C1)  )
        C3 = self.activ( self.down_t2(C2)  )
        #C5 = self.activ( self.down_f0(C3))
        C5 = self.norm_2(C3)
        
        P0 = self.embeds(pitch)
        P0 = P0.transpose(1,2)
        P1 = self.activ( self.down_p_t0(P0)  )
        P2 = self.activ( self.down_p_t1(P1)  )
        P3 = self.activ( self.down_p_t2(P2)  )
        
        
        C6 = self.gb0(C5)

        C6 = F.interpolate(C6, scale_factor=2, mode='nearest')
        
        C6 = torch.stack([C6, self.gb1_refine(C6)], dim = 1).view(C6.size(0), 2 * C6.size(1), -1)
        C6 = C6 + self.res_block3(C2) + self.gb1(torch.cat([C6, P3], dim = 1))
        
        C5 = F.interpolate(C6, scale_factor=2, mode='nearest')
        C5 = torch.stack([C5, self.gb2_refine(C5)], dim = 1).view(C5.size(0), 2 * C5.size(1), -1)

        C5 = C5 + self.res_block2(C1) + self.gb2(torch.cat([C5, P2], dim = 1))

        C4 = F.interpolate(C5, scale_factor=2, mode='nearest')
        C4 = torch.stack([C4, self.gb3_refine(C4)], dim = 1).view(C4.size(0), 2 * C4.size(1), -1)

        C4 = C4 + self.res_block1(inp) + self.gb3(torch.cat([C4, P1], dim = 1))

        return C4

class NetSing2SpeechPitchExperiment4X(nn.Module):
    def __init__(self, freq ):
        super().__init__()
        mfd = 1024 #512
        num_groups = 4*4
        pdim = 128

        
        self.down_t0 = nn.Conv1d(freq, freq//2, 6, stride=4, padding=1)
        self.down_t1 = nn.Conv1d((freq//2), (freq//4), 6, stride=4, padding=1)
        self.down_t2 = nn.Conv1d((freq//4), (freq//8), 6, stride=4, padding=1)
        self.activ = nn.LeakyReLU()

        self.gb0 = GBlock(freq//8 , freq//8, mfd, num_groups)
        self.gb1 = GBlock(freq//4 + pdim, freq//4, mfd, num_groups)
        
        self.gb2 = GBlock(freq//2 + pdim, freq//2, mfd, num_groups)
        self.gb3 = GBlock(freq + pdim, freq, mfd, num_groups)
        self.gb4 = GBlock(freq + pdim, freq, mfd, num_groups)

        self.gb1_refine = GBlock(freq//8 , freq//8, mfd, num_groups)
        
        self.gb2_refine = GBlock(freq//4 , freq//4, mfd, num_groups)
        self.gb3_refine = GBlock(freq//2 , freq//2, mfd, num_groups)
        
        self.res_block1 = res_block((freq), 128)
        self.res_block2 = res_block((freq//2), 128)      
        self.res_block3 = res_block((freq//4), 128)
        
        self.norm = nn.InstanceNorm1d((freq), track_running_stats=False)
        self.norm_2 = nn.InstanceNorm1d((freq//8), track_running_stats=False)
        
        
        self.embeds = nn.Embedding(1000, pdim)
        self.down_p_t0 = nn.Conv1d(pdim, pdim, 3, stride=1, padding=1)
        self.down_p_t1 = nn.Conv1d((pdim), (pdim), 6, stride=4, padding=1)
        self.down_p_t2 = nn.Conv1d((pdim), (pdim), 6, stride=4, padding=1)
        self.down_p_t3 = nn.Conv1d((pdim), (pdim), 6, stride=4, padding=1)

    def forward(self, inp, pitch):
        inp = self.norm(inp)
        
        C1 = self.activ( self.down_t0(inp)  )
        C2 = self.activ( self.down_t1(C1)  )
        C3 = self.activ( self.down_t2(C2)  )

        C5 = self.norm_2(C3)
        
        P0 = self.embeds(pitch)
        P0 = P0.transpose(1,2)
        P1 = self.activ( self.down_p_t0(P0)  )
        P2 = self.activ( self.down_p_t1(P1)  )
        P3 = self.activ( self.down_p_t2(P2)  )
        

        C6 = self.gb0(C5)

        C6 = F.interpolate(C6, scale_factor=4, mode='nearest')
        
        C6 = torch.stack([C6, self.gb1_refine(C6)], dim = 1).view(C6.size(0), 2 * C6.size(1), -1)
        
        C6 = C6 + self.res_block3(C2) + self.gb1(torch.cat([C6, P3], dim = 1))
        
        C5 = F.interpolate(C6, scale_factor=4, mode='nearest')
        C5 = torch.stack([C5, self.gb2_refine(C5)], dim = 1).view(C5.size(0), 2 * C5.size(1), -1)

        C5 = C5 + self.res_block2(C1) + self.gb2(torch.cat([C5, P2], dim = 1))

        C4 = F.interpolate(C5, scale_factor=4, mode='nearest')
        C4 = torch.stack([C4, self.gb3_refine(C4)], dim = 1).view(C4.size(0), 2 * C4.size(1), -1)

        C4 = C4 + self.res_block1(inp) + self.gb3(torch.cat([C4, P1], dim = 1))

        return C4

class NetSing2SpeechPitchCat4X(nn.Module):
    def __init__(self, freq ):
        super().__init__()
        mfd = 1024 #512
        num_groups = 4*4
        pdim = 128

        
        self.down_t0 = nn.Conv1d(freq, freq//2, 6, stride=4, padding=1)
        self.down_t1 = nn.Conv1d((freq//2), (freq//4), 6, stride=4, padding=1)
        self.down_t2 = nn.Conv1d((freq//4), (freq//8), 6, stride=4, padding=1)
        self.activ = nn.LeakyReLU()

        self.gb0 = GBlock(freq//8 , freq//8, mfd, num_groups)
        self.gb1 = GBlock(freq//4 + pdim, freq//4, mfd, num_groups)
        
        self.gb2 = GBlock(freq//2 + pdim, freq//2, mfd, num_groups)
        self.gb3 = GBlock(freq + pdim, freq, mfd, num_groups)
        self.gb4 = GBlock(freq + pdim, freq, mfd, num_groups)

        self.gb1_refine = GBlock(freq//8 , freq//8, mfd, num_groups)
        
        self.gb2_refine = GBlock(freq//4 , freq//4, mfd, num_groups)
        self.gb3_refine = GBlock(freq//2 , freq//2, mfd, num_groups)
        
        self.res_block1 = res_block((freq), 128)
        self.res_block2 = res_block((freq//2), 128)      
        self.res_block3 = res_block((freq//4), 128)
        
        self.norm = nn.InstanceNorm1d((freq), track_running_stats=False)
        self.norm_2 = nn.InstanceNorm1d((freq//8), track_running_stats=False)
        
        
        self.embeds = nn.Embedding(1000, pdim)
        self.down_p_t0 = nn.Conv1d(pdim, pdim, 3, stride=1, padding=1)
        self.down_p_t1 = nn.Conv1d((pdim), (pdim), 6, stride=4, padding=1)
        self.down_p_t2 = nn.Conv1d((pdim), (pdim), 6, stride=4, padding=1)
        self.down_p_t3 = nn.Conv1d((pdim), (pdim), 6, stride=4, padding=1)

    def forward(self, inp, pitch):
        inp = self.norm(inp)
        
        C1 = self.activ( self.down_t0(inp)  )
        C2 = self.activ( self.down_t1(C1)  )
        C3 = self.activ( self.down_t2(C2)  )

        C5 = self.norm_2(C3)
        
        P0 = self.embeds(pitch)
        P0 = P0.transpose(1,2)
        P1 = self.activ( self.down_p_t0(P0)  )
        P2 = self.activ( self.down_p_t1(P1)  )
        P3 = self.activ( self.down_p_t2(P2)  )
        

        C6 = self.gb0(C5)

        C6 = F.interpolate(C6, scale_factor=4, mode='nearest')
        
        C6 = torch.cat([C6, self.gb1_refine(C6)], dim = 1)
        
        C6 = C6 + self.res_block3(C2) + self.gb1(torch.cat([C6, P3], dim = 1))
        
        C5 = F.interpolate(C6, scale_factor=4, mode='nearest')
        C5 = torch.cat([C5, self.gb2_refine(C5)], dim = 1)

        C5 = C5 + self.res_block2(C1) + self.gb2(torch.cat([C5, P2], dim = 1))

        C4 = F.interpolate(C5, scale_factor=4, mode='nearest')
        C4 = torch.cat([C4, self.gb3_refine(C4)], dim = 1)

        C4 = C4 + self.res_block1(inp) + self.gb3(torch.cat([C4, P1], dim = 1))

        return C4

class NetSing2SpeechPitchUnpaired4X(nn.Module):
    def __init__(self, freq ):
        super().__init__()
        mfd = 512
        num_groups = 4*4
        pdim = 128

        
        self.down_t0 = nn.Conv1d(freq, freq//2, 6, stride=4, padding=1)
        self.down_t1 = nn.Conv1d((freq//2), (freq//4), 6, stride=4, padding=1)
        self.down_t2 = nn.Conv1d((freq//4), (freq//8), 6, stride=4, padding=1)
        self.activ = nn.LeakyReLU()

        self.gb0 = GBlock(freq//8 , freq//8, mfd, num_groups)
        self.gb1 = GBlock(freq//4 , freq//4, mfd, num_groups)
        
        self.gb2 = GBlock(freq//2 , freq//2, mfd, num_groups)
        self.gb3 = GBlock(freq , freq, mfd, num_groups)

        self.gb1_refine = GBlock(freq//8 , freq//8, mfd, num_groups)
        
        self.gb2_refine = GBlock(freq//4 , freq//4, mfd, num_groups)
        self.gb3_refine = GBlock(freq//2 , freq//2, mfd, num_groups)
        
        self.res_block1 = res_block((freq), 128)
        self.res_block2 = res_block((freq//2), 128)      
        self.res_block3 = res_block((freq//4), 128)
        
        self.norm = nn.InstanceNorm1d((freq), track_running_stats=False)
        self.norm_2 = nn.InstanceNorm1d((freq//8), track_running_stats=False)
        
    def forward(self, inp):
        inp = self.norm(inp)
        
        C1 = self.activ( self.down_t0(inp)  )
        C2 = self.activ( self.down_t1(C1)  )
        C3 = self.activ( self.down_t2(C2)  )

        C5 = self.norm_2(C3)
        
        
        
        C6 = self.gb0(C5)

        C6 = F.interpolate(C6, scale_factor=4, mode='nearest')
        
        C6 = torch.stack([C6, self.gb1_refine(C6)], dim = 1).view(C6.size(0), 2 * C6.size(1), -1)
        
        C6 = C6 + self.res_block3(C2) + self.gb1(C6)
        
        C5 = F.interpolate(C6, scale_factor=4, mode='nearest')
        C5 = torch.stack([C5, self.gb2_refine(C5)], dim = 1).view(C5.size(0), 2 * C5.size(1), -1)

        C5 = C5 + self.res_block2(C1) + self.gb2(C5)

        C4 = F.interpolate(C5, scale_factor=4, mode='nearest')
        C4 = torch.stack([C4, self.gb3_refine(C4)], dim = 1).view(C4.size(0), 2 * C4.size(1), -1)

        C4 = C4 + self.res_block1(inp) + self.gb3(C4)

        return C4

class NetSing2SpeechPitchOri(nn.Module):
    def __init__(self, freq ):
        super().__init__()
        mfd = 512
        num_groups = 4
        pdim = 128

        self.down_f0 = nn.Conv1d(freq, freq//8, 3, stride=1, padding=1)
        
        self.down_t0 = nn.Conv1d(freq, freq, 5, stride=2, padding=2)
        self.down_t1 = nn.Conv1d((freq), (freq), 5, stride=2, padding=2)
        self.down_t2 = nn.Conv1d((freq), (freq), 5, stride=2, padding=2)
        self.activ = nn.LeakyReLU()

        self.gb0 = GBlock(freq//8 , freq, mfd, num_groups)
        self.gb1 = GBlock(freq + pdim, freq, mfd, num_groups)
        
        self.gb2 = GBlock(freq + pdim, freq, mfd, num_groups)
        self.gb3 = GBlock(freq + pdim, freq, mfd, num_groups)
        self.gb4 = GBlock(freq + pdim, freq, mfd, num_groups)

        
        self.res_block1 = GBlock(freq, freq, mfd, num_groups)#res_block((freq), 128)
        self.res_block2 = GBlock(freq, freq, mfd, num_groups)
        self.res_block3 = GBlock(freq, freq, mfd, num_groups)
        
        self.norm = nn.InstanceNorm1d((freq), track_running_stats=False)
        self.norm_2 = nn.InstanceNorm1d((freq//8), track_running_stats=False)
        
        
        self.embeds = nn.Embedding(1000, pdim)
        self.down_p_t0 = nn.Conv1d(pdim, pdim, 3, stride=1, padding=1)
        self.down_p_t1 = nn.Conv1d((pdim), (pdim), 5, stride=2, padding=2)
        self.down_p_t2 = nn.Conv1d((pdim), (pdim), 5, stride=2, padding=2)
        self.down_p_t3 = nn.Conv1d((pdim), (pdim), 5, stride=2, padding=2)

    def forward(self, inp, pitch):
        inp = self.norm(inp)
        C1 = self.activ( self.down_t0(inp)  )
        C2 = self.activ( self.down_t1(C1)  )
        C3 = self.activ( self.down_t2(C2)  )
        C5 = self.activ( self.down_f0(C3))
        C5 = self.norm_2(C5)
        
        P0 = self.embeds(pitch)
        P0 = P0.transpose(1,2)
        P1 = self.activ( self.down_p_t0(P0)  )
        P2 = self.activ( self.down_p_t1(P1)  )
        P3 = self.activ( self.down_p_t2(P2)  )
        
        
        C6 = self.gb0(C5)

        C6 = F.interpolate(C6, scale_factor=2, mode='nearest')
        
        C6 = C6 + self.res_block3(C2) + self.gb1(torch.cat([C6, P3], dim = 1))
        
        C5 = F.interpolate(C6, scale_factor=2, mode='nearest')
        
        C5 = C5 + self.res_block2(C1) + self.gb2(torch.cat([C5, P2], dim = 1))

        C4 = F.interpolate(C5, scale_factor=2, mode='nearest')
        C4 = C4 + self.res_block1(inp) + self.gb3(torch.cat([C4, P1], dim = 1))

        return C4


class NetGExperiment(nn.Module):
    def __init__(self, feat_dim, z_dim, z_scale_factors=[4,4,4]):
        super().__init__()

        # ks = 3  # filter size
        mfd = 512
        num_groups = 4
        self.num_groups = num_groups
        self.mfd = mfd

        self.feat_dim = feat_dim
        self.z_dim = z_dim
        self.z_scale_factors = z_scale_factors

        self.block0 = GBlock(z_dim, z_dim, mfd, num_groups)
        # ### Main body ###

        blocks = []
        for i in range(len(z_scale_factors)):
            block = GBlock(z_dim * (2**(i )), z_dim*(2**(i)), mfd, num_groups)
            blocks.append(block)
        blocks_refine = []

        for i in range(len(z_scale_factors)):
            block = GBlock(z_dim * (2**i), z_dim* (2**i), mfd, num_groups)
            blocks_refine.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.blocks_refine = nn.ModuleList(blocks_refine)

        
        # ### Head ###
        # self.head = nn.Conv1d(mfd, feat_dim, 3, 1, 1)

    def forward(self, z):

        # SBlock0
        z_scale_factors = self.z_scale_factors
        # nf = min(z.size(2), cond_.size(2))
        # zc = torch.cat([z[:, :, :nf], cond_[:, :, :nf]], dim=1)

        x = self.block0(z)

        # print(len(self.blocks))
        for ii, (block, block_refine, scale_factor) in enumerate(zip(self.blocks, self.blocks_refine, z_scale_factors)):
            x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
            # print(total_scale_factor, x.shape, cond_.shape)
            # nf = min(x.size(2), cond_.size(2))
            # c = torch.cat([x[:, :, :nf], cond_[:, :, :nf]], dim=1)
            x = x + block(x)
            x = torch.cat([x, block_refine(x)], dim = 1) #--> it is good
            #x = torch.stack([x, block_refine(x)], dim = 1).view(x.size(0), 2 * x.size(1), -1)
        
        #/nas189/homes/ericwudayi/DAMP/10seconds_mel
        # Head
        # shape=(bs, feat_dim, nf)
        # x = torch.sigmoid(self.head(x))
        # x = torch.sigmoid(x)

        return x

class NetGExperimentStack(nn.Module):
    def __init__(self, feat_dim, z_dim, z_scale_factors=[4,4,4]):
        super().__init__()

        # ks = 3  # filter size
        mfd = 512
        num_groups = 4
        self.num_groups = num_groups
        self.mfd = mfd

        self.feat_dim = feat_dim
        self.z_dim = z_dim
        self.z_scale_factors = z_scale_factors

        self.block0 = GBlock(z_dim, z_dim, mfd, num_groups)
        # ### Main body ###

        blocks = []
        for i in range(len(z_scale_factors)):
            block = GBlock(z_dim * (2**(i )), z_dim*(2**(i)), mfd, num_groups)
            blocks.append(block)

        blocks_refine = []

        for i in range(len(z_scale_factors)):
            block = GBlock(z_dim * (2**i), z_dim* (2**i), mfd, num_groups)
            blocks_refine.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.blocks_refine = nn.ModuleList(blocks_refine)

        
        # ### Head ###
        # self.head = nn.Conv1d(mfd, feat_dim, 3, 1, 1)

    def forward(self, z):

        # SBlock0
        z_scale_factors = self.z_scale_factors
        # nf = min(z.size(2), cond_.size(2))
        # zc = torch.cat([z[:, :, :nf], cond_[:, :, :nf]], dim=1)

        x = self.block0(z)

        # print(len(self.blocks))
        for ii, (block, block_refine, scale_factor) in enumerate(zip(self.blocks, self.blocks_refine, z_scale_factors)):
            x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
            # print(total_scale_factor, x.shape, cond_.shape)
            # nf = min(x.size(2), cond_.size(2))
            # c = torch.cat([x[:, :, :nf], cond_[:, :, :nf]], dim=1)
            x = x + block(x)
            #x = torch.cat([x, block_refine(x)], dim = 1) #--> it is good
            x = torch.stack([x, block_refine(x)], dim = 1).view(x.size(0), 2 * x.size(1), -1)
        
        #/nas189/homes/ericwudayi/DAMP/10seconds_mel
        # Head
        # shape=(bs, feat_dim, nf)
        # x = torch.sigmoid(self.head(x))
        # x = torch.sigmoid(x)

        return x


class NetGExperimentStackWithHead(nn.Module):
    def __init__(self, feat_dim, z_dim, z_scale_factors=[2, 2, 16]):
        super().__init__()

        # ks = 3  # filter size
        mfd = 512
        num_groups = 4
        self.num_groups = num_groups
        self.mfd = mfd

        self.feat_dim = feat_dim
        self.z_dim = z_dim
        self.z_scale_factors = z_scale_factors

        block_head = []
        for i in range(5):
            block_head += [GBlock(z_dim, z_dim, mfd, num_groups)]
        
        # ### Main body ###

        blocks = []
        for i in range(len(z_scale_factors)):
            block = GBlock(z_dim * (2**(i)), z_dim*(2**(i)), mfd, num_groups)
            blocks.append(block)

        blocks_refine = []

        for i in range(len(z_scale_factors)):
            block = GBlock(z_dim * (2**i), z_dim* (2**i), mfd, num_groups)
            blocks_refine.append(block)

        blocks_refine2 = []
        for i in range(len(z_scale_factors)):
            block = GBlock(z_dim * (2**(i+1)), z_dim* (2**(i+1)), mfd, num_groups)
            blocks_refine2.append(block)

        self.block_head = nn.ModuleList(block_head)
        self.blocks = nn.ModuleList(blocks)
        self.blocks_refine = nn.ModuleList(blocks_refine)
        self.blocks_refine2 = nn.ModuleList(blocks_refine2)
        
        # ### Head ###
        self.head = spectral_norm(nn.Conv1d(z_dim * 2**3, feat_dim, 3, 1, 1))

    def forward(self, x):

        # SBlock0
        z_scale_factors = self.z_scale_factors
        # nf = min(z.size(2), cond_.size(2))
        # zc = torch.cat([z[:, :, :nf], cond_[:, :, :nf]], dim=1)
        for block in self.block_head:
            x = block(x)

        # print(len(self.blocks))
        for ii, (block, block_refine, block_refine2, scale_factor) in enumerate(zip(self.blocks, self.blocks_refine, self.blocks_refine2, z_scale_factors)):
            x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
            # print(total_scale_factor, x.shape, cond_.shape)
            # nf = min(x.size(2), cond_.size(2))
            # c = torch.cat([x[:, :, :nf], cond_[:, :, :nf]], dim=1)
            x = x + block(x)
            x = torch.cat([x, block_refine(x)], dim = 1) #--> it is good
            x = x + block_refine2(x)
            #x = torch.stack([x, block_refine(x)], dim = 1).view(x.size(0), 2 * x.size(1), -1)
        
        # Head
        # shape=(bs, feat_dim, nf)
        x = (self.head(x))
        # x = torch.sigmoid(x)

        return x


class NetGExperimentAccom(nn.Module):
    def __init__(self, feat_dim, z_dim, z_scale_factors=[2,2,2]):
        super().__init__()

        # ks = 3  # filter size
        mfd = 512
        num_groups = 16
        self.num_groups = num_groups
        self.mfd = mfd

        self.feat_dim = feat_dim
        self.z_dim = z_dim
        self.z_scale_factors = z_scale_factors

        self.block0 = GBlock(z_dim, z_dim, mfd, num_groups)
        # ### Main body ###

        blocks = []
        for i in range(len(z_scale_factors)):
            block = GBlock(z_dim * (2**(i + 1)), z_dim*(2**(i+1)), mfd, num_groups)
            blocks.append(block)
        blocks_refine = []

        for i in range(len(z_scale_factors)):
            block = GBlock(z_dim * (2**i), z_dim* 2**i, mfd, num_groups)
            blocks_refine.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.blocks_refine = nn.ModuleList(blocks_refine)

        
        # ### Head ###
        # self.head = nn.Conv1d(mfd, feat_dim, 3, 1, 1)

    def forward(self, z):

        # SBlock0
        z_scale_factors = self.z_scale_factors
        # nf = min(z.size(2), cond_.size(2))
        # zc = torch.cat([z[:, :, :nf], cond_[:, :, :nf]], dim=1)

        x = self.block0(z)

        # print(len(self.blocks))
        for ii, (block, block_refine, scale_factor) in enumerate(zip(self.blocks, self.blocks_refine, z_scale_factors)):
            #x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
            x = torch.cat([x, x], dim = 2)
            x = torch.cat([x, block_refine(x)], dim = 1)
            # print(total_scale_factor, x.shape, cond_.shape)
            # nf = min(x.size(2), cond_.size(2))
            # c = torch.cat([x[:, :, :nf], cond_[:, :, :nf]], dim=1)
            x = x + block(x)
        
        #x = x + self.block2D(x)
        # Head
        # shape=(bs, feat_dim, nf)
        # x = torch.sigmoid(self.head(x))
        # x = torch.sigmoid(x)

        return x





class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.embedding = nn.Embedding(n_embed,dim)
        self.inorm = nn.InstanceNorm1d(dim)
    def forward(self, input):
        embed = (self.embedding.weight.detach()).transpose(0,1)
        
        embed = (embed)/(torch.norm(embed,dim=0))
        #input = input / torch.norm(input, dim = 2, keepdim=True)
        flatten = input.reshape(-1, self.dim).detach()
        
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*input.shape[:-1]).detach().cpu().cuda()
        quantize = self.embedding(embed_ind)
        diff = (quantize - input).pow(2).mean()
        quantize_1 = input + (quantize - input).detach()
        
        return (quantize+quantize_1)/2, diff







class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()
        '''
        blocks = [nn.Conv1d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))
        '''
        blocks = []
        blocks_refine = []
        resblock = []
        num_groups = 4
        #print ("decoder in: ", in_channel)
        self.block0 = GBlock(in_channel//8, in_channel//8, channel, num_groups)
        if stride == 8:
            for i in range(1,4,1):
                block = GBlock(in_channel//2**(i), in_channel//2**(i), channel, num_groups)
                blocks.append(block)
            for i in range(1,4,1):
                block = GBlock(in_channel//2**(i), in_channel//2**(i), channel, num_groups)
                blocks_refine.append(block)
            for i in range(1,4,1):
                block = GBlock(in_channel//2**(i), in_channel//2**(i), channel, num_groups)
                resblock += [block]
        
        self.blocks = nn.ModuleList(blocks[::-1])
        self.blocks_refine = nn.ModuleList(blocks_refine[::-1])
        self.resblock = nn.ModuleList(resblock[::-1])

        self.z_scale_factors = [2,2,2]

    def forward(self, q_after, sp_embed, std_embed):
        q_after = q_after[::-1]
        sp_embed = sp_embed[::-1]
        std_embed = std_embed[::-1]
        x = q_after[0]
        
        x = self.block0(x * std_embed[0] + sp_embed[0])
        for i, (block, block_refine, res, scale_factor) in enumerate(zip(self.blocks, self.blocks_refine, self.resblock, self.z_scale_factors)):
            q_after[i] = F.interpolate(q_after[i], scale_factor=scale_factor, mode='nearest')
            x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
            #print (std_embed[i].size(), sp_embed[i].size(), q_after[i].size(), x.size())
            x = x + res(q_after[i]*std_embed[i] + sp_embed[i])
            x = x + block(x)
            x = torch.cat([x, x + block_refine(x)], dim = 1)
            #x = torch.stack([x, block_refine(x)], dim = 1).view(x.size(0), 2 * x.size(1), -1)  
        return x
class VQAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channel=80,
        channel=512,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=4096,
        n_embed=64,
        decay=0.99,
        embed_pre = 256
    ):
        super().__init__()

        blocks = []
        for i in range(3):
            blocks += [
            nn.Sequential(*[
                nn.Conv1d(in_channel//2**(i), channel, 4, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Conv1d(channel, in_channel//2**(i+1), 3, 1, 1),
                
            ])]
        self.enc = nn.ModuleList(blocks)
        
        quantize_blocks = []
        
        for i in range(3):
            quantize_blocks += [
            Quantize(in_channel//2**(i+1), n_embed)]
        self.quantize = nn.ModuleList(quantize_blocks)
        
        
        self.dec = Decoder(
            in_channel ,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=8,
        )
    def forward(self, input):
        enc_b, sp_embed, std_block, diff = self.encode(input)
        #print (enc_speaker.size(), enc_b.size(), enc_b_content.size())
        dec_1= self.decode(enc_b, sp_embed, std_block)
        idx = torch.randperm(enc_b[0].size(0))
        sp_shuffle = []
        std_shuffle = []
        for sm in (sp_embed):
            sp_shuffle += [sm[idx]]
        for std in std_block:
            std_shuffle += [std[idx]]
        
        dec_2 = self.decode(enc_b, sp_shuffle, std_shuffle)
        return dec_1, diff

    def encode(self, input):
        x = input
        sp_embedding_block = []
        q_after_block = []
        std_block = []
        diff_total = 0


        for i, (enc_block, quant) in enumerate(zip(self.enc, self.quantize)):
            x = enc_block(x)   
            
            x_ = x - torch.mean(x, dim = 2, keepdim = True)
            std_ = torch.norm(x_, dim= 2, keepdim = True) + 1e-4
            std_block += [std_]
            x_ = x_ / std_
            
            x_ = x_ / torch.norm(x_, dim = 1, keepdim = True)
            q_after, diff = quant(x_.permute(0,2,1))
            q_after = q_after.permute(0,2,1)
            
            sp_embed = torch.mean(x - q_after, 2, True)
            sp_embed = sp_embed / (torch.norm(sp_embed, dim = 1, keepdim=True)+1e-4) / 3

            sp_embedding_block += [sp_embed]
            q_after_block += [q_after]
            diff_total += diff
        return q_after_block, sp_embedding_block, std_block, diff_total 

    def decode(self, quant_b, sp, std):
        
        dec_1 = self.dec(quant_b, sp, std)
        
        return dec_1
    
class AutoEncoder(nn.Module):
    def __init__(self, freq):
        super().__init__()
        mfd = 512 
        num_groups = 4*4

        
        self.down_t0 = nn.Conv1d(freq, freq//2, 5, stride=2, padding=2)
        self.down_t1 = nn.Conv1d((freq//2), (freq//4), 5, stride=2, padding=2)
        self.down_t2 = nn.Conv1d((freq//4), (freq//8), 5, stride=2, padding=2)
        self.activ = nn.LeakyReLU()

        self.gb0 = GBlock(freq//8 , freq//8, mfd, num_groups)
        self.gb1 = GBlock(freq//4 , freq//4, mfd, num_groups)
        
        self.gb2 = GBlock(freq//2 , freq//2, mfd, num_groups)
        self.gb3 = GBlock(freq, freq, mfd, num_groups)
        self.gb4 = GBlock(freq, freq, mfd, num_groups)

        self.gb1_refine = GBlock(freq//8 , freq//8, mfd, num_groups)
        
        self.gb2_refine = GBlock(freq//4 , freq//4, mfd, num_groups)
        self.gb3_refine = GBlock(freq//2 , freq//2, mfd, num_groups)
        
        self.res_block1 = res_block((freq), 128)
        self.res_block2 = res_block((freq//2), 128)      
        self.res_block3 = res_block((freq//4), 128)
        
        self.norm = nn.InstanceNorm1d((freq), track_running_stats=False)
        self.norm_2 = nn.InstanceNorm1d((freq//8), track_running_stats=False)
        

    def forward(self, inp):
        inp = self.norm(inp)
        
        C1 = self.activ( self.down_t0(inp)  )
        C2 = self.activ( self.down_t1(C1)  )
        C3 = self.activ( self.down_t2(C2)  )

        C5 = self.norm_2(C3)
        
        C6 = self.gb0(C5)

        C6 = F.interpolate(C6, scale_factor=2, mode='nearest')
        
        C6 = torch.cat([C6, self.gb1_refine(C6)], dim = 1)
        
        C6 = C6 + self.res_block3(C2) + self.gb1(C6)
        
        C5 = F.interpolate(C6, scale_factor=2, mode='nearest')
        C5 = torch.cat([C5, self.gb2_refine(C5)], dim = 1)

        C5 = C5 + self.res_block2(C1) + self.gb2(C5)

        C4 = F.interpolate(C5, scale_factor=2, mode='nearest')
        C4 = torch.cat([C4, self.gb3_refine(C4)], dim = 1)

        C4 = C4 + self.res_block1(inp) + self.gb3(C4)

        return C4


class net_in_v2(nn.Module):
    def __init__(self, input_size=80, output_size=80, freq=80):
        super(net_in_v2, self).__init__()
        self.down_f0 = nn.Conv1d(freq, freq, 3, stride=1, padding=1)
        self.down_f1 = nn.Conv1d(freq, (freq)//2, 5, stride=1, padding=2)
        self.down_f2 = nn.Conv1d((freq)//2, (freq)//4, 5, stride=1, padding=2)
        self.down_f3 = nn.Conv1d((freq)//4, (freq)//8, 5, stride=1, padding=2)
        self.down_t0 = nn.Conv1d(freq, freq, 5, stride=2, padding=2)
        self.down_t1 = nn.Conv1d((freq)//2, (freq)//2, 5, stride=2, padding=2)
        self.down_t2 = nn.Conv1d((freq)//4, (freq)//4, 5, stride=2, padding=2)

        pdim = freq
        self.down_f0p = nn.Conv1d(pdim,pdim , 3, stride=1, padding=1)
        self.down_f1p = nn.Conv1d(pdim, (pdim)//2, 5, stride=1, padding=2)
        self.down_f2p = nn.Conv1d((pdim)//2, (pdim)//4, 5, stride=1, padding=2)
        self.down_f3p = nn.Conv1d((pdim)//4, (pdim)//8, 5, stride=1, padding=2)
        self.down_t0p = nn.Conv1d(pdim, pdim, 5, stride=2, padding=2)
        self.down_t1p = nn.Conv1d((pdim)//2, (pdim)//2, 5, stride=2, padding=2)
        self.down_t2p = nn.Conv1d((pdim)//2, (pdim)//4, 5, stride=2, padding=2)

        self.up_f1 = nn.ConvTranspose1d((freq)//4, (freq)//4, 5, padding=2)
        self.up_f2 = nn.ConvTranspose1d((freq)//4, (freq)//2, 5, padding=2)
        self.up_f3 = nn.ConvTranspose1d((freq)//2, freq, 5, padding=2)
        self.up_f4 = nn.ConvTranspose1d(freq, freq, 5, padding=2)
        self.up_t1 = nn.ConvTranspose1d((freq)//4, (freq)//4, 3, stride=2, padding=1, output_padding=1)
        self.up_t2 = nn.ConvTranspose1d((freq)//2, (freq)//2, 3, stride=2, padding=1, output_padding=1)
        self.up_t3 = nn.ConvTranspose1d(freq, freq, 3, stride=2, padding=1, output_padding=1)

        self.activ = nn.LeakyReLU()
        self.embeds = nn.Embedding(1000, pdim)
        self.inp_size = input_size
        self.out_size = output_size
        self.lstm_1 = nn.GRU((freq)//4, (freq)//4, num_layers=1, batch_first=True, dropout=0.3)
        self.lstm_2 = nn.GRU(freq, freq, num_layers=1, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.05)
        self.freq = freq

        self.norm_f = nn.InstanceNorm1d(freq, track_running_stats=False)
        self.norm_0 = nn.InstanceNorm1d(freq, track_running_stats=False)
        self.norm_1 = nn.InstanceNorm1d((freq)//2, track_running_stats=False)
        self.norm_2 = nn.InstanceNorm1d((freq)//4, track_running_stats=False)
        self.norm_3 = nn.InstanceNorm1d((freq)//8, track_running_stats=False)

        self.res_block1 = res_block((freq)//1, 32)
        self.res_block2 = res_block((freq)//2, 32)
        self.res_block22 = res_block((freq)//2, 32)
        self.res_block3 = res_block((freq)//4, 32)
        self.res_block32 = res_block((freq)//4, 32)



    def forward(self, inp, pitch_code):
        # Encode
        #print (inp.size(), pitch_code.size())
        #pitch_code = pitch_code.float()
        C1 = self.activ( self.down_f0(inp)  )
        C1 = self.activ( self.down_t0(C1)  )
        C2 = self.activ( self.down_f1(C1)  )
        C3 = self.activ( self.down_t1(C2)  )
        C4 = self.norm_2( self.down_f2(C3) )

        C4, hidden = self.lstm_1(C4.permute(0,2,1))
        C4 = C4.permute(0,2,1)
        C5 = self.activ(  self.down_t2(C4)  )
        encoding1 = self.activ(  self.down_f3(C5)  )
        
        pitch_code = self.embeds(pitch_code).transpose(1,2)
        
        pitch_code = self.activ( self.down_f0p(pitch_code)  )
        pitch_code = self.activ( self.down_t0p(pitch_code)  )
        pitch_code = self.activ( self.down_f1p(pitch_code)  )
        pitch_code = self.activ( self.down_t1p(pitch_code)  )
        pitch_code = self.activ(  self.down_t2p(pitch_code)  )
        pitch_code = self.activ(  self.down_f3p(pitch_code)  )
        
        encoding = torch.cat((encoding1, pitch_code), 1)

        # Decode
        C6 = self.activ( self.up_f1(encoding)   )
        C5 = self.activ( self.up_t1(C6 + self.res_block3(C5))    )
        C4 = self.activ( self.up_f2(C5)   )
        C3 = self.activ( self.up_t2(C4 + self.res_block2(C3))  )
        C2 = self.activ( self.norm_0( self.up_f3(C3)  ) )
        output = self.norm_0( self.up_t3(C2 + self.res_block1(C1)) )

        #output = self.activ(  self.up_f4(output)   )

        output, hidden = self.lstm_2(output.permute(0,2,1))

        output = self.activ(  self.up_f4(output.permute(0,2,1))   )
        
        return output
        #return output, C2


