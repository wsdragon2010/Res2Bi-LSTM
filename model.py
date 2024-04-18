'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

'''

import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
import pooling_layers as pooling_layers


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 

class Bottle2neck_cnn_lstm(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck_cnn_lstm, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        lstms       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            lstms.append(nn.LSTM(width, int(width/2), num_layers=1, bias=True, batch_first=True, bidirectional=True))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.lstms  = nn.ModuleList(lstms)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]

        #   sp = sp.transpose(1,2)
        #   print(f'sp:{sp}')
          # CNN层(3*1)
          sp = self.convs[i](sp)
          # LSTM层
        #   print(f'sp_cnn:{sp}')
          sp = sp.transpose(1,2)
        #   print(f'sp_cnn_size:{sp.shape}')
          sp,_ = self.lstms[i](sp)
        #   print(f'sp_lstm:{len(sp)}')
        #   print(f'sp_lstm_size:{sp[0].shape}')
        #   print(f'sp_lstm:{sp}')
          sp = sp.transpose(1,2)

          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out

class Bottle2neck_lstm_cnn(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck_lstm_cnn, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        lstms       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            lstms.append(nn.LSTM(width, int(width/2), num_layers=1, bias=True, batch_first=True, bidirectional=True))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.lstms  = nn.ModuleList(lstms)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]

          # LSTM层
          sp = sp.transpose(1,2)
          sp,_ = self.lstms[i](sp)
          sp = sp.transpose(1,2)

          # CNN层(3*1)
          sp = self.convs[i](sp)


          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out
        
class Bi_Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bi_Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        convs_r     = []
        bns_r       = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
            convs_r.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns_r.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.convs_r= nn.ModuleList(convs_r)
        self.bns_r  = nn.ModuleList(bns_r)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        spx_r = spx
        for i in range(self.nums,0,-1):
          if i==self.nums:
            sp_r = spx_r[i]
          else:
            sp_r = sp_r + spx_r[i]
          sp_r = self.convs_r[i-1](sp_r)
          sp_r = self.relu(sp_r)
          sp_r = self.bns_r[i-1](sp_r)
          if i==self.nums:
            out_r = sp_r
          else:
            out_r = torch.cat((out_r, sp_r), 1)
        out_r = torch.cat((out_r, spx_r[0]),1)
        # out_r = torch.flip(out, dims=[1])
        # spx_r = torch.split(out_r, self.width, 1)
        # for i in range(self.nums):
        #   if i==0:
        #     sp_r = spx_r[i]
        #   else:
        #     sp_r = sp_r + spx_r[i]
        #   sp_r = self.convs_r[i](sp_r)
        #   sp_r = self.relu(sp_r)
        #   sp_r = self.bns_r[i](sp_r)
        #   if i==0:
        #     out_r = sp_r
        #   else:
        #     out_r = torch.cat((out_r, sp_r), 1)
        # out_r = torch.cat((out_r, spx_r[self.nums]),1)

        out = out + out_r

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 
    
class Bottle2neck_se(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck_se, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        ses         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            ses.append(SEModule(width))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.ses  = nn.ModuleList(ses)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.ses[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out
    
class Bottle2neck_lstm(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck_lstm, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        # print(int(width/2))
        for i in range(self.nums):
            # convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            convs.append(nn.LSTM(width, int(width/2), num_layers=1, bias=True, batch_first=True, bidirectional=True))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        # out = out.transpose(1,2)
        spx = torch.split(out, self.width, 1)
        # print(out.shape)
        # spx = torch.split(out, self.width, dim=2)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = sp.transpose(1,2)
        #   print(sp.shape)
          sp,_ = self.convs[i](sp)
          sp = sp.transpose(1,2)
        #   print(sp.shape)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
            # out = torch.cat((out, sp), 2)
        out = torch.cat((out, spx[self.nums]),1)
        # out = torch.cat((out, spx[self.nums]),2)
        # out = out.transpose(1,2)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 

class Bottle2neck_GRU(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck_GRU, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        # print(int(width/2))
        for i in range(self.nums):
            # convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            # convs.append(nn.GRU(width, width, num_layers=1, bias=True, batch_first=True))
            convs.append(nn.GRU(width, int(width/2), num_layers=1, bias=True, batch_first=True, bidirectional=True))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        # out = out.transpose(1,2)
        spx = torch.split(out, self.width, 1)
        # print(out.shape)
        # spx = torch.split(out, self.width, dim=2)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = sp.transpose(1,2)
        #   print(sp.shape)
          sp,_ = self.convs[i](sp)
          sp = sp.transpose(1,2)
        #   print(sp.shape)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
            # out = torch.cat((out, sp), 2)
        out = torch.cat((out, spx[self.nums]),1)
        # out = torch.cat((out, spx[self.nums]),2)
        # out = out.transpose(1,2)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 
        
class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class ECAPA_TDNN(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            # torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=80, melkwargs={"n_fft": 512, "win_length": 400, \
            #                                     "hop_length": 160, "n_mels": 160, "f_min": 20, "f_max": 7600},log_mels=True),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

class ECAPA_TDNN_CNN_LSTM(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN_CNN_LSTM, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            # torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=80, melkwargs={"n_fft": 512, "win_length": 400, \
            #                                     "hop_length": 160, "n_mels": 160, "f_min": 20, "f_max": 7600},log_mels=True),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck_cnn_lstm(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck_cnn_lstm(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck_cnn_lstm(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

class ECAPA_TDNN_LSTM_CNN(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN_LSTM_CNN, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            # torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=80, melkwargs={"n_fft": 512, "win_length": 400, \
            #                                     "hop_length": 160, "n_mels": 160, "f_min": 20, "f_max": 7600},log_mels=True),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck_lstm_cnn(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck_lstm_cnn(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck_lstm_cnn(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x
      
class ECAPA_TDNN_BiBottle(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN_BiBottle, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            # torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=80, melkwargs={"n_fft": 512, "win_length": 400, \
            #                                     "hop_length": 160, "n_mels": 160, "f_min": 20, "f_max": 7600},log_mels=True),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bi_Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bi_Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bi_Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x
    
class ECAPA_TDNN_se(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN_se, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck_se(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck_se(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck_se(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x
    
class ECAPA_TDNN_lstm(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN_lstm, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            # torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=80, melkwargs={"n_fft": 512, "win_length": 400, \
            #                                     "hop_length": 160, "n_mels": 160, "f_min": 20, "f_max": 7600},log_mels=True),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck_lstm(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck_lstm(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck_lstm(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

class ECAPA_TDNN_GRU(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN_GRU, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck_GRU(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck_GRU(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck_GRU(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x
         
class Bi_ECAPA_TDNN_Add(nn.Module):
    # 双向res2net结果，结果相加
    def __init__(self, C):

        super(Bi_ECAPA_TDNN_Add, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer1_r = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer2_r = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer3_r = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x_r = torch.flip(x, dims=[1])

        x1_p = self.layer1(x)
        x1_r = self.layer1_r(x_r)
        # x1_r = torch.flip(x1_r, dims=[1])
        x1 = x1_p + x1_r

        x2_p = self.layer2(x+x1)
        x2_r = self.layer2_r(torch.flip(x+x1, dims=[1]))
        # x2_r = torch.flip(x2_r, dims=[1])
        x2 = x2_p + x2_r

        x3_p = self.layer3(x+x1+x2)
        x3_r = self.layer3_r(torch.flip(x+x1+x2, dims=[1]))
        # x3_r = torch.flip(x3_r, dims=[1])
        x3 = x3_p + x3_r

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

class Bi_ECAPA_TDNN_lstm_Add(nn.Module):

    def __init__(self, C):

        super(Bi_ECAPA_TDNN_lstm_Add, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck_lstm(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer1_r = Bottle2neck_lstm(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck_lstm(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer2_r = Bottle2neck_lstm(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck_lstm(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer3_r = Bottle2neck_lstm(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x_r = torch.flip(x, dims=[1])

        x1_p = self.layer1(x)
        x1_r = self.layer1_r(x_r)
        # x1_r = torch.flip(x1_r, dims=[1])
        x1 = x1_p + x1_r

        x2_p = self.layer2(x+x1)
        x2_r = self.layer2_r(torch.flip(x+x1, dims=[1]))
        # x2_r = torch.flip(x2_r, dims=[1])
        x2 = x2_p + x2_r

        x3_p = self.layer3(x+x1+x2)
        x3_r = self.layer3_r(torch.flip(x+x1+x2, dims=[1]))
        # x3_r = torch.flip(x3_r, dims=[1])
        x3 = x3_p + x3_r
        # x1 = self.layer1(x)
        # x2 = self.layer2(x+x1)
        # x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution without padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1_1d(in_planes, out_planes, stride=1):
    "1x1 convolution without padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

def conv3x3_1d(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class AFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels * 2, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x, ds_y):
        xa = torch.cat((x, ds_y), dim=1)
        x_att = self.local_att(xa)
        x_att = 1.0 + torch.tanh(x_att)
        xo = torch.mul(x, x_att) + torch.mul(ds_y, 2.0-x_att)

        return xo
    
class BasicBlockERes2Net(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, baseWidth=32, scale=2):
        super(BasicBlockERes2Net, self).__init__()
        width = int(math.floor(planes*(baseWidth/64.0)))
        self.conv1 = conv1x1(in_planes, width*scale, stride)
        self.bn1 = nn.BatchNorm2d(width*scale)
        self.nums = scale

        convs=[]
        bns=[]
        for i in range(self.nums):
          convs.append(conv3x3(width,width))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)
        
        self.conv3 = conv1x1(width*scale,planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out,self.width,1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out,sp),1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out

class BasicBlockERes2Net_diff_AFF(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, baseWidth=32, scale=2):
        super(BasicBlockERes2Net_diff_AFF, self).__init__()
        width = int(math.floor(planes*(baseWidth/64.0)))
        self.conv1 = conv1x1(in_planes, width*scale, stride)
        self.bn1 = nn.BatchNorm2d(width*scale)
        self.nums = scale

        convs=[]
        fuse_models=[]
        bns=[]
        for i in range(self.nums):
          convs.append(conv3x3(width,width))
          bns.append(nn.BatchNorm2d(width))
        for j in range(self.nums - 1):
            fuse_models.append(AFF(channels=width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)
        
        self.conv3 = conv1x1(width*scale,planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out,self.width,1)     
        for i in range(self.nums):
            if i==0:
                sp = spx[i]
            else:
                sp = self.fuse_models[i-1](sp, spx[i])
                
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i==0:
                out = sp
            else:
                out = torch.cat((out,sp),1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out

class ERes2Net(nn.Module):
    def __init__(self,
                 block=BasicBlockERes2Net,
                 block_fuse=BasicBlockERes2Net_diff_AFF,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=32,
                 feat_dim=80,
                 embedding_size=192,
                 pooling_func='TSTP',
                 two_emb_layer=False):
        super(ERes2Net, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embedding_size = embedding_size
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block_fuse,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=2)
        self.layer4 = self._make_layer(block_fuse,
                                       m_channels * 8,
                                       num_blocks[3],
                                       stride=2)

        # Downsampling module for each layer
        self.layer1_downsample = nn.Conv2d(m_channels * 2, m_channels * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_downsample = nn.Conv2d(m_channels * 4, m_channels * 8, kernel_size=3, padding=1, stride=2, bias=False)
        self.layer3_downsample = nn.Conv2d(m_channels * 8, m_channels * 16, kernel_size=3, padding=1, stride=2, bias=False)

        # Bottom-up fusion module
        self.fuse_mode12 = AFF(channels=m_channels * 4)
        self.fuse_mode123 = AFF(channels=m_channels * 8)
        self.fuse_mode1234 = AFF(channels=m_channels * 16)

        self.n_stats = 1 if pooling_func == 'TAP' or pooling_func == "TSDP" else 2
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * block.expansion)
        self.seg_1 = nn.Linear(self.stats_dim * block.expansion * self.n_stats,
                               embedding_size)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embedding_size, affine=False)
            self.seg_2 = nn.Linear(embedding_size, embedding_size)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = x.unsqueeze_(1)
        # print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out1_downsample = self.layer1_downsample(out1)
        fuse_out12 = self.fuse_mode12(out2, out1_downsample)   
        out3 = self.layer3(out2)
        fuse_out12_downsample = self.layer2_downsample(fuse_out12)
        fuse_out123 = self.fuse_mode123(out3, fuse_out12_downsample)
        out4 = self.layer4(out3)
        fuse_out123_downsample = self.layer3_downsample(fuse_out123)
        fuse_out1234 = self.fuse_mode1234(out4, fuse_out123_downsample)
        stats = self.pool(fuse_out1234)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a

class BasicBlockRes2Net(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, baseWidth=32, scale=2):
        super(BasicBlockRes2Net, self).__init__()
        width = int(math.floor(planes*(baseWidth/64.0)))
        self.conv1 = conv1x1(in_planes, width*scale, stride)
        self.bn1 = nn.BatchNorm2d(width*scale)
        self.nums = scale -1
        convs=[]
        bns=[]
        for i in range(self.nums):
          convs.append(conv3x3(width,width))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)
        
        self.conv3 = conv1x1(width*scale,planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out,self.width,1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out,sp),1)
        
        out = torch.cat((out,spx[self.nums]),1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out

class Res2Net(nn.Module):
    def __init__(self,
                 block=BasicBlockRes2Net,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=32,
                 feat_dim=80,
                 embedding_size=192,
                 pooling_func='TSTP',
                 two_emb_layer=False):
        super(Res2Net, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embedding_size = embedding_size
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       m_channels * 8,
                                       num_blocks[3],
                                       stride=2)

        self.n_stats = 1 if pooling_func == 'TAP' or pooling_func == "TSDP" else 2
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * block.expansion)
        self.seg_1 = nn.Linear(self.stats_dim * block.expansion * self.n_stats,
                               embedding_size)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embedding_size, affine=False)
            self.seg_2 = nn.Linear(embedding_size, embedding_size)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        # x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)

        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        stats = self.pool(out)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a
        
class BasicBlockRes2Net_1d(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, baseWidth=32, scale=2):
        super(BasicBlockRes2Net_1d, self).__init__()
        width = int(math.floor(planes*(baseWidth/64.0)))
        self.conv1 = conv1x1_1d(in_planes, width*scale, stride)
        self.bn1 = nn.BatchNorm1d(width*scale)
        self.nums = scale -1
        convs=[]
        bns=[]
        for i in range(self.nums):
          convs.append(conv3x3_1d(width,width))
          bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)
        
        self.conv3 = conv1x1_1d(width*scale,planes*self.expansion)
        self.bn3 = nn.BatchNorm1d(planes*self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm1d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out,self.width,1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out,sp),1)
        
        out = torch.cat((out,spx[self.nums]),1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out

class Res2Net_1d(nn.Module):
    def __init__(self,
                 block=BasicBlockRes2Net_1d,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=128,
                 feat_dim=80,
                 embedding_size=192,
                 pooling_func='TSTP',
                 two_emb_layer=False):
        super(Res2Net_1d, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embedding_size = embedding_size
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv1d(80,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(m_channels)
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       m_channels * 8,
                                       num_blocks[3],
                                       stride=2)

        self.n_stats = 1 if pooling_func == 'TAP' or pooling_func == "TSDP" else 2
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * block.expansion)
        # self.seg_1 = nn.Linear(self.stats_dim * block.expansion * self.n_stats,
                              #  embedding_size)
        self.seg_1 = nn.Linear(4096,
                               embedding_size)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embedding_size, affine=False)
            self.seg_2 = nn.Linear(embedding_size, embedding_size)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        # x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)

        # x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print(out.shape)
        stats = self.pool(out)
        # print(stats.shape)
        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a
        
class BasicBlockRes2Net_1d_LSTM(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, baseWidth=32, scale=2):
        super(BasicBlockRes2Net_1d_LSTM, self).__init__()
        width = int(math.floor(planes*(baseWidth/64.0)))
        self.conv1 = conv1x1_1d(in_planes, width*scale, stride)
        self.bn1 = nn.BatchNorm1d(width*scale)
        self.nums = scale -1
        convs=[]
        lstms=[]
        bns=[]
        for i in range(self.nums):
        #   convs.append(conv3x3_1d(width,width))
          lstms.append(nn.LSTM(width, int(width/2), num_layers=1, bias=True, batch_first=True, bidirectional=True))
          bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.lstms = nn.ModuleList(lstms)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)
        
        self.conv3 = conv1x1_1d(width*scale,planes*self.expansion)
        self.bn3 = nn.BatchNorm1d(planes*self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm1d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out,self.width,1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]

          sp = sp.transpose(1,2)
          sp, _ = self.lstms[i](sp)
          sp = sp.transpose(1,2)

          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out,sp),1)
        
        out = torch.cat((out,spx[self.nums]),1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out

class Res2Net_1d_LSTM(nn.Module):
    def __init__(self,
                 block=BasicBlockRes2Net_1d_LSTM,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=128,
                 feat_dim=80,
                 embedding_size=192,
                 pooling_func='TSTP',
                 two_emb_layer=False):
        super(Res2Net_1d_LSTM, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embedding_size = embedding_size
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv1d(80,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(m_channels)
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       m_channels * 8,
                                       num_blocks[3],
                                       stride=2)

        self.n_stats = 1 if pooling_func == 'TAP' or pooling_func == "TSDP" else 2
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * block.expansion)
        # self.seg_1 = nn.Linear(self.stats_dim * block.expansion * self.n_stats,
                              #  embedding_size)
        self.seg_1 = nn.Linear(4096,
                               embedding_size)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embedding_size, affine=False)
            self.seg_2 = nn.Linear(embedding_size, embedding_size)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        # x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)

        # x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print(out.shape)
        stats = self.pool(out)
        # print(stats.shape)
        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a