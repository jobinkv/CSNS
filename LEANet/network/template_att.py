###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
from network.PosEmbedding import PosEmbedding1D, PosEncoding1D, PosEncodingOnly
from ipdb import set_trace as st
from network.mynn import Upsample
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module','TPL_Module','POS_Inject','LEAM_Module']


class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class TPL_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(TPL_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x,tpl):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(tpl).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class POS_Inject(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim,pos_rfactor ):
        super(POS_Inject, self).__init__()
        self.chanel_in = in_dim
        self.pos_emb1d_1st = PosEncoding1D(pos_rfactor, dim=in_dim, pos_noise=0.1)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.rowpool = nn.AdaptiveAvgPool2d((128//pos_rfactor,1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x,pos):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        x1d = self.rowpool(x).squeeze(3)
        x1d = self.pos_emb1d_1st(x1d,pos)
        x2d = Upsample(x1d.unsqueeze(3),(height, width))

        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x2d).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class VPOS_Injection(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim,pos_rfactor):
        super(VPOS_Injection, self).__init__()
        self.chanel_in = in_dim
        self.pos_emb1d_1st = PosEncoding1D(pos_rfactor, dim=in_dim, pos_noise=0.1)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.colpool = nn.AdaptiveAvgPool2d((1,128//pos_rfactor))

        self.softmax = Softmax(dim=-1)
    def forward(self, x,pos):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        x1d = self.colpool(x).squeeze(2)
        x1d = self.pos_emb1d_1st(x1d,pos)
        x2d = Upsample(x1d.unsqueeze(2),(height, width))
        if False:
            import cv2
            temp1  = x2d[0,0,:,:]
            temp1 = temp1/torch.max(temp1)
            temp1 = np.uint8(temp1.detach().cpu()*255)
            tmp = cv2.applyColorMap(temp1, cv2.COLORMAP_JET)
            cv2.imwrite('pos_rfactor16_cheeting.jpg',tmp)

        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x2d).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class Only_loc(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim,pos_rfactor):
        super(Only_loc, self).__init__()
        self.chanel_in = in_dim
        self.pos_emb1d_wst = PosEncodingOnly(pos_rfactor, dim=in_dim, pos_noise=0.1)
        self.pos_emb1d_hst = PosEncodingOnly(pos_rfactor, dim=in_dim, pos_noise=0.1)

        self.colpool = nn.AdaptiveAvgPool2d((1,128//pos_rfactor))
        self.rowpool = nn.AdaptiveAvgPool2d((128//pos_rfactor,1))

        self.softmax = Softmax(dim=-1)
    def forward(self,x,pos):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        x1d = self.pos_emb1d_wst(pos)
        xwd = Upsample(x1d.unsqueeze(2),(height, width))
        x2d = self.pos_emb1d_hst(pos)
        xhd = Upsample(x2d.unsqueeze(3),(height, width))
        if False:
            import cv2
            temp1  = xhd[0,0,:,:]
            temp1 = temp1/torch.max(temp1)
            temp1 = np.uint8(temp1.detach().cpu()*255)
            tmp = cv2.applyColorMap(temp1, cv2.COLORMAP_JET)
            cv2.imwrite('pos_rfactor16_height.jpg',tmp)
            temp1  = xwd[0,0,:,:]
            temp1 = temp1/torch.max(temp1)
            temp1 = np.uint8(temp1.detach().cpu()*255)
            tmp = cv2.applyColorMap(temp1, cv2.COLORMAP_JET)
            cv2.imwrite('pos_rfactor16_width.jpg',tmp)
            st()
        return xwd,xhd

class Only_loc_addition(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim,pos_rfactor):
        super(Only_loc_addition, self).__init__()
        self.chanel_in = in_dim
        self.pos_rfactor = pos_rfactor
        self.pos_emb1d_wst = PosEncodingOnly(pos_rfactor, dim=in_dim, pos_noise=0.1)
        self.pos_emb1d_hst = PosEncodingOnly(pos_rfactor, dim=in_dim, pos_noise=0.1)
        self.beta = 0.5 #Parameter(torch.zeros(1))
        #self.colpool = nn.AdaptiveAvgPool2d((1,128//pos_rfactor))
        #self.rowpool = nn.AdaptiveAvgPool2d((128//pos_rfactor,1))

        self.softmax = Softmax(dim=-1)
    def forward(self,x,pos):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        x1d = self.pos_emb1d_wst(pos)
        xwd = Upsample(x1d.unsqueeze(2),(height, width))
        x2d = self.pos_emb1d_hst(pos)
        xhd = Upsample(x2d.unsqueeze(3),(height, width))
        if False:
            import cv2
            temp1  = xhd[0,0,:,:]
            temp1 = temp1/torch.max(temp1)
            temp1 = np.uint8(temp1.detach().cpu()*255)
            tmp = cv2.applyColorMap(temp1, cv2.COLORMAP_JET)
            cv2.imwrite('pos_rfactor'+str(self.pos_rfactor)+'_height.jpg',tmp)
            temp1  = xwd[0,0,:,:]
            temp1 = temp1/torch.max(temp1)
            temp1 = np.uint8(temp1.detach().cpu()*255)
            tmp = cv2.applyColorMap(temp1, cv2.COLORMAP_JET)
            cv2.imwrite('pos_rfactor'+str(self.pos_rfactor)+'_width.jpg',tmp)
            xhw = xwd*.5+xhd*.5
            temp1  = xhw[0,0,:,:]
            temp1 = temp1/torch.max(temp1)
            temp1 = np.uint8(temp1.detach().cpu()*255)
            tmp = cv2.applyColorMap(temp1, cv2.COLORMAP_JET)
            cv2.imwrite('pos_rfactor'+str(self.pos_rfactor)+'_widthsHeight.jpg',tmp)
            st()
        return self.beta*xwd+(1-self.beta)*xhd
class POS_Injectv1(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim,pos_rfactor ):
        super(POS_Injectv1, self).__init__()
        self.chanel_in = in_dim
        self.pos_emb1d_1st = PosEncoding1D(pos_rfactor, dim=in_dim, pos_noise=0.1)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.rowpool = nn.AdaptiveAvgPool2d((128//pos_rfactor,1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x,pos):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        x1d = self.rowpool(x).squeeze(3)
        x1d = self.pos_emb1d_1st(x1d,pos)
        x = Upsample(x1d.unsqueeze(3),(height, width))

        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class LEAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim,pos_rfactor):
        super(LEAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.loc = Only_loc_addition(in_dim, pos_rfactor)

        self.softmax = Softmax(dim=-1)
    def forward(self, x,pos):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        loc_embeding = self.loc(x,pos) 
        proj_query = self.query_conv(x+loc_embeding).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

