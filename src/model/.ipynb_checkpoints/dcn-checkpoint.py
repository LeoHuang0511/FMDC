import torch
import torchvision.ops
from torch import nn
import torch.nn.functional as F

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 offset_groups = 4,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False,
                 offset = None
                 ):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        self.kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.offset_groups = offset_groups
        self.offset_conv = nn.Conv2d(in_channels // offset_groups * 2, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)
        self.offset = offset

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     offset_groups * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)
        self.offset_loss = 0
    def forward(self, warp_ref, source):
        # h, w = ref.shape[2:]
        # max_offset = max(h, w)/4.
        num_group_channel = warp_ref.size()[1] // self.offset_groups
        offset_map = []
        if self.offset != None:
            offset_map = self.offset.repeat(1,self.kernel_size[0] * self.kernel_size[1],1,1)
        else:
            for i in range(self.offset_groups):
                offset_input = torch.concat([warp_ref[:,i*num_group_channel:(i+1)*num_group_channel,:,:], 
                    source[:,i*num_group_channel:(i+1)*num_group_channel,:,:]], axis = 1)
                offset_map.append(self.offset_conv(offset_input))
            offset_map = torch.concat(offset_map, axis = 1)
        #offset range
        offset_map = 100* torch.sigmoid(offset_map) - 50
        offset_y = offset_map[:,0::2,:,:] #vertical
        offset_x = offset_map[:,1::2,:,:] #horizontal
        offset_y_mean = torch.mean(offset_x,dim = 1, keepdims = True)
        offset_x_mean = torch.mean(offset_y,dim = 1, keepdims = True)
        offset = torch.concat([offset_x_mean,offset_y_mean],axis = 1)
        # offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(warp_ref))
        self.offset_loss = F.l1_loss(offset_x,offset_x_mean.detach()) + F.l1_loss(offset_y, offset_y_mean.detach())
        x = torchvision.ops.deform_conv2d(input=warp_ref, 
                                          offset=offset_map, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x,offset

