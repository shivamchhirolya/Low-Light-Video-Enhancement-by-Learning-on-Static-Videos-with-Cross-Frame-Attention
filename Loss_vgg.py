import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16

cuda1 = torch.device('cuda:0')
                
class SSIM(torch.nn.Module):
    
    def __init__(self, window_size = 7, size_average = True):
        super(SSIM, self).__init__()

        self.mse_loss = nn.L1Loss()
        self.size_average = size_average
   
        vgg = vgg16(pretrained=True)
        self.loss_network = nn.Sequential(*list(vgg.features)[:10]).eval()
        #self.loss_network = nn.Sequential(*list(vgg.features)[:17]).eval()
        self.loss_network = self.loss_network.cuda(cuda1)
       
    def forward(self, lp, gt_img):
          
        ## vgg_loss and pixel_loss
        
        pixel_loss = self.mse_loss(torch.cat(((lp[:,0:1,:,:]-0.485)/0.229, (lp[:,1:2,:,:]-0.456)/0.224, (lp[:,2:3,:,:]-0.406)/0.225), dim=1),  torch.cat(((gt_img[:,0:1,:,:]-0.485)/0.229, (gt_img[:,1:2,:,:]-0.456)/0.224, (gt_img[:,2:3,:,:]-0.406)/0.225), dim=1))
    
        out_f2 = self.loss_network(torch.cat(((lp[:,0:1,:,:]-0.485)/0.229, (lp[:,1:2,:,:]-0.456)/0.224, (lp[:,2:3,:,:]-0.406)/0.225), dim=1))
        gt_f2 = self.loss_network(torch.cat(((gt_img[:,0:1,:,:]-0.485)/0.229, (gt_img[:,1:2,:,:]-0.456)/0.224, (gt_img[:,2:3,:,:]-0.406)/0.225), dim=1))
        
        vgg_loss = self.mse_loss(out_f2, gt_f2)
        
        # at 2nd max pooling layer
        loss = pixel_loss + vgg_loss 

        return loss