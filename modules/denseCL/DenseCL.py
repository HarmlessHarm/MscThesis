
import torch

from segmentation_models_pytorch.encoders import get_encoder


from ..Transforms import CombiTransform
from . import ProjectionModule
from . import DenseContrastiveLoss
from . import GlobalContrastiveLoss


class DenseCL(torch.nn.Module):
    
    def __init__(self, lam=0.5, combination=None):
        super().__init__()
        
        self.Lambda = lam
        
        if combination is not None:
            self.transformer = CombiTransform(combination)
        else:
            self.transformer = CombiTransform()
            
        self.encoder = get_encoder('resnet18', 3, 5, None)
#         self.encoder = get_encoder('resnet18', 3, 5, 'imagenet')
        self.encoder.layer4 = torch.nn.Identity()
        
        self.project = ProjectionModule()
        
#         self.enc_proj = DenseContrastiveModule()
        
        self.dense_loss = DenseContrastiveLoss()
        self.glob_loss = GlobalContrastiveLoss()
        
        
    def forward(self, image_1, image_2):
        
        img_Q = self.transformer(image_1)
        img_pos = self.transformer(image_1)
        img_neg = self.transformer(image_2)
        
        img_Q = self.encoder(img_Q)[-1]
        img_pos = self.encoder(img_pos)[-1]
        img_neg = self.encoder(img_neg)[-1]
        
        dense_Q, glob_Q = self.project(img_Q)
        dense_pos, glob_pos = self.project(img_pos)
        dense_neg, glob_neg = self.project(img_neg)
        
        d_loss = self.dense_loss(dense_Q, dense_pos, dense_neg)
        g_loss = self.glob_loss(glob_Q, glob_pos, glob_neg)
        
        loss = (1 - self.Lambda) * g_loss + self.Lambda * d_loss
        
        return loss