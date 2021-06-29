
import torch

class ProjectionModule(torch.nn.Module):
    
    def __init__(self, dense_head_shape='5x5'):
        super().__init__()
        # Reduce bottleneck from 14x14x256 to 5x5x128 or 3x3x128
        if dense_head_shape == '5x5':
            self.densePool = torch.nn.MaxPool2d(3, padding=1)
            self.projDense = torch.nn.Conv2d(256, 128, (1,1))
        
        else:
            self.densePool = torch.nn.MaxPool2d(4)
            self.projDense = torch.nn.Conv2d(256, 128, (1,1))

        self.projGlobal = torch.nn.AvgPool2d(14)
        self.batchNorm = torch.nn.BatchNorm2d(256)
        pass
    
    def forward(self, input):
#         encoded = self.encoder(input)[-1]
        normalized = self.batchNorm(input)
        
        pooled = self.densePool(normalized)
        dense = self.projDense(pooled)
        glob = self.projGlobal(normalized)
        
        return dense, glob

