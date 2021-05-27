import torch
import torch.nn as nn
import kornia as K
from kornia.augmentation import AugmentationBase2D


# Random Affine: K.augmentation.RandomAffine(params)(img)

# Random Conv: K.filters(input, kernel)



# ColorJitter: K.augmentation.ColorJitter(params)(img)
class ColorJitter(K.augmentation.ColorJitter):
	"""docstring for ColorJitter"""
	def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, return_transform=False, same_on_batch=True):
		super().__init__(brightness, contrast, saturation, hue, return_transform, same_on_batch)
		

# Noise: K.augmentation.Random_Gaussian_Noise(params)(img)
class Noise(K.augmentation.RandomGaussianNoise):
	def __init__(self, mean=0, std=0, return_transform=False, same_on_batch=True):
		super().__init__(mean, std, return_transform, same_on_batch)


# ElasticTransform: K.geometry.transform.elastic_transform2d(img, params)
class ElasticTransform(nn.Module):
	"""docstring for ElasticTransform"""
	def __init__(self, noise, kernel_size=(3,3), sigma=(4.0,4.0), alpha=(32.0,32.0), align_corners=False, mode='bilinear'):
		super().__init__()
		self.noise = noise
		self.kernel_size = kernel_size
		self.sigma = sigma
		self.alpha = alpha
		self.align_corners = align_corners
		self.mode = mode


	def forward(self, image_batch):
		# maybe check image_batch?
		img_hat = K.geometry.transform.elastic_transform2d(image_batch, 
														self.noise, 
														self.kernel_size, 
														self.sigma, 
														self.alpha, 
														self.align_corners, 
														self.mode)

		# img_hat.mean().backward()
		

		return img_hat


class RandomConvolution(nn.Module):
	"""docstring for RandomConvolution"""
	def __init__(self, kernel_size=(3,3), min=-.1, max=.1, border_type='reflect', normalized='False'):
		super().__init__()
		self.kernel_size = kernel_size
		self.min = min
		self.max = max
		self.border_type = border_type
		self.normalized = normalized
		self.random_kernel = (max - min) * torch.rand(kernel_size) + min
		self.random_kernel = self.random_kernel.unsqueeze(0)
		# normalize to 1
		self.random_kernel = self.random_kernel / torch.sum(self.random_kernel)

		print(self.random_kernel)




	def forward(self, image_batch):

		return K.filters.filter2D(image_batch, self.random_kernel, self.border_type, self.normalized)

		