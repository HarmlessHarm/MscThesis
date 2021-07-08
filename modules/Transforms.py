import torch
import torch.nn as nn
import torchvision.transforms as T
import kornia as K
from kornia.augmentation import AugmentationBase2D

# Random Affine: K.augmentation.RandomAffine(params)(img)

# Random Conv: K.filters(input, kernel)

if torch.cuda.is_available():
    device = torch.device('cuda')
    # print('using cuda')
else:
    device = torch.device('cpu')
    # print('using cpu')

# ColorJitter: K.augmentation.ColorJitter(params)(img)
class ColorJitter(torch.nn.Module):
	"""docstring for ColorJitter"""
	def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, same_on_batch=True, channelwise_hue=False):
		super().__init__()
		self.channelwise_hue = channelwise_hue
		if self.channelwise_hue:
			self.channelwise_transform = T.ColorJitter(brightness=hue)
			hue = 0

		self.global_transform = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

	def forward(self, x):

		if self.channelwise_hue:
			img_channels = [x[:,c,:,:].unsqueeze(1) for c in range(3)]

			img_hat_channels = list()

			for img_c in img_channels:
				img_c_hat = self.channelwise_transform(img_c)
				img_hat_channels.append(img_c_hat)

			x = torch.cat(img_hat_channels, 1)

		return self.global_transform(x)
		

# Noise: K.augmentation.Random_Gaussian_Noise(params)(img)
class Noise(K.augmentation.RandomGaussianNoise):
	def __init__(self, mean=0, std=0, return_transform=False, same_on_batch=True):
		super().__init__(mean, std, return_transform, same_on_batch)


# ElasticTransform: K.geometry.transform.elastic_transform2d(img, params)
class ElasticTransform(nn.Module):
	"""docstring for ElasticTransform"""
	def __init__(self, noise=None, kernel_size=(35,35), sigma=(11,11), alpha=(0.1,0.1), align_corners=False, mode='bilinear', device=None):
		super().__init__()
		self.noise = noise
		self.kernel_size = kernel_size
		self.sigma = sigma
		self.alpha = alpha
		self.align_corners = align_corners
		self.mode = mode
		self.device = device


	def forward(self, image_batch):
		# maybe check image_batch?
		if self.noise is None:
			self.noise = ElasticTransform.generate_local_shifts(shape=image_batch.shape[2:], std=5)
			
			# print(self.noise.shape)
			# print(image_batch.shape)
			
			# if torch.cuda.is_available():
			# 	device = torch.device('cuda')
			# 	print('using cuda')
			# else:
			# 	device = torch.device('cpu')
			# 	print('using cpu')

			B = image_batch.shape[0]
			self.noise = self.noise.repeat(B,1,1,1)
			self.noise = self.noise.to(device)

			# self.noise = self.noise.to(device)
			# print(self.noise.device)
			# print(image_batch.device)

			

		img_hat = K.geometry.transform.elastic_transform2d(image_batch, 
														self.noise, 
														self.kernel_size, 
														self.sigma, 
														self.alpha, 
														self.align_corners, 
														self.mode)

		# img_hat.mean().backward()
		

		return img_hat

	@staticmethod
	def generate_local_shifts(shape, std=7):
	    mean = 0
	    x_shift = std*torch.randn(1, 1, shape[0], shape[1]) + mean
	    y_shift = std*torch.randn(1, 1, shape[0], shape[1]) + mean
	    
	    xy_shifts = torch.cat([x_shift, y_shift], dim=1)
	    return xy_shifts

class RandomConvolution(nn.Module):
	"""docstring for RandomConvolution"""
	def __init__(self, kernel_size=(3,3), min=-1, max=1, border_type='reflect', normalized='False'):
		super().__init__()
		self.kernel_size = kernel_size
		self.min = min
		self.max = max
		self.border_type = border_type
		self.normalized = normalized

		# print(torch.min(self.random_kernel), torch.max(self.random_kernel), torch.mean(self.random_kernel))
		self.random_kernel = (max - min) * torch.rand(kernel_size) + min
		# normalize to 1
		self.random_kernel = self.random_kernel / torch.sum(self.random_kernel)
		# Add channel batch dimension
		self.random_kernel = self.random_kernel.unsqueeze(0)

	def forward(self, image_batch):

		return K.filters.filter2D(image_batch, self.random_kernel, self.border_type, self.normalized)
		# return 

		
class Sobel(nn.Module):

	def __init__(self, normalized=True, eps=1e-06):
		super().__init__()
		self.normalized = normalized
		self.eps = eps

	def forward(self, image_batch):

		return K.filters.sobel(image_batch, self.normalized, self.eps)



class CombiTransform(nn.Module):

	def __init__(self, combination=['elastic', 'color', 'random', 'noise'], device=None):
		super().__init__()

		transform_options = {
			'elastic': ElasticTransform(device=device),
			'color': ColorJitter(hue=0.5, channelwise_hue=True),
			'random': RandomConvolution(kernel_size=(2,2),min=-0.5, max=0.5),  
			'noise': Noise(0, 0.1),
			'sobel': Sobel(),
		}

		print(combination)

		self.device = device

		self.transform = T.Compose([transform_options[name] for name in combination])

	def forward(self, input):

		return self.transform(input)


