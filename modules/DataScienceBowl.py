import numpy as np
import matplotlib.pyplot as plt
import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

from kornia.utils import image_to_tensor




class DataScienceBowl(Dataset):
	"""
	Data Science Bowl PyTorch Dataset
	"""

	def __init__(self, root_dir, transform=None):
		
		self.root_dir = root_dir
		self.transform = transform

		self.paths = list()

		for img_dir in os.listdir(root_dir):
			img_dir_path = os.path.join(root_dir, img_dir, 'images')
			mask_dir_path = os.path.join(root_dir, img_dir, 'masks')
			
			img_path = [os.path.join(img_dir_path, f) for f in os.listdir(img_dir_path) if os.path.isfile(os.path.join(img_dir_path, f))][0]
			mask_paths = [os.path.join(mask_dir_path, f) for f in os.listdir(mask_dir_path) if os.path.isfile(os.path.join(mask_dir_path, f))]

			path_obj = { 'image': img_path, 'masks': mask_paths, 'img_id': img_dir }

			self.paths.append(path_obj)


	def __len__(self):
		return len(self.paths)


	def __getitem__(self, idx):

		# Get image
		img_path_obj = self.paths[idx]
		# read image and get rid of Alpha channel
		# img range is 0-1 not 0-255
		img = read_image(img_path_obj['image'], ImageReadMode.RGB).type(torch.FloatTensor) / 255

		# img = image_to_tensor(img)


		# Get masks
		combined_mask = torch.zeros(img[0,:,:].shape)

		for mask_path in img_path_obj['masks']:
			mask = read_image(mask_path, ImageReadMode.GRAY).type(torch.FloatTensor)
			# Combine masks
			combined_mask = torch.logical_or(combined_mask, mask).type(torch.FloatTensor)
	

		if self.transform:
			img = self.transform(img)
			combined_mask = self.transform(combined_mask)

		sample = { 'image': img, 'mask': combined_mask }

		return sample


if __name__ == '__main__':
	import torch
	import torchvision.transforms as T

	transform = T.Compose([
		T.Resize(256),
		T.CenterCrop(224),
	#     T.ToTensor()
	#     T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
	])
	dataset = DataScienceBowl('data/data_science_train', transform=transform)

	print(dataset[0]['image'].shape)

	# plt.imshow(dataset[0]['image'].permute(1,2,0), cmap='gray')
	# plt.show()

	

