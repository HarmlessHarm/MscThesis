# Source https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-training-3-4-8242d31de234
import numpy as np
import torch
from sklearn.metrics import jaccard_score

class Trainer:
	"""Trainer class for pytorch"""
	def __init__(self, 
		model: torch.nn.Module,
		device: torch.device,
		criterion: torch.nn.Module,
		optimizer: torch.optim.Optimizer,
		training_DataLoader: torch.utils.data.Dataset,
		validation_DataLoader: torch.utils.data.Dataset = None,
		test_DataLoader: torch.utils.data.Dataset = None,
		lr_scheduler: torch.optim.lr_scheduler = None,
		epochs: int = 100,
		epoch: int = 0,
		notebook: bool = False,
		seed: int = None,
		):


		self.model = model
		self.device = device
		self.criterion = criterion
		self.optimizer = optimizer
		self.training_DataLoader = training_DataLoader
		self.validation_DataLoader = validation_DataLoader
		self.test_DataLoader = test_DataLoader
		self.lr_scheduler = lr_scheduler
		self.epochs = epochs
		self.epoch = epoch
		self.notebook = notebook

		self.training_loss = list()
		self.validation_loss = list()
		self.test_loss = 0
		self.learning_rate = list()

		if seed is not None:
			torch.manual_seed(seed)
			random.seed(seed)
			np.random.seed(seed)

	def run_trainer(self):

		if self.notebook:
			from tqdm.notebook import tqdm, trange
		else:
			from tqdm import tqdm, trange

		progressbar = trange(self.epochs, desc="Progress")

		for i in progressbar:

			self.epoch += 1

			self._train()

			if self.validation_DataLoader is not None:
				self._validate()

			if self.lr_scheduler != None:
				if self.validation_DataLoader != None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
					self.lr_scheduler.batch(self.validation_loss[i])
				else:
					self.lr_scheduler.batch()

		if self.test_DataLoader is not None:
			self._test()

		return self.training_loss, self.validation_loss, self.learning_rate, self.test_loss, self.test_IoU



	def _train(self):

		if self.notebook:
			from tqdm.notebook import tqdm, trange
		else:
			from tqdm import tqdm, trange


		self.model.train()
		train_losses = list()
		batch_iter = tqdm(enumerate(self.training_DataLoader), "Training", 
							total=len(self.training_DataLoader), 
							leave=False)


		# Change (x, y) for DataScienceBowl Dataset
		for i, sample in batch_iter:
			x, y = sample.values() 
			input, target = x.to(self.device), y.to(self.device)

			self.optimizer.zero_grad()
			out = self.model(input)

			# out = out.squeeze(1)
			# target = target.squeeze(1)

			loss = self.criterion(out, target)
			loss_value = loss.item()
			train_losses.append(loss_value)
			loss.backward()
			self.optimizer.step()

			batch_iter.set_description(f'Training: (loss {loss_value:.4f})')

		self.training_loss.append(np.mean(train_losses))
		self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

		batch_iter.close()


	def _validate(self):

		if self.notebook:
			from tqdm.notebook import tqdm, trange
		else:
			from tqdm import tqdm, trange

		self.model.eval()
		valid_losses = list() 
		batch_iter = tqdm(enumerate(self.validation_DataLoader), "Validation", total=len(self.validation_DataLoader), leave=False)

		for i, sample in batch_iter:

			input, target = sample['image'].to(self.device), sample['mask'].to(self.device)

			with torch.no_grad():
				out = self.model(input)

				loss = self.criterion(out, target)
				loss_value = loss.item()
				valid_losses.append(loss_value)

				batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

		self.validation_loss.append(np.mean(valid_losses))

		batch_iter.close()

	def _test(self):
		# TODO work with test batch size larger than 1
		if self.notebook:
			from tqdm.notebook import tqdm, trange
		else:
			from tqdm import tqdm, trange

		self.model.eval()
		test_losses = list()
		test_IoUs = list()

		batch_iter = tqdm(enumerate(self.test_DataLoader), "Test", total=len(self.test_DataLoader))

		for i, sample in batch_iter:

			input, target = sample['image'].to(self.device) , sample['mask'].to(self.device)

			with torch.no_grad():
				# predict
				out = self.model(input)
				# loss
				loss = self.criterion(out, target)
				test_losses.append(loss.item())
				# IoU
				# pred_mask = torch.sigmoid(out) > 0.5
				# mask = pred_mask.detach().cpu().numpy().reshape(-1).astype('int')
				# target = target.detach().cpu().numpy().reshape(-1).astype('int')

				batch_IoU = calculate_IoU(out, target)
				test_IoUs.append(batch_IoU)


		self.test_loss = np.mean(test_losses)
		self.test_IoU = np.mean(test_IoUs)

		batch_iter.close()

	def calculate_IoU(out, target):
		# IoU
		pred_mask = torch.sigmoid(out) > 0.5
		mask = pred_mask.detach().cpu().numpy().reshape(-1).astype('int')
		target = target.detach().cpu().numpy().reshape(-1).astype('int')

		return jaccard_score(mask, target)

if __name__ == '__main__':
	from DataScienceBowl import DataScienceBowl
	import segmentation_models_pytorch as smp
	import matplotlib.pyplot as plt

	import torch
	from torch.utils.data import DataLoader
	import torchvision.transforms as T


	if torch.cuda.is_available():
		device = torch.device('cuda')
		print('using cuda')
	else:
		torch.device('cpu')
		print('using cpu')

	model = smp.Unet(
		encoder_name="resnet18",
		encoder_weights="imagenet",
		in_channels=3,
		classes=1,
	)

	transform = T.Compose([
		T.Resize(256),
		T.CenterCrop(224),
	#     T.ToTensor()
	#     T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
	])
	model.to(device)
	# print(model)

	# Dataset
	dataset = DataScienceBowl('data/data_science_train', transform=transform)

	indices = torch.randperm(len(dataset)).tolist()
	train_dataset = torch.utils.data.Subset(dataset, indices[:50])
	validation_dataset = torch.utils.data.Subset(dataset, indices[-50:])

	dataLoader_training = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
	dataLoader_validation = DataLoader(dataset=validation_dataset, batch_size=4, shuffle=True)



	# Binary Cross Entropy with Logits
	criterion = torch.nn.BCEWithLogitsLoss()
	# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

	trainer = Trainer(model=model,
					 device=device,
					 criterion=criterion,
					 optimizer=optimizer,
					 training_DataLoader=dataLoader_training,
					 validation_DataLoader=dataLoader_validation,
					 notebook=False)

	trainer.run_trainer()