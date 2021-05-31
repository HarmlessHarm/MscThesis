import torch
import segmentation_models_pytorch as smp

# if torch.cuda.is_available():
# 	device = torch.device('cuda')
# 	print('using cuda')
# else:
# 	torch.device('cpu')
# 	print('using cpu')

class CustomUnet(torch.nn.Module):

	class DecoderIdentity(torch.nn.Identity):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
		
		def forward(self, x, skip=None):
			return x

	"""docstring for CustomUnet"""
	def __init__(self):
		super(CustomUnet, self).__init__()

		self.model = smp.Unet(
			encoder_name = "resnet18",
			encoder_weights = "imagenet",
			in_channels = 3,
			classes = 1
		)

		self.model.encoder.layer4 = torch.nn.Identity()
		self.model.decoder.blocks[0] = self.DecoderIdentity()


	def forward(self, x):
		return self.model(x)