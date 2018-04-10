from torch import nn
import torch

class GroupNormalization(nn.Module):
	"""for GroupNormalization"""
	def __init__(self, ChannelNum, GroupNum = 32, eps = 1e-10):
		super(GroupNormalization, self).__init__()
		self.GroupNum = GroupNum
		self.gamma = nn.Parameter(torch.ones(ChannnelNum, 1, 1))
		self.beta = nn.Parameter(torch.zeros(ChannnelNum, 1, 1))
		self.eps =  eps

	def forward(self, x):
		N, C, H, W = x.size()
		x = x.view(N, self.GroupNum, -1)

		mean = x.mean(dim = 2, keepdim = True)
		std = x.std(dim = 2, keepdim = True)
		x = (x - mean) / (std + self.eps)
		x = x.view(N, C, H, W)

		return x * self.gamma + self.beta