import torch
import torch.nn as nn
import util

def define_tsnet(name, num_class, cuda=True):
	if name == 'resnet20':
		net = resnet20(num_class=num_class)
	elif name == 'resnet110':
		net = resnet110(num_class=num_class)
	else:
		raise Exception('model name does not exist.')

	if cuda:
		net = torch.nn.DataParallel(net).cuda()
	util.print_network(net)
	return net

class resblock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(resblock, self).__init__()
		self.downsample = (in_channels != out_channels)
		if self.downsample:
			self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
			self.ds    = nn.Sequential(*[
							nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
							nn.BatchNorm2d(out_channels)
							])
		else:
			self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
			self.ds    = None
		self.bn1   = nn.BatchNorm2d(out_channels)
		self.relu  = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2   = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample:
			residual = self.ds(x)

		out += residual
		out = self.relu(out)

		return out

class resnet20(nn.Module):
	def __init__(self, num_class):
		super(resnet20, self).__init__()
		self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1     = nn.BatchNorm2d(16)
		self.relu    = nn.ReLU(inplace=True)

		self.res1 = self.make_layer(resblock, 3, 16, 16)
		self.res2 = self.make_layer(resblock, 3, 16, 32)
		self.res3 = self.make_layer(resblock, 3, 32, 64)

		self.avgpool = nn.AvgPool2d(8)
		self.fc      = nn.Linear(64, num_class)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def make_layer(self, block, num, in_channels, out_channels):
		layers = [block(in_channels, out_channels)]
		for i in range(num-1):
			layers.append(block(out_channels, out_channels))
		return nn.Sequential(*layers)

	def forward(self, x):
		pre = self.conv1(x)
		pre = self.bn1(pre)
		pre = self.relu(pre)

		rb1 = self.res1(pre)
		rb2 = self.res2(rb1)
		rb3 = self.res3(rb2)

		out = self.avgpool(rb3)
		out = out.view(out.size(0), -1)
		out = self.fc(out)

		return pre, rb1, rb2, rb3, out

class resnet110(nn.Module):
	def __init__(self, num_class):
		super(resnet110, self).__init__()
		self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1     = nn.BatchNorm2d(16)
		self.relu    = nn.ReLU(inplace=True)

		self.res1 = self.make_layer(resblock, 18, 16, 16)
		self.res2 = self.make_layer(resblock, 18, 16, 32)
		self.res3 = self.make_layer(resblock, 18, 32, 64)

		self.avgpool = nn.AvgPool2d(8)
		self.fc      = nn.Linear(64, num_class)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def make_layer(self, block, num, in_channels, out_channels):
		layers = [block(in_channels, out_channels)]
		for i in range(num-1):
			layers.append(block(out_channels, out_channels))
		return nn.Sequential(*layers)

	def forward(self, x):
		pre = self.conv1(x)
		pre = self.bn1(pre)
		pre = self.relu(pre)

		rb1 = self.res1(pre)
		rb2 = self.res2(rb1)
		rb3 = self.res3(rb2)

		out = self.avgpool(rb3)
		out = out.view(out.size(0), -1)
		out = self.fc(out)

		return pre, rb1, rb2, rb3, out


# for train_ft (factor transfer)
def define_paraphraser(k, cuda=True):
	net = paraphraser(k)
	if cuda:
		net = torch.nn.DataParallel(net).cuda()
	util.print_network(net)
	return net

class paraphraser(nn.Module):
	def __init__(self, k):
		super(paraphraser, self).__init__()
		self.encoder = nn.Sequential(*[
				nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
				nn.BatchNorm2d(64),
				nn.ReLU(),
				nn.Conv2d(64, int(64*k), kernel_size=3, stride=1, padding=1, bias=False)
			])
		self.decoder = nn.Sequential(*[
				nn.BatchNorm2d(int(64*k)),
				nn.ReLU(),
				nn.Conv2d(int(64*k), 64, kernel_size=3, stride=1, padding=1, bias=False),
				nn.BatchNorm2d(64),
				nn.ReLU(),
				nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
			])

		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		z   = self.encoder(x)
		out = self.decoder(z)
		return z, out

def define_translator(k, cuda=True):
	net = translator(k)
	if cuda:
		net = torch.nn.DataParallel(net).cuda()
	util.print_network(net)
	return net

class translator(nn.Module):
	def __init__(self, k):
		super(translator, self).__init__()
		self.encoder = nn.Sequential(*[
				nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
				nn.BatchNorm2d(64),
				nn.ReLU(),
				nn.Conv2d(64, int(64*k), kernel_size=3, stride=1, padding=1, bias=False)
			])

		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		z   = self.encoder(x)
		return z
