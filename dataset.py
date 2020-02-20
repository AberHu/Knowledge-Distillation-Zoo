from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import numpy as np
from PIL import Image
import torchvision.datasets as dst

'''
Modified from https://github.com/HobbitLong/RepDistiller/blob/master/dataset/cifar100.py
'''

class CIFAR10IdxSample(dst.CIFAR10):
	def __init__(self, root, train=True, 
				 transform=None, target_transform=None,
				 download=False, n=4096, mode='exact', percent=1.0):
		super().__init__(root=root, train=train, download=download,
						 transform=transform, target_transform=target_transform)
		self.n = n
		self.mode = mode

		num_classes = 10
		num_samples = len(self.data)
		labels = self.targets

		self.cls_positive = [[] for _ in range(num_classes)]
		for i in range(num_samples):
			self.cls_positive[labels[i]].append(i)

		self.cls_negative = [[] for _ in range(num_classes)]
		for i in range(num_classes):
			for j in range(num_classes):
				if j == i:
					continue
				self.cls_negative[i].extend(self.cls_positive[j])

		self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
		self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

		if 0 < percent < 1:
			num = int(len(self.cls_negative[0]) * percent)
			self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:num]
								 for i in range(num_classes)]

		self.cls_positive = np.asarray(self.cls_positive)
		self.cls_negative = np.asarray(self.cls_negative)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]

		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		if self.mode == 'exact':
			pos_idx = index
		elif self.mode == 'relax':
			pos_idx = np.random.choice(self.cls_positive[target], 1)[0]
		else:
			raise NotImplementedError(self.mode)
		replace = True if self.n > len(self.cls_negative[target]) else False
		neg_idx = np.random.choice(self.cls_negative[target], self.n, replace=replace)
		sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

		return img, target, index, sample_idx


class CIFAR100IdxSample(dst.CIFAR100):
	def __init__(self, root, train=True, 
				 transform=None, target_transform=None,
				 download=False, n=4096, mode='exact', percent=1.0):
		super().__init__(root=root, train=train, download=download,
						 transform=transform, target_transform=target_transform)
		self.n = n
		self.mode = mode

		num_classes = 100
		num_samples = len(self.data)
		labels = self.targets

		self.cls_positive = [[] for _ in range(num_classes)]
		for i in range(num_samples):
			self.cls_positive[labels[i]].append(i)

		self.cls_negative = [[] for _ in range(num_classes)]
		for i in range(num_classes):
			for j in range(num_classes):
				if j == i:
					continue
				self.cls_negative[i].extend(self.cls_positive[j])

		self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
		self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

		if 0 < percent < 1:
			num = int(len(self.cls_negative[0]) * percent)
			self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:num]
								 for i in range(num_classes)]

		self.cls_positive = np.asarray(self.cls_positive)
		self.cls_negative = np.asarray(self.cls_negative)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]

		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		if self.mode == 'exact':
			pos_idx = index
		elif self.mode == 'relax':
			pos_idx = np.random.choice(self.cls_positive[target], 1)[0]
		else:
			raise NotImplementedError(self.mode)
		replace = True if self.n > len(self.cls_negative[target]) else False
		neg_idx = np.random.choice(self.cls_negative[target], self.n, replace=replace)
		sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

		return img, target, index, sample_idx

