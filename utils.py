from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import shutil
import numpy as np
import torch


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val   = 0
		self.avg   = 0
		self.sum   = 0
		self.count = 0

	def update(self, val, n=1):
		self.val   = val
		self.sum   += val * n
		self.count += n
		self.avg   = self.sum / self.count


def count_parameters_in_MB(model):
	return sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6


def create_exp_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	print('Experiment dir : {}'.format(path))


def load_pretrained_model(model, pretrained_dict):
	model_dict = model.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict) 
	# 3. load the new state dict
	model.load_state_dict(model_dict)


def transform_time(s):
	m, s = divmod(int(s), 60)
	h, m = divmod(m, 60)
	return h,m,s


def save_checkpoint(state, is_best, save_root):
	save_path = os.path.join(save_root, 'checkpoint.pth.tar')
	torch.save(state, save_path)
	if is_best:
		best_save_path = os.path.join(save_root, 'model_best.pth.tar')
		shutil.copyfile(save_path, best_save_path)


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred    = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res
