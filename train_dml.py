from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst

from utils import AverageMeter, accuracy, transform_time
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB
from network import define_tsnet
from kd_losses import *

parser = argparse.ArgumentParser(description='deep mutual learning (only two nets)')

# various path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')
parser.add_argument('--net1_init', type=str, required=True, help='initial parameters of net1')
parser.add_argument('--net2_init', type=str, required=True, help='initial parameters of net2')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')

# net and dataset choosen
parser.add_argument('--data_name', type=str, required=True, help='name of dataset') # cifar10/cifar100
parser.add_argument('--net1_name', type=str, required=True, help='name of net1')    # resnet20/resnet110
parser.add_argument('--net2_name', type=str, required=True, help='name of net2')    # resnet20/resnet110

# hyperparameter lambda
parser.add_argument('--lambda_kd', type=float, default=1.0)


args, unparsed = parser.parse_known_args()

args.save_root = os.path.join(args.save_root, args.note)
create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)
		cudnn.enabled = True
		cudnn.benchmark = True
	logging.info("args = %s", args)
	logging.info("unparsed_args = %s", unparsed)

	logging.info('----------- Network Initialization --------------')
	net1 = define_tsnet(name=args.net1_name, num_class=args.num_class, cuda=args.cuda)
	checkpoint = torch.load(args.net1_init)
	load_pretrained_model(net1, checkpoint['net'])
	logging.info('Net1: %s', net1)
	logging.info('Net1 param size = %fMB', count_parameters_in_MB(net1))

	net2 = define_tsnet(name=args.net2_name, num_class=args.num_class, cuda=args.cuda)
	checkpoint = torch.load(args.net2_init)
	load_pretrained_model(net2, checkpoint['net'])
	logging.info('Net2: %s', net1)
	logging.info('Net2 param size = %fMB', count_parameters_in_MB(net2))
	logging.info('-----------------------------------------------')

	# initialize optimizer
	optimizer1 = torch.optim.SGD(net1.parameters(),
								 lr = args.lr, 
								 momentum = args.momentum, 
								 weight_decay = args.weight_decay,
								 nesterov = True)
	optimizer2 = torch.optim.SGD(net2.parameters(),
								 lr = args.lr, 
								 momentum = args.momentum, 
								 weight_decay = args.weight_decay,
								 nesterov = True)

	# define loss functions
	criterionKD = DML()
	if args.cuda:
		criterionCls = torch.nn.CrossEntropyLoss().cuda()
	else:
		criterionCls = torch.nn.CrossEntropyLoss()

	# define transforms
	if args.data_name == 'cifar10':
		dataset = dst.CIFAR10
		mean = (0.4914, 0.4822, 0.4465)
		std  = (0.2470, 0.2435, 0.2616)
	elif args.data_name == 'cifar100':
		dataset = dst.CIFAR100
		mean = (0.5071, 0.4865, 0.4409)
		std  = (0.2673, 0.2564, 0.2762)
	else:
		raise Exception('Invalid dataset name...')

	train_transform = transforms.Compose([
			transforms.Pad(4, padding_mode='reflect'),
			transforms.RandomCrop(32),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean,std=std)
		])
	test_transform = transforms.Compose([
			transforms.CenterCrop(32),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean,std=std)
		])

	# define data loader
	train_loader = torch.utils.data.DataLoader(
			dataset(root      = args.img_root,
					transform = train_transform,
					train     = True,
					download  = True),
			batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(
			dataset(root      = args.img_root,
					transform = test_transform,
					train     = False,
					download  = True),
			batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

	# warp nets and criterions for train and test
	nets = {'net1':net1, 'net2':net2}
	criterions = {'criterionCls':criterionCls, 'criterionKD':criterionKD}
	optimizers = {'optimizer1':optimizer1, 'optimizer2':optimizer2}

	best_top1 = 0
	best_top5 = 0
	for epoch in range(1, args.epochs+1):
		adjust_lr(optimizers, epoch)

		# train one epoch
		epoch_start_time = time.time()
		train(train_loader, nets, optimizers, criterions, epoch)

		# evaluate on testing set
		logging.info('Testing the models......')
		test_top11, test_top15, test_top21, test_top25 = test(test_loader, nets, criterions)
		
		epoch_duration = time.time() - epoch_start_time
		logging.info('Epoch time: {}s'.format(int(epoch_duration)))

		# save model
		is_best = False
		if max(test_top11, test_top21) > best_top1:
			best_top1 = max(test_top11, test_top21)
			best_top5 = max(test_top15, test_top25)
			is_best = True
		logging.info('Saving models......')
		save_checkpoint({
			'epoch': epoch,
			'net1': net1.state_dict(),
			'net2': net2.state_dict(),
			'prec1@1': test_top11,
			'prec1@5': test_top15,
			'prec2@1': test_top21,
			'prec2@5': test_top25,
		}, is_best, args.save_root)


def train(train_loader, nets, optimizers, criterions, epoch):
	batch_time  = AverageMeter()
	data_time   = AverageMeter()
	cls1_losses = AverageMeter()
	kd1_losses  = AverageMeter()
	cls2_losses = AverageMeter()
	kd2_losses  = AverageMeter()
	top11       = AverageMeter()
	top15       = AverageMeter()
	top21       = AverageMeter()
	top25       = AverageMeter()

	net1 = nets['net1']
	net2 = nets['net2']

	criterionCls = criterions['criterionCls']
	criterionKD  = criterions['criterionKD']

	optimizer1 = optimizers['optimizer1']
	optimizer2 = optimizers['optimizer2']

	net1.train()
	net2.train()

	end = time.time()
	for i, (img, target) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)

		if args.cuda:
			img = img.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

		_, _, _, _, _, out1 = net1(img)
		_, _, _, _, _, out2 = net2(img)

		# for net1
		cls1_loss = criterionCls(out1, target)
		kd1_loss  = criterionKD(out1, out2.detach()) * args.lambda_kd
		net1_loss = cls1_loss + kd1_loss

		prec11, prec15 = accuracy(out1, target, topk=(1,5))
		cls1_losses.update(cls1_loss.item(), img.size(0))
		kd1_losses.update(kd1_loss.item(), img.size(0))
		top11.update(prec11.item(), img.size(0))
		top15.update(prec15.item(), img.size(0))

		# for net2
		cls2_loss = criterionCls(out2, target)
		kd2_loss  = criterionKD(out2, out1.detach()) * args.lambda_kd
		net2_loss = cls2_loss + kd2_loss

		prec21, prec25 = accuracy(out2, target, topk=(1,5))
		cls2_losses.update(cls2_loss.item(), img.size(0))
		kd2_losses.update(kd2_loss.item(), img.size(0))
		top21.update(prec21.item(), img.size(0))
		top25.update(prec25.item(), img.size(0))

		# update net1 & net2
		optimizer1.zero_grad()
		net1_loss.backward()
		optimizer1.step()

		optimizer2.zero_grad()
		net2_loss.backward()
		optimizer2.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
					   'Time:{batch_time.val:.4f} '
					   'Data:{data_time.val:.4f}  '
					   'Cls1:{cls1_losses.val:.4f}({cls1_losses.avg:.4f})  '
					   'KD1:{kd1_losses.val:.4f}({kd1_losses.avg:.4f})  '
					   'Cls2:{cls2_losses.val:.4f}({cls2_losses.avg:.4f})  '
					   'KD2:{kd2_losses.val:.4f}({kd2_losses.avg:.4f})  '
					   'prec1@1:{top11.val:.2f}({top11.avg:.2f})  '
					   'prec1@5:{top15.val:.2f}({top15.avg:.2f})  '
					   'prec2@1:{top21.val:.2f}({top21.avg:.2f})  '
					   'prec2@5:{top25.val:.2f}({top25.avg:.2f})'.format(
					   epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
					   cls1_losses=cls1_losses, kd1_losses=kd1_losses, top11=top11, top15=top15,
					   cls2_losses=cls2_losses, kd2_losses=kd2_losses, top21=top21, top25=top25))
			logging.info(log_str)


def test(test_loader, nets, criterions):
	cls1_losses = AverageMeter()
	kd1_losses  = AverageMeter()
	cls2_losses = AverageMeter()
	kd2_losses  = AverageMeter()
	top11       = AverageMeter()
	top15       = AverageMeter()
	top21       = AverageMeter()
	top25       = AverageMeter()

	net1 = nets['net1']
	net2 = nets['net2']

	criterionCls = criterions['criterionCls']
	criterionKD  = criterions['criterionKD']

	net1.eval()
	net2.eval()

	end = time.time()
	for i, (img, target) in enumerate(test_loader, start=1):
		if args.cuda:
			img = img.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

		with torch.no_grad():
			_, _, _, _, _, out1 = net1(img)
			_, _, _, _, _, out2 = net2(img)

		# for net1
		cls1_loss = criterionCls(out1, target)
		kd1_loss  = criterionKD(out1, out2.detach()) * args.lambda_kd

		prec11, prec15 = accuracy(out1, target, topk=(1,5))
		cls1_losses.update(cls1_loss.item(), img.size(0))
		kd1_losses.update(kd1_loss.item(), img.size(0))
		top11.update(prec11.item(), img.size(0))
		top15.update(prec15.item(), img.size(0))

		# for net2
		cls2_loss = criterionCls(out2, target)
		kd2_loss  = criterionKD(out2, out1.detach()) * args.lambda_kd

		prec21, prec25 = accuracy(out2, target, topk=(1,5))
		cls2_losses.update(cls2_loss.item(), img.size(0))
		kd2_losses.update(kd2_loss.item(), img.size(0))
		top21.update(prec21.item(), img.size(0))
		top25.update(prec25.item(), img.size(0))

	f_l  = [cls1_losses.avg, kd1_losses.avg, top11.avg, top15.avg]
	f_l += [cls2_losses.avg, kd2_losses.avg, top21.avg, top25.avg]
	logging.info('Cls1: {:.4f}, KD1: {:.4f}, Prec1@1: {:.2f}, Prec1@5: {:.2f}'
		  		 'Cls2: {:.4f}, KD2: {:.4f}, Prec2@1: {:.2f}, Prec2@5: {:.2f}'.format(*f_l))

	return top11.avg, top15.avg, top21.avg, top25.avg


def adjust_lr(optimizers, epoch):
	scale   = 0.1
	lr_list =  [args.lr] * 100
	lr_list += [args.lr*scale] * 50
	lr_list += [args.lr*scale*scale] * 50

	lr = lr_list[epoch-1]
	logging.info('epoch: {}  lr: {:.3f}'.format(epoch, lr))
	for param_group in optimizers['optimizer1'].param_groups:
		param_group['lr'] = lr
	for param_group in optimizers['optimizer2'].param_groups:
		param_group['lr'] = lr


if __name__ == '__main__':
	main()