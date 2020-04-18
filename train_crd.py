from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst

from utils import AverageMeter, accuracy, transform_time
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB
from dataset import CIFAR10IdxSample, CIFAR100IdxSample
from network import define_tsnet
from kd_losses import CRD


parser = argparse.ArgumentParser(description='contrastive representation distillation')

# various path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')
parser.add_argument('--s_init', type=str, required=True, help='initial parameters of student model')
parser.add_argument('--t_model', type=str, required=True, help='path name of teacher model')

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
parser.add_argument('--t_name', type=str, required=True, help='name of teacher')    # resnet20/resnet110
parser.add_argument('--s_name', type=str, required=True, help='name of student')    # resnet20/resnet110

# hyperparameter
parser.add_argument('--lambda_kd', type=float, default=0.2, help='trade-off parameter for kd loss')
parser.add_argument('--feat_dim', type=int, default=128, help='dimension of the projection space')
parser.add_argument('--nce_n', type=int, default=16384, help='number of negatives paired with each positive')
parser.add_argument('--nce_t', type=float, default=0.1, help='temperature parameter')
parser.add_argument('--nce_mom', type=float, default=0.5, help='momentum for non-parametric updates')
parser.add_argument('--mode', type=str, default='exact', choices=['exact', 'relax'])


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
	snet = define_tsnet(name=args.s_name, num_class=args.num_class, cuda=args.cuda)
	checkpoint = torch.load(args.s_init)
	load_pretrained_model(snet, checkpoint['net'])
	logging.info('Student: %s', snet)
	logging.info('Student param size = %fMB', count_parameters_in_MB(snet))

	tnet = define_tsnet(name=args.t_name, num_class=args.num_class, cuda=args.cuda)
	checkpoint = torch.load(args.t_model)
	load_pretrained_model(tnet, checkpoint['net'])
	tnet.eval()
	for param in tnet.parameters():
		param.requires_grad = False
	logging.info('Teacher: %s', tnet)
	logging.info('Teacher param size = %fMB', count_parameters_in_MB(tnet))
	logging.info('-----------------------------------------------')

	# define transforms
	if args.data_name == 'cifar10':
		train_dataset = CIFAR10IdxSample
		test_dataset  = dst.CIFAR10
		mean = (0.4914, 0.4822, 0.4465)
		std  = (0.2470, 0.2435, 0.2616)
	elif args.data_name == 'cifar100':
		train_dataset = CIFAR100IdxSample
		test_dataset  = dst.CIFAR100
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
			train_dataset(root      = args.img_root,
						  transform = train_transform,
						  train     = True,
						  download  = True,
						  n         = args.nce_n,
						  mode      = args.mode),
			batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(
			test_dataset(root      = args.img_root,
						 transform = test_transform,
						 train     = False,
						 download  = True),
			batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

	# define loss functions
	s_dim = snet.module.get_channel_num()[-2]
	t_dim = tnet.module.get_channel_num()[-2]
	if args.cuda:
		criterionCls = torch.nn.CrossEntropyLoss().cuda()
		criterionKD  = CRD(s_dim, t_dim, args.feat_dim, args.nce_n,
						   args.nce_t, args.nce_mom, len(train_loader.dataset)).cuda()
	else:
		criterionCls = torch.nn.CrossEntropyLoss()
		criterionKD  = CRD(s_dim, t_dim, args.feat_dim, args.nce_n,
						   args.nce_t, args.nce_mom, len(train_loader.dataset))

	# initialize optimizer
	optimizer = torch.optim.SGD(chain(snet.parameters(), 
									  criterionKD.embed_t.parameters(),
									  criterionKD.embed_s.parameters()),
								lr = args.lr, 
								momentum = args.momentum, 
								weight_decay = args.weight_decay,
								nesterov = True)
	
	# warp nets and criterions for train and test
	nets = {'snet':snet, 'tnet':tnet}
	criterions = {'criterionCls':criterionCls, 'criterionKD':criterionKD}

	best_top1 = 0
	best_top5 = 0
	for epoch in range(1, args.epochs+1):
		adjust_lr(optimizer, epoch)

		# train one epoch
		epoch_start_time = time.time()
		train(train_loader, nets, optimizer, criterions, epoch)

		# evaluate on testing set
		logging.info('Testing the models......')
		test_top1, test_top5 = test(test_loader, nets, criterions, epoch)

		epoch_duration = time.time() - epoch_start_time
		logging.info('Epoch time: {}s'.format(int(epoch_duration)))

		# save model
		is_best = False
		if test_top1 > best_top1:
			best_top1 = test_top1
			best_top5 = test_top5
			is_best = True
		logging.info('Saving models......')
		save_checkpoint({
			'epoch': epoch,
			'snet': snet.state_dict(),
			'tnet': tnet.state_dict(),
			'prec@1': test_top1,
			'prec@5': test_top5,
		}, is_best, args.save_root)


def train(train_loader, nets, optimizer, criterions, epoch):
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	cls_losses = AverageMeter()
	kd_losses  = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	snet = nets['snet']
	tnet = nets['tnet']

	criterionCls = criterions['criterionCls']
	criterionKD  = criterions['criterionKD']

	snet.train()
	criterionKD.embed_s.train()
	criterionKD.embed_t.train()

	end = time.time()
	for i, (img, target, idx, sample_idx) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)

		if args.cuda:
			img = img.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)
			idx = idx.cuda(non_blocking=True)
			sample_idx = sample_idx.cuda(non_blocking=True)

		_, _, _, _, feat_s, out_s = snet(img)
		_, _, _, _, feat_t, out_t = tnet(img)

		cls_loss = criterionCls(out_s, target)
		kd_loss  = criterionKD(feat_s, feat_t, idx, sample_idx) * args.lambda_kd
		loss = cls_loss + kd_loss

		prec1, prec5 = accuracy(out_s, target, topk=(1,5))
		cls_losses.update(cls_loss.item(), img.size(0))
		kd_losses.update(kd_loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
					   'Time:{batch_time.val:.4f} '
					   'Data:{data_time.val:.4f}  '
					   'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
					   'KD:{kd_losses.val:.4f}({kd_losses.avg:.4f})  '
					   'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
					   'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
					   epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
					   cls_losses=cls_losses, kd_losses=kd_losses, top1=top1, top5=top5))
			logging.info(log_str)


def test(test_loader, nets, criterions, epoch):
	cls_losses = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	snet = nets['snet']

	criterionCls = criterions['criterionCls']

	snet.eval()

	end = time.time()
	for i, (img, target) in enumerate(test_loader, start=1):
		if args.cuda:
			img = img.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

		with torch.no_grad():
			_, _, _, _, _, out_s = snet(img)

		cls_loss = criterionCls(out_s, target)

		prec1, prec5 = accuracy(out_s, target, topk=(1,5))
		cls_losses.update(cls_loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

	f_l = [cls_losses.avg, top1.avg, top5.avg]
	logging.info('Cls: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

	return top1.avg, top5.avg


def adjust_lr(optimizer, epoch):
	scale   = 0.1
	lr_list =  [args.lr] * 100
	lr_list += [args.lr*scale] * 50
	lr_list += [args.lr*scale*scale] * 50

	lr = lr_list[epoch-1]
	logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


if __name__ == '__main__':
	main()
