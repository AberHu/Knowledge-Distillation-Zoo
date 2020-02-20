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
from network import define_tsnet
from kd_losses import *

parser = argparse.ArgumentParser(description='train kd')

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
parser.add_argument('--kd_mode', type=str, required=True, help='mode of kd, which can be:'
															   'logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/'
															   'sp/sobolev/cc/lwm/irg/vid/ofd/afd')
parser.add_argument('--lambda_kd', type=float, default=1.0, help='trade-off parameter for kd loss')
parser.add_argument('--T', type=float, default=4.0, help='temperature for ST')
parser.add_argument('--p', type=float, default=2.0, help='power for AT')
parser.add_argument('--w_dist', type=float, default=25.0, help='weight for RKD distance')
parser.add_argument('--w_angle', type=float, default=50.0, help='weight for RKD angle')
parser.add_argument('--m', type=float, default=2.0, help='margin for AB')
parser.add_argument('--gamma', type=float, default=0.4, help='gamma in Gaussian RBF for CC')
parser.add_argument('--P_order', type=int, default=2, help='P-order Taylor series of Gaussian RBF for CC')
parser.add_argument('--w_irg_vert', type=float, default=0.1, help='weight for IRG vertex')
parser.add_argument('--w_irg_edge', type=float, default=5.0, help='weight for IRG edge')
parser.add_argument('--w_irg_tran', type=float, default=5.0, help='weight for IRG transformation')
parser.add_argument('--sf', type=float, default=1.0, help='scale factor for VID, i.e. mid_channels = sf * out_channels')
parser.add_argument('--init_var', type=float, default=5.0, help='initial variance for VID')
parser.add_argument('--att_f', type=float, default=1.0, help='attention factor of mid_channels for AFD')


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

	# define loss functions
	if args.kd_mode == 'logits':
		criterionKD = Logits()
	elif args.kd_mode == 'st':
		criterionKD = SoftTarget(args.T)
	elif args.kd_mode == 'at':
		criterionKD = AT(args.p)
	elif args.kd_mode == 'fitnet':
		criterionKD = Hint()
	elif args.kd_mode == 'nst':
		criterionKD = NST()
	elif args.kd_mode == 'pkt':
		criterionKD = PKTCosSim()
	elif args.kd_mode == 'fsp':
		criterionKD = FSP()
	elif args.kd_mode == 'rkd':
		criterionKD = RKD(args.w_dist, args.w_angle)
	elif args.kd_mode == 'ab':
		criterionKD = AB(args.m)
	elif args.kd_mode == 'sp':
		criterionKD = SP()
	elif args.kd_mode == 'sobolev':
		criterionKD = Sobolev()
	elif args.kd_mode == 'cc':
		criterionKD = CC(args.gamma, args.P_order)
	elif args.kd_mode == 'lwm':
		criterionKD = LwM()
	elif args.kd_mode == 'irg':
		criterionKD = IRG(args.w_irg_vert, args.w_irg_edge, args.w_irg_tran)
	elif args.kd_mode == 'vid':
		s_channels  = snet.module.get_channel_num()[1:4]
		t_channels  = tnet.module.get_channel_num()[1:4]
		criterionKD = []
		for s_c, t_c in zip(s_channels, t_channels):
			criterionKD.append(VID(s_c, int(args.sf*t_c), t_c, args.init_var))
		criterionKD = [c.cuda() for c in criterionKD] if args.cuda else criterionKD
		criterionKD = [None] + criterionKD # None is a placeholder
	elif args.kd_mode == 'ofd':
		s_channels  = snet.module.get_channel_num()[1:4]
		t_channels  = tnet.module.get_channel_num()[1:4]
		criterionKD = []
		for s_c, t_c in zip(s_channels, t_channels):
			criterionKD.append(OFD(s_c, t_c).cuda() if args.cuda else OFD(s_c, t_c))
		criterionKD = [None] + criterionKD # None is a placeholder
	elif args.kd_mode == 'afd':
		# t_channels is same with s_channels
		s_channels  = snet.module.get_channel_num()[1:4]
		t_channels  = tnet.module.get_channel_num()[1:4]
		criterionKD = []
		for t_c in t_channels:
			criterionKD.append(AFD(t_c, args.att_f).cuda() if args.cuda else AFD(t_c, args.att_f))
		criterionKD = [None] + criterionKD # None is a placeholder
		# # t_chws is same with s_chws
		# s_chws = snet.module.get_chw_num()[1:4]
		# t_chws = tnet.module.get_chw_num()[1:4]
		# criterionKD = []
		# for t_chw in t_chws:
		# 	criterionKD.append(AFD(t_chw).cuda() if args.cuda else AFD(t_chw))
		# criterionKD = [None] + criterionKD # None is a placeholder
	else:
		raise Exception('Invalid kd mode...')
	if args.cuda:
		criterionCls = torch.nn.CrossEntropyLoss().cuda()
	else:
		criterionCls = torch.nn.CrossEntropyLoss()

	# initialize optimizer
	if args.kd_mode in ['vid', 'ofd', 'afd']:
		optimizer = torch.optim.SGD(chain(snet.parameters(), 
										  *[c.parameters() for c in criterionKD[1:]]),
									lr = args.lr, 
									momentum = args.momentum, 
									weight_decay = args.weight_decay,
									nesterov = True)
	else:
		optimizer = torch.optim.SGD(snet.parameters(),
									lr = args.lr, 
									momentum = args.momentum, 
									weight_decay = args.weight_decay,
									nesterov = True)

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
	nets = {'snet':snet, 'tnet':tnet}
	criterions = {'criterionCls':criterionCls, 'criterionKD':criterionKD}

	# first initilizing the student nets
	if args.kd_mode in ['fsp', 'ab']:
		logging.info('The first stage, student initialization......')
		train_init(train_loader, nets, optimizer, criterions, 50)
		args.lambda_kd = 0.0
		logging.info('The second stage, softmax training......')

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


def train_init(train_loader, nets, optimizer, criterions, total_epoch):
	snet = nets['snet']
	tnet = nets['tnet']

	criterionCls = criterions['criterionCls']
	criterionKD  = criterions['criterionKD']

	snet.train()

	for epoch in range(1, total_epoch+1):
		adjust_lr_init(optimizer, epoch)

		batch_time = AverageMeter()
		data_time  = AverageMeter()
		cls_losses = AverageMeter()
		kd_losses  = AverageMeter()
		top1       = AverageMeter()
		top5       = AverageMeter()

		epoch_start_time = time.time()
		end = time.time()
		for i, (img, target) in enumerate(train_loader, start=1):
			data_time.update(time.time() - end)

			if args.cuda:
				img = img.cuda(non_blocking=True)
				target = target.cuda(non_blocking=True)

			stem_s, rb1_s, rb2_s, rb3_s, feat_s, out_s = snet(img)
			stem_t, rb1_t, rb2_t, rb3_t, feat_t, out_t = tnet(img)

			cls_loss = criterionCls(out_s, target) * 0.0
			if args.kd_mode in ['fsp']:
				kd_loss = (criterionKD(stem_s[1], rb1_s[1], stem_t[1].detach(), rb1_t[1].detach()) +
						   criterionKD(rb1_s[1],  rb2_s[1], rb1_t[1].detach(),  rb2_t[1].detach()) +
						   criterionKD(rb2_s[1],  rb3_s[1], rb2_t[1].detach(),  rb3_t[1].detach())) / 3.0 * args.lambda_kd
			elif args.kd_mode in ['ab']:
				kd_loss = (criterionKD(rb1_s[0], rb1_t[0].detach()) +
						   criterionKD(rb2_s[0], rb2_t[0].detach()) +
						   criterionKD(rb3_s[0], rb3_t[0].detach())) / 3.0 * args.lambda_kd
			else:
				raise Exception('Invalid kd mode...')
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

		epoch_duration = time.time() - epoch_start_time
		logging.info('Epoch time: {}s'.format(int(epoch_duration)))


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
	if args.kd_mode in ['vid', 'ofd']:
		for i in range(1,4):
			criterionKD[i].train()

	end = time.time()
	for i, (img, target) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)

		if args.cuda:
			img = img.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

		if args.kd_mode in ['sobolev', 'lwm']:
			img.requires_grad = True

		stem_s, rb1_s, rb2_s, rb3_s, feat_s, out_s = snet(img)
		stem_t, rb1_t, rb2_t, rb3_t, feat_t, out_t = tnet(img)

		cls_loss = criterionCls(out_s, target)
		if args.kd_mode in ['logits', 'st']:
			kd_loss = criterionKD(out_s, out_t.detach()) * args.lambda_kd
		elif args.kd_mode in ['fitnet', 'nst']:
			kd_loss = criterionKD(rb3_s[1], rb3_t[1].detach()) * args.lambda_kd
		elif args.kd_mode in ['at', 'sp']:
			kd_loss = (criterionKD(rb1_s[1], rb1_t[1].detach()) +
					   criterionKD(rb2_s[1], rb2_t[1].detach()) +
					   criterionKD(rb3_s[1], rb3_t[1].detach())) / 3.0 * args.lambda_kd
		elif args.kd_mode in ['pkt', 'rkd', 'cc']:
			kd_loss = criterionKD(feat_s, feat_t.detach()) * args.lambda_kd
		elif args.kd_mode in ['fsp']:
			kd_loss = (criterionKD(stem_s[1], rb1_s[1], stem_t[1].detach(), rb1_t[1].detach()) +
					   criterionKD(rb1_s[1],  rb2_s[1], rb1_t[1].detach(),  rb2_t[1].detach()) +
					   criterionKD(rb2_s[1],  rb3_s[1], rb2_t[1].detach(),  rb3_t[1].detach())) / 3.0 * args.lambda_kd
		elif args.kd_mode in ['ab']:
			kd_loss = (criterionKD(rb1_s[0], rb1_t[0].detach()) +
					   criterionKD(rb2_s[0], rb2_t[0].detach()) +
					   criterionKD(rb3_s[0], rb3_t[0].detach())) / 3.0 * args.lambda_kd
		elif args.kd_mode in ['sobolev']:
			kd_loss = criterionKD(out_s, out_t, img, target) * args.lambda_kd
		elif args.kd_mode in ['lwm']:
			kd_loss = criterionKD(out_s, rb2_s[1], out_t, rb2_t[1], target) * args.lambda_kd
		elif args.kd_mode in ['irg']:
			kd_loss = criterionKD([rb2_s[1], rb3_s[1], feat_s, out_s],
								  [rb2_t[1].detach(),
								   rb3_t[1].detach(),
								   feat_t.detach(), 
								   out_t.detach()]) * args.lambda_kd
		elif args.kd_mode in ['vid', 'afd']:
			kd_loss = (criterionKD[1](rb1_s[1], rb1_t[1].detach()) +
					   criterionKD[2](rb2_s[1], rb2_t[1].detach()) +
					   criterionKD[3](rb3_s[1], rb3_t[1].detach())) / 3.0 * args.lambda_kd
		elif args.kd_mode in ['ofd']:
			kd_loss = (criterionKD[1](rb1_s[0], rb1_t[0].detach()) +
					   criterionKD[2](rb2_s[0], rb2_t[0].detach()) +
					   criterionKD[3](rb3_s[0], rb3_t[0].detach())) / 3.0 * args.lambda_kd
		else:
			raise Exception('Invalid kd mode...')
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
	kd_losses  = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	snet = nets['snet']
	tnet = nets['tnet']

	criterionCls = criterions['criterionCls']
	criterionKD  = criterions['criterionKD']

	snet.eval()
	if args.kd_mode in ['vid', 'ofd']:
		for i in range(1,4):
			criterionKD[i].eval()

	end = time.time()
	for i, (img, target) in enumerate(test_loader, start=1):
		if args.cuda:
			img = img.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

		if args.kd_mode in ['sobolev', 'lwm']:
			img.requires_grad = True
			stem_s, rb1_s, rb2_s, rb3_s, feat_s, out_s = snet(img)
			stem_t, rb1_t, rb2_t, rb3_t, feat_t, out_t = tnet(img)
		else:
			with torch.no_grad():
				stem_s, rb1_s, rb2_s, rb3_s, feat_s, out_s = snet(img)
				stem_t, rb1_t, rb2_t, rb3_t, feat_t, out_t = tnet(img)

		cls_loss = criterionCls(out_s, target)
		if args.kd_mode in ['logits', 'st']:
			kd_loss  = criterionKD(out_s, out_t.detach()) * args.lambda_kd
		elif args.kd_mode in ['fitnet', 'nst']:
			kd_loss = criterionKD(rb3_s[1], rb3_t[1].detach()) * args.lambda_kd
		elif args.kd_mode in ['at', 'sp']:
			kd_loss = (criterionKD(rb1_s[1], rb1_t[1].detach()) +
					   criterionKD(rb2_s[1], rb2_t[1].detach()) +
					   criterionKD(rb3_s[1], rb3_t[1].detach())) / 3.0 * args.lambda_kd
		elif args.kd_mode in ['pkt', 'rkd', 'cc']:
			kd_loss = criterionKD(feat_s, feat_t.detach()) * args.lambda_kd
		elif args.kd_mode in ['fsp']:
			kd_loss = (criterionKD(stem_s[1], rb1_s[1], stem_t[1].detach(), rb1_t[1].detach()) +
					   criterionKD(rb1_s[1],  rb2_s[1], rb1_t[1].detach(),  rb2_t[1].detach()) +
					   criterionKD(rb2_s[1],  rb3_s[1], rb2_t[1].detach(),  rb3_t[1].detach())) / 3.0 * args.lambda_kd
		elif args.kd_mode in ['ab']:
			kd_loss = (criterionKD(rb1_s[0], rb1_t[0].detach()) +
					   criterionKD(rb2_s[0], rb2_t[0].detach()) +
					   criterionKD(rb3_s[0], rb3_t[0].detach())) / 3.0 * args.lambda_kd
		elif args.kd_mode in ['sobolev']:
			kd_loss = criterionKD(out_s, out_t, img, target) * args.lambda_kd
		elif args.kd_mode in ['lwm']:
			kd_loss = criterionKD(out_s, rb2_s[1], out_t, rb2_t[1], target) * args.lambda_kd
		elif args.kd_mode in ['irg']:
			kd_loss = criterionKD([rb2_s[1], rb3_s[1], feat_s, out_s],
								  [rb2_t[1].detach(),
								   rb3_t[1].detach(),
								   feat_t.detach(), 
								   out_t.detach()]) * args.lambda_kd
		elif args.kd_mode in ['vid', 'afd']:
			kd_loss = (criterionKD[1](rb1_s[1], rb1_t[1].detach()) +
					   criterionKD[2](rb2_s[1], rb2_t[1].detach()) +
					   criterionKD[3](rb3_s[1], rb3_t[1].detach())) / 3.0 * args.lambda_kd
		elif args.kd_mode in ['ofd']:
			kd_loss = (criterionKD[1](rb1_s[0], rb1_t[0].detach()) +
					   criterionKD[2](rb2_s[0], rb2_t[0].detach()) +
					   criterionKD[3](rb3_s[0], rb3_t[0].detach())) / 3.0 * args.lambda_kd
		else:
			raise Exception('Invalid kd mode...')

		prec1, prec5 = accuracy(out_s, target, topk=(1,5))
		cls_losses.update(cls_loss.item(), img.size(0))
		kd_losses.update(kd_loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

	f_l = [cls_losses.avg, kd_losses.avg, top1.avg, top5.avg]
	logging.info('Cls: {:.4f}, KD: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

	return top1.avg, top5.avg


def adjust_lr_init(optimizer, epoch):
	scale   = 0.1
	lr_list = [args.lr*scale] * 30
	lr_list += [args.lr*scale*scale] * 10
	lr_list += [args.lr*scale*scale*scale] * 10

	lr = lr_list[epoch-1]
	logging.info('Epoch: {}  lr: {:.4f}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


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