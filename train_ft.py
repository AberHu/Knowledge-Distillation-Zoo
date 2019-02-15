from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst

import argparse
import os
import time
from itertools import chain

from util import AverageMeter, accuracy, transform_time
from util import load_pretrained_model, save_checkpoint
from network import define_tsnet, define_paraphraser, define_translator

parser = argparse.ArgumentParser(description='factor transfer')

# various path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')
parser.add_argument('--s_init', type=str, required=True, help='initial parameters of student model')
parser.add_argument('--t_model', type=str, required=True, help='path name of teacher model')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate of student net')
parser.add_argument('--lr_para', type=float, default=0.01, help='initial learning rate of paraphraser')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)

# net and dataset choosen
parser.add_argument('--data_name', type=str, required=True, help='name of dataset')# cifar10/cifar100
parser.add_argument('--t_name', type=str, required=True, help='name of teacher')
parser.add_argument('--s_name', type=str, required=True, help='name of student')

# hyperparameter lambda
parser.add_argument('--lambda_ft', type=float, default=100.0)
parser.add_argument('--k', type=float, default=0.5)

def main():
	global args
	args = parser.parse_args()
	print(args)

	if not os.path.exists(os.path.join(args.save_root,'checkpoint')):
		os.makedirs(os.path.join(args.save_root,'checkpoint'))

	if args.cuda:
		cudnn.benchmark = True

	print('----------- Network Initialization --------------')
	snet = define_tsnet(name=args.s_name, num_class=args.num_class, cuda=args.cuda)
	checkpoint = torch.load(args.s_init)
	load_pretrained_model(snet, checkpoint['net'])

	tnet = define_tsnet(name=args.t_name, num_class=args.num_class, cuda=args.cuda)
	checkpoint = torch.load(args.t_model)
	load_pretrained_model(tnet, checkpoint['net'])
	tnet.eval()
	for param in tnet.parameters():
		param.requires_grad = False

	paraphraser = define_paraphraser(k=args.k, cuda=args.cuda)
	translator = define_translator(k=args.k, cuda=args.cuda)
	print('-----------------------------------------------')

	# initialize optimizer
	optimizer_para = torch.optim.SGD(paraphraser.parameters(),
									 lr = args.lr_para, 
									 momentum = args.momentum, 
									 weight_decay = args.weight_decay,
									 nesterov = True)
	optimizer = torch.optim.SGD(chain(snet.parameters(),translator.parameters()),
								lr = args.lr, 
								momentum = args.momentum, 
								weight_decay = args.weight_decay,
								nesterov = True)

	# define loss functions
	if args.cuda:
		criterionCls  = torch.nn.CrossEntropyLoss().cuda()
		criterionFT   = torch.nn.L1Loss().cuda()
		criterionPara = torch.nn.MSELoss().cuda()
	else:
		criterionCls  = torch.nn.CrossEntropyLoss()
		criterionFT   = torch.nn.L1Loss()
		criterionPara = torch.nn.MSELoss()

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
		raise Exception('invalid dataset name...')

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

	# train paraphraser for 100 epochs
	print('training paraphraser for 100 epochs')
	for epoch in range(1, 101):
		epoch_start_time = time.time()

		adjust_lr_para(optimizer_para, epoch)

		# train one epoch
		nets = {'tnet':tnet, 'paraphraser':paraphraser}
		train_para(train_loader, nets, optimizer_para, criterionPara, epoch)
		epoch_time = time.time() - epoch_start_time
		print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))

	paraphraser.eval()
	for param in paraphraser.parameters():
		param.requires_grad = False

	print('training student network')
	for epoch in range(1, args.epochs+1):
		epoch_start_time = time.time()

		adjust_lr(optimizer, epoch)

		# train one epoch
		nets = {'snet':snet, 'tnet':tnet, 'paraphraser':paraphraser, 'translator':translator}
		criterions = {'criterionCls':criterionCls, 'criterionFT':criterionFT}
		train(train_loader, nets, optimizer, criterions, epoch)
		epoch_time = time.time() - epoch_start_time
		print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))

		# evaluate on testing set
		print('testing the models......')
		test_start_time = time.time()
		test(test_loader, nets, criterions)
		test_time = time.time() - test_start_time
		print('testing time is {:02}h{:02}m{:02}s'.format(*transform_time(test_time)))

		# save model
		print('saving models......')
		save_name = 'ft_r{}_r{}_{:>03}.ckp'.format(args.t_name[6:], args.s_name[6:], epoch)
		save_name = os.path.join(args.save_root, 'checkpoint', save_name)
		if epoch == 1:
			save_checkpoint({
				'epoch': epoch,
				'snet': snet.state_dict(),
				'tnet': tnet.state_dict(),
			}, save_name)
		else:
			save_checkpoint({
				'epoch': epoch,
				'snet': snet.state_dict(),
			}, save_name)

def train_para(train_loader, nets, optimizer_para, criterionPara, epoch):
	batch_time  = AverageMeter()
	data_time   = AverageMeter()
	para_losses = AverageMeter()

	tnet        = nets['tnet']
	paraphraser = nets['paraphraser']

	paraphraser.train()

	end = time.time()
	for idx, (img, _) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)

		if args.cuda:
			img = img.cuda()

		_, _, _, rb3, _ = tnet(img)
		_, rb3_rec = paraphraser(rb3.detach())

		para_loss = criterionPara(rb3_rec, rb3.detach())
		para_losses.update(para_loss.item(), img.size(0))

		optimizer_para.zero_grad()
		para_loss.backward()
		optimizer_para.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if idx % args.print_freq == 0:
			print('Epoch[{0}]:[{1:03}/{2:03}] '
				  'Time:{batch_time.val:.4f} '
				  'Data:{data_time.val:.4f}  '
				  'Para:{para_losses.val:.4f}({para_losses.avg:.4f})'.format(
				  epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time,
				  para_losses=para_losses))

def train(train_loader, nets, optimizer, criterions, epoch):
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	cls_losses = AverageMeter()
	ft_losses  = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	snet = nets['snet']
	tnet = nets['tnet']
	paraphraser = nets['paraphraser']
	translator  = nets['translator']

	criterionCls = criterions['criterionCls']
	criterionFT  = criterions['criterionFT']

	snet.train()
	translator.train()

	end = time.time()
	for idx, (img, target) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)

		if args.cuda:
			img = img.cuda()
			target = target.cuda()

		_, _, _, rb3_s, output_s = snet(img)
		_, _, _, rb3_t, _        = tnet(img)

		factor_s    = translator(rb3_s)
		factor_s    = normalize(factor_s)
		factor_t, _ = paraphraser(rb3_t)
		factor_t    = normalize(factor_t)

		cls_loss = criterionCls(output_s, target)
		ft_loss  = criterionFT(factor_s, factor_t.detach()) * args.lambda_ft
		loss = cls_loss + ft_loss

		prec1, prec5 = accuracy(output_s, target, topk=(1,5))
		cls_losses.update(cls_loss.item(), img.size(0))
		ft_losses.update(ft_loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if idx % args.print_freq == 0:
			print('Epoch[{0}]:[{1:03}/{2:03}] '
				  'Time:{batch_time.val:.4f} '
				  'Data:{data_time.val:.4f}  '
				  'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
				  'FT:{ft_losses.val:.4f}({ft_losses.avg:.4f})  '
				  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
				  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
				  epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time,
				  cls_losses=cls_losses, ft_losses=ft_losses, top1=top1, top5=top5))

def test(test_loader, nets, criterions):
	cls_losses = AverageMeter()
	ft_losses  = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	snet = nets['snet']
	tnet = nets['tnet']
	paraphraser = nets['paraphraser']
	translator  = nets['translator']

	criterionCls = criterions['criterionCls']
	criterionFT  = criterions['criterionFT']

	snet.eval()
	translator.eval()

	end = time.time()
	for idx, (img, target) in enumerate(test_loader, start=1):
		if args.cuda:
			img = img.cuda()
			target = target.cuda()

		with torch.no_grad():
			_, _, _, rb3_s, output_s = snet(img)
			_, _, _, rb3_t, _        = tnet(img)

			factor_s    = translator(rb3_s)
			factor_s    = normalize(factor_s)
			factor_t, _ = paraphraser(rb3_t)
			factor_t    = normalize(factor_t)

		cls_loss = criterionCls(output_s, target)
		ft_loss  = criterionFT(factor_s, factor_t.detach()) * args.lambda_ft

		prec1, prec5 = accuracy(output_s, target, topk=(1,5))
		cls_losses.update(cls_loss.item(), img.size(0))
		ft_losses.update(ft_loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

	f_l = [cls_losses.avg, ft_losses.avg, top1.avg, top5.avg]
	print('Cls: {:.4f}, FT: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

def adjust_lr(optimizer, epoch):
	scale   = 0.1
	lr_list =  [args.lr] * 100
	lr_list += [args.lr*scale] * 50
	lr_list += [args.lr*scale*scale] * 50

	lr = lr_list[epoch-1]
	print('epoch: {}  lr: {}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def adjust_lr_para(optimizer_para, epoch):
	scale   = 0.1
	lr_para_list =  [args.lr_para] * 50
	lr_para_list += [args.lr_para*scale] * 25
	lr_para_list += [args.lr_para*scale*scale] * 25

	lr_para = lr_para_list[epoch-1]
	print('epoch: {}  lr_para: {}'.format(epoch, lr_para))
	for param_group in optimizer_para.param_groups:
		param_group['lr'] = lr_para

def normalize(factor, eps=1e-5):
	norm = torch.norm(factor.view(factor.size(0),-1), dim=1)
	norm = norm.view(norm.size(0), 1, 1, 1)
	norm_factor = torch.div(factor, norm+eps)

	return norm_factor

if __name__ == '__main__':
	main()