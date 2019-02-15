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

from util import AverageMeter, accuracy, transform_time
from util import load_pretrained_model, save_checkpoint
from network import define_tsnet

parser = argparse.ArgumentParser(description='deep mutual learning (only two nets)')

# various path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')
parser.add_argument('--net1_init', type=str, required=True, help='initial parameters of net1')
parser.add_argument('--net2_init', type=str, required=True, help='initial parameters of net2')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)

# net and dataset choosen
parser.add_argument('--data_name', type=str, required=True, help='name of dataset')# cifar10/cifar100
parser.add_argument('--net1_name', type=str, required=True, help='name of net1')
parser.add_argument('--net2_name', type=str, required=True, help='name of net2')

# hyperparameter lambda
parser.add_argument('--lambda_dml', type=float, default=1.0)

def main():
	global args
	args = parser.parse_args()
	print(args)

	if not os.path.exists(os.path.join(args.save_root,'checkpoint')):
		os.makedirs(os.path.join(args.save_root,'checkpoint'))

	if args.cuda:
		cudnn.benchmark = True

	print('----------- Network Initialization --------------')
	net1 = define_tsnet(name=args.net1_name, num_class=args.num_class, cuda=args.cuda)
	checkpoint = torch.load(args.net1_init)
	load_pretrained_model(net1, checkpoint['net'])

	net2 = define_tsnet(name=args.net2_name, num_class=args.num_class, cuda=args.cuda)
	checkpoint = torch.load(args.net2_init)
	load_pretrained_model(net2, checkpoint['net'])
	print('-----------------------------------------------')

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
	if args.cuda:
		criterionCls = torch.nn.CrossEntropyLoss().cuda()
		criterionDML = torch.nn.KLDivLoss(reduction='sum').cuda()
	else:
		criterionCls = torch.nn.CrossEntropyLoss()
		criterionDML = torch.nn.KLDivLoss(reduction='sum')

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

	for epoch in range(1, args.epochs+1):
		epoch_start_time = time.time()

		optimizers = {'optimizer1':optimizer1, 'optimizer2':optimizer2}
		adjust_lr(optimizers, epoch)

		# train one epoch
		nets = {'net1':net1, 'net2':net2}
		criterions = {'criterionCls':criterionCls, 'criterionDML':criterionDML}
		train(train_loader, nets, optimizers, criterions, epoch)
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
		save_name = 'dml_r{}_r{}_{:>03}.ckp'.format(args.net1_name[6:], args.net2_name[6:], epoch)
		save_name = os.path.join(args.save_root, 'checkpoint', save_name)
		save_checkpoint({
			'epoch': epoch,
			'net1': net1.state_dict(),
			'net2': net2.state_dict(),
		}, save_name)

def train(train_loader, nets, optimizers, criterions, epoch):
	batch_time  = AverageMeter()
	data_time   = AverageMeter()
	cls1_losses = AverageMeter()
	dml1_losses = AverageMeter()
	cls2_losses = AverageMeter()
	dml2_losses = AverageMeter()
	top11       = AverageMeter()
	top51       = AverageMeter()
	top12       = AverageMeter()
	top52       = AverageMeter()

	net1 = nets['net1']
	net2 = nets['net2']

	criterionCls = criterions['criterionCls']
	criterionDML = criterions['criterionDML']

	optimizer1 = optimizers['optimizer1']
	optimizer2 = optimizers['optimizer2']

	net1.train()
	net2.train()

	end = time.time()
	for idx, (img, target) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)

		if args.cuda:
			img = img.cuda()
			target = target.cuda()

		_, _, _, _, output1 = net1(img)
		_, _, _, _, output2 = net2(img)

		# for net1
		cls1_loss = criterionCls(output1, target)
		dml1_loss = criterionDML(F.log_softmax(output1, dim=1),
								 F.softmax(output2.detach(), dim=1)) / img.size(0)
		dml1_loss = dml1_loss * args.lambda_dml
		net1_loss = cls1_loss + dml1_loss

		prec11, prec51 = accuracy(output1, target, topk=(1,5))
		cls1_losses.update(cls1_loss.item(), img.size(0))
		dml1_losses.update(dml1_loss.item(), img.size(0))
		top11.update(prec11.item(), img.size(0))
		top51.update(prec51.item(), img.size(0))

		# for net2
		cls2_loss = criterionCls(output2, target)
		dml2_loss = criterionDML(F.log_softmax(output2, dim=1),
								 F.softmax(output1.detach(), dim=1)) / img.size(0)
		dml2_loss = dml2_loss * args.lambda_dml
		net2_loss = cls2_loss + dml2_loss

		prec12, prec52 = accuracy(output2, target, topk=(1,5))
		cls2_losses.update(cls2_loss.item(), img.size(0))
		dml2_losses.update(dml2_loss.item(), img.size(0))
		top12.update(prec12.item(), img.size(0))
		top52.update(prec52.item(), img.size(0))

		# update net1 & net2
		optimizer1.zero_grad()
		net1_loss.backward()
		optimizer1.step()

		optimizer2.zero_grad()
		net2_loss.backward()
		optimizer2.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if idx % args.print_freq == 0:
			print('Epoch[{0}]:[{1:03}/{2:03}] '
				  'Time:{batch_time.val:.4f} '
				  'Data:{data_time.val:.4f}  '
				  'Cls1:{cls1_losses.val:.4f}({cls1_losses.avg:.4f})  '
				  'DML1:{dml1_losses.val:.4f}({dml1_losses.avg:.4f})  '
				  'Cls1:{cls2_losses.val:.4f}({cls2_losses.avg:.4f})  '
				  'DML1:{dml2_losses.val:.4f}({dml2_losses.avg:.4f})  '
				  'prec@1_1:{top11.val:.2f}({top11.avg:.2f})  '
				  'prec@5_1:{top51.val:.2f}({top51.avg:.2f})  '
				  'prec@1_2:{top12.val:.2f}({top12.avg:.2f})  '
				  'prec@5_2:{top52.val:.2f}({top52.avg:.2f})'.format(
				  epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time,
				  cls1_losses=cls1_losses, dml1_losses=dml1_losses, top11=top11, top51=top51,
				  cls2_losses=cls2_losses, dml2_losses=dml2_losses, top12=top12, top52=top52))

def test(test_loader, nets, criterions):
	cls1_losses = AverageMeter()
	dml1_losses = AverageMeter()
	cls2_losses = AverageMeter()
	dml2_losses = AverageMeter()
	top11       = AverageMeter()
	top51       = AverageMeter()
	top12       = AverageMeter()
	top52       = AverageMeter()

	net1 = nets['net1']
	net2 = nets['net2']

	criterionCls = criterions['criterionCls']
	criterionDML = criterions['criterionDML']

	net1.eval()
	net2.eval()

	end = time.time()
	for idx, (img, target) in enumerate(test_loader, start=1):
		if args.cuda:
			img = img.cuda()
			target = target.cuda()

		with torch.no_grad():
			_, _, _, _, output1 = net1(img)
			_, _, _, _, output2 = net2(img)

		# for net1
		cls1_loss = criterionCls(output1, target)
		dml1_loss = criterionDML(F.log_softmax(output1, dim=1),
								 F.softmax(output2.detach(), dim=1)) / img.size(0)
		dml1_loss = dml1_loss * args.lambda_dml

		prec11, prec51 = accuracy(output1, target, topk=(1,5))
		cls1_losses.update(cls1_loss.item(), img.size(0))
		dml1_losses.update(dml1_loss.item(), img.size(0))
		top11.update(prec11.item(), img.size(0))
		top51.update(prec51.item(), img.size(0))

		# for net2
		cls2_loss = criterionCls(output2, target)
		dml2_loss = criterionDML(F.log_softmax(output2, dim=1),
								 F.softmax(output1.detach(), dim=1)) / img.size(0)
		dml2_loss = dml2_loss * args.lambda_dml

		prec12, prec52 = accuracy(output2, target, topk=(1,5))
		cls2_losses.update(cls2_loss.item(), img.size(0))
		dml2_losses.update(dml2_loss.item(), img.size(0))
		top12.update(prec12.item(), img.size(0))
		top52.update(prec52.item(), img.size(0))

	f_l  = [cls1_losses.avg, dml1_losses.avg, top11.avg, top51.avg]
	f_l += [cls2_losses.avg, dml2_losses.avg, top12.avg, top52.avg]
	print('Cls1: {:.4f}, DML1: {:.4f}, Prec@1_1: {:.2f}, Prec@5_1: {:.2f}'
		  'Cls2: {:.4f}, DML2: {:.4f}, Prec@1_2: {:.2f}, Prec@5_2: {:.2f}'.format(*f_l))

def adjust_lr(optimizers, epoch):
	scale   = 0.1
	lr_list =  [args.lr] * 100
	lr_list += [args.lr*scale] * 50
	lr_list += [args.lr*scale*scale] * 50

	lr = lr_list[epoch-1]
	print('epoch: {}  lr: {}'.format(epoch, lr))
	for param_group in optimizers['optimizer1'].param_groups:
		param_group['lr'] = lr
	for param_group in optimizers['optimizer2'].param_groups:
		param_group['lr'] = lr

if __name__ == '__main__':
	main()