# Knowledge-Distillation-Zoo
Pytorch implementation of various Knowledge Distillation (KD) methods. 

This repository is a simple reference, mainly focuses on basic knowledge distillation/transfer methods. Thus many tricks and variations, such as step-by-step training, iterative training, ensemble of teachers, ensemble of KD methods, data-free, self-distillation, quantization etc. are not considered. Hope it is useful for your project or research.

## Lists
  Name | Method | Paper Link | Code Link
  :---- | ----- | :----: | :----:
  Baseline | basic model with softmax loss | — | [code]()
  Logits   | mimic learning via regressing logits | [paper](http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf) | [code]()
  ST       | soft target | [paper](https://arxiv.org/pdf/1503.02531.pdf) | [code]()
  AT       | attention transfer | [paper](https://arxiv.org/pdf/1612.03928.pdf) | [code]()
  Fitnet   | hints for thin deep nets | [paper](https://arxiv.org/pdf/1412.6550.pdf) | [code]()
  NST      | neural selective transfer | [paper](https://arxiv.org/pdf/1707.01219.pdf) | [code]()
  PKT      | probabilistic knowledge transfer | [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf) | [code]()
  FSP      | flow of solution procedure | [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf) | [code]()
  FT       | factor transfer | [paper](http://papers.nips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer.pdf) | [code]()
  RKD      | relational knowledge distillation | [paper](https://arxiv.org/pdf/1904.05068.pdf) | [code]()
  AB       | activation boundary | [paper](https://arxiv.org/pdf/1811.03233.pdf) | [code]()
  SP       | similarity preservation | [paper](https://arxiv.org/pdf/1907.09682.pdf) | [code]()
  Sobolev  | sobolev/jacobian matching | [paper](https://arxiv.org/pdf/1706.04859.pdf) | [code]()
  BSS      | boundary supporting samples | [paper](https://arxiv.org/pdf/1805.05532.pdf) | [code]()
  CC       | correlation congruence | [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf) | [code]()
  LwM      | learning without memorizing | [paper](https://arxiv.org/pdf/1811.08051.pdf) | [code]()
  IRG      | instance relationship graph | [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Knowledge_Distillation_via_Instance_Relationship_Graph_CVPR_2019_paper.pdf) | [code]()
  VID      | variational information distillation | [paper](https://zpascal.net/cvpr2019/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.pdf) | [code]()
  OFD      | overhaul of feature distillation | [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf) | [code]()
  AFD      | attention feature distillation | [paper](https://openreview.net/pdf?id=ryxyCeHtPB) | [code]()
  CRD      | contrastive representation distillation | [paper](https://openreview.net/pdf?id=SkgpBJrtvS) | [code]()
  DML      | deep mutual learning | [paper](https://zpascal.net/cvpr2018/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf) | [code]()

- Note, there are some differences between this repository and the original papers：
	- For `AT`: I use the sum of absolute values with power p=2 as the attention.
	- For `Fitnet`: The training procedure is one stage without hint layer.
	- For `NST`: I employ polynomial kernel with d=2 and c=0.
	- For `AB`: Two-stage training, the first 50 epochs for initialization, the second stage only employs CE without ST.
	- For `BSS`: 75% epochs employ CE+BSS loss, the rest 25% only uses CE. In addition, warmup for the first 10 epochs.
	- For `CC`: For consistency, I only consider CC without instance congruence. Gaussian RBF kernel is employed because Bilinear Pool kernel is similar with PKT. I choose P=2 order Taylor of Gaussian RBF kernel. No special sampling strategy.
	- For `LwM`: I employ it after rb2 (middle conv layer) but not rb3 (last conv layer), because the base net is resnet with the end of GAP followed by a classifier. If after rb3, the grad-CAN has the same values across H and W in each channel.
	- For `IRG`: I only use one-to-one mode.
	- For `VID`: I set the hidden channel size to be same with the output channel size and remove BN in μ.
	- For `AFD`: I find the original implementation of attention is unstable, thus replace it with a SE block.
	- For `DML`: Just two nets are employed.

## Datasets
- CIFAR10
- CIFAR100

## Networks
- Resnet-20
- Resnet-110

The networks are same with Tabel 6 in [paper](https://arxiv.org/pdf/1512.03385.pdf).

## Training
- Creating `./dataset` directory and downloading CIFAR10/CIFAR100 in it.
- Using the script `example_train_script.sh` to train various KD methods. One can simply specify the hyper-parameters listed in `train_xxx.py` or manually change them.
- The hyper-parameters I used can be found in the [training logs](https://pan.baidu.com/s/1OpNH0E8IcQkiv1tFWsQt_w?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#list/path=%2F).
- Some Notes:
	- Sobolev/LwM alone is unstable and may be used in conjunction with other KD methods.
	- If not specified in the original papers, all the methods can be used on the middle feature maps or multiple feature maps are only employed after the last conv layer. It is simple to extend to multiple feature maps.
	- I assume the size (C, H, W) of features between teacher and student are the same. If not, one could employ 1\*1 conv, linear or pooling to rectify them.

## Results
- The trained baseline models are used as teachers. For fair comparison, all the student nets have same initialization with the baseline models.
- The initial models, trained models and training logs are uploaded [here](https://pan.baidu.com/s/1OpNH0E8IcQkiv1tFWsQt_w?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#list/path=%2F).
- The trade-off parameter `--lambda_kd` and other hyper-parameters are not chosen carefully. Thus the following results do not reflect which method is better than the others.
- Some relation based methods, e.g. PKT, RKD and CC, have less effectiveness on CIFAR100 dataset. It may be because there are more inter classes but less intra classes in one batch. One could increase the batch size, create memory bank or design advance batch sampling methods.

<table>
   <tr>
      <td>Teacher</td>
      <td>Student</td>
      <td>Method</td>
      <td>CIFAR10</td>
      <td>CIFAR100</td>
   </tr>
   <tr>
      <td>-</td>
      <td>resnet-20</td>
      <td>baseline</td>
      <td>92.18%</td>
      <td>68.33%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>logits</td>
      <td>93.01%</td>
      <td>69.87%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>st</td>
      <td>92.54%</td>
      <td>69.92%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>fitnet</td>
      <td>92.48%</td>
      <td>69.05%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>at</td>
      <td>92.58%</td>
      <td>68.56%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>fsp</td>
      <td>92.57%</td>
      <td>69.10%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>nst</td>
      <td>92.35%</td>
      <td>68.35%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>pkt</td>
      <td>92.83%</td>
      <td>68.83%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>ft</td>
      <td>92.92%</td>
      <td>68.86%</td>
   </tr>
</table>

<table>
   <tr>
      <td>Teacher</td>
      <td>Student</td>
      <td>Method</td>
      <td>CIFAR10</td>
      <td>CIFAR100</td>
   </tr>
   <tr>
      <td>-</td>
      <td>resnet-20</td>
      <td>baseline</td>
      <td>92.18%</td>
      <td>68.33%</td>
   </tr>
   <tr>
      <td>-</td>
      <td>resnet-110</td>
      <td>baseline</td>
      <td>94.04%</td>
      <td>72.65%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>logits</td>
      <td>93.33%</td>
      <td>69.94%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>st</td>
      <td>92.82%</td>
      <td>69.45%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>fitnet</td>
      <td>92.55%</td>
      <td>69.68%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>at</td>
      <td>92.84%</td>
      <td>69.05%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>fsp</td>
      <td>92.83%</td>
      <td>69.38%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>nst</td>
      <td>92.51%</td>
      <td>68.41%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>pkt</td>
      <td>92.95%</td>
      <td>69.04%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>ft</td>
      <td>93.20%</td>
      <td>69.45%</td>
   </tr>
</table>

<table>
   <tr>
      <td>Teacher</td>
      <td>Student</td>
      <td>Method</td>
      <td>CIFAR10</td>
      <td>CIFAR100</td>
   </tr>
   <tr>
      <td>-</td>
      <td>resnet-110</td>
      <td>baseline</td>
      <td>94.04%</td>
      <td>72.65%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>logits</td>
      <td>94.48%</td>
      <td>74.72%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>st</td>
      <td>94.30%</td>
      <td>74.29%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>fitnet</td>
      <td>94.58%</td>
      <td>73.21%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>at</td>
      <td>94.34%</td>
      <td>73.81%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>fsp</td>
      <td>94.29%</td>
      <td>73.71%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>nst</td>
      <td>94.27%</td>
      <td>72.84%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>pkt</td>
      <td>94.76%</td>
      <td>73.73%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>ft</td>
      <td>94.46%</td>
      <td>73.41%</td>
   </tr>
</table>

<table>
   <tr>
      <td>Net1</td>
      <td>Net2</td>
      <td>Method</td>
      <td>CIFAR10</td>
      <td>CIFAR100</td>
   </tr>
   <tr>
      <td>-</td>
      <td>resnet-20</td>
      <td>baseline</td>
      <td>92.18%</td>
      <td>68.33%</td>
   </tr>
   <tr>
      <td>-</td>
      <td>resnet-110</td>
      <td>baseline</td>
      <td>94.04%</td>
      <td>72.65%</td>
   </tr>
   <tr>
      <td>resnet20</td>
      <td>resnet20</td>
      <td>dml</td>
      <td>92.99%/92.81%</td>
      <td>70.30%/70.19%</td>
   </tr>
   <tr>
      <td>resnet110</td>
      <td>resnet20</td>
      <td>dml</td>
      <td>94.52%/92.72%</td>
      <td>75.25%/70.26%</td>
   </tr>
   <tr>
      <td>resnet110</td>
      <td>resnet110</td>
      <td>dml</td>
      <td>94.92%/94.46%</td>
      <td>74.70%/74.91%</td>
   </tr>
</table>

## Todo List
- [ ] QuEST

## Requirements
- python 3.7
- pytorch 1.3.1
- torchvision 0.4.2
