# Knowledge-Distillation-Zoo
Pytorch implementation of various Knowledge Distillation (KD) methods. 

This repository is a simple reference, mainly focuses on basic knowledge distillation/transfer methods. Thus many tricks and variations, such as step-by-step training, iterative training, ensemble of teachers, ensemble of KD methods, data-free, self-distillation, quantization etc. are not considered. Hope it is useful for your project or research.

I will update this repo regularly with new KD methods. If there some basic methods I missed, please contact with me.

## Lists
  Name | Method | Paper Link | Code Link
  :---- | ----- | :----: | :----:
  Baseline | basic model with softmax loss | — | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/train_base.py)
  Logits   | mimic learning via regressing logits | [paper](http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/logits.py)
  ST       | soft target | [paper](https://arxiv.org/pdf/1503.02531.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/st.py)
  AT       | attention transfer | [paper](https://arxiv.org/pdf/1612.03928.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/at.py)
  Fitnet   | hints for thin deep nets | [paper](https://arxiv.org/pdf/1412.6550.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/fitnet.py)
  NST      | neural selective transfer | [paper](https://arxiv.org/pdf/1707.01219.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/nst.py)
  PKT      | probabilistic knowledge transfer | [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/pkt.py)
  FSP      | flow of solution procedure | [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/fsp.py)
  FT       | factor transfer | [paper](http://papers.nips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/ft.py)
  RKD      | relational knowledge distillation | [paper](https://arxiv.org/pdf/1904.05068.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/rkd.py)
  AB       | activation boundary | [paper](https://arxiv.org/pdf/1811.03233.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/ab.py)
  SP       | similarity preservation | [paper](https://arxiv.org/pdf/1907.09682.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/sp.py)
  Sobolev  | sobolev/jacobian matching | [paper](https://arxiv.org/pdf/1706.04859.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/sobolev.py)
  BSS      | boundary supporting samples | [paper](https://arxiv.org/pdf/1805.05532.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/bss.py)
  CC       | correlation congruence | [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/cc.py)
  LwM      | learning without memorizing | [paper](https://arxiv.org/pdf/1811.08051.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/lwm.py)
  IRG      | instance relationship graph | [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Knowledge_Distillation_via_Instance_Relationship_Graph_CVPR_2019_paper.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/irg.py)
  VID      | variational information distillation | [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/vid.py)
  OFD      | overhaul of feature distillation | [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/ofd.py)
  AFD      | attention feature distillation | [paper](https://openreview.net/pdf?id=ryxyCeHtPB) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/afd.py)
  CRD      | contrastive representation distillation | [paper](https://openreview.net/pdf?id=SkgpBJrtvS) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/crd.py)
  DML      | deep mutual learning | [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf) | [code](https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/dml.py)

- Note, there are some differences between this repository and the original papers：
	- For `AT`: I use the sum of absolute values with power p=2 as the attention.
	- For `Fitnet`: The training procedure is one stage without hint layer.
	- For `NST`: I employ polynomial kernel with d=2 and c=0.
	- For `AB`: Two-stage training, the first 50 epochs for initialization, the second stage only employs CE without ST.
	- For `BSS`: 80% epochs employ CE+BSS loss, the rest 20% only uses CE. In addition, warmup for the first 10 epochs.
	- For `CC`: For consistency, I only consider CC without instance congruence. Gaussian RBF kernel is employed because Bilinear Pool kernel is similar with PKT. I choose P=2 order Taylor of Gaussian RBF kernel. No special sampling strategy.
	- For `LwM`: I employ it after rb2 (middle conv layer) but not rb3 (last conv layer), because the base net is resnet with the end of GAP followed by a classifier. If after rb3, the grad-CAN has the same values across H and W in each channel.
	- For `IRG`: I only use one-to-one mode.
	- For `VID`: I set the hidden channel size to be same with the output channel size and remove BN in μ.
	- For `AFD`: I find the original implementation of attention is unstable, thus replace it with a SE block.
	- For `DML`: Just two nets are employed. Synchronous update to avoid multiple forwards.

## Datasets
- CIFAR10
- CIFAR100

## Networks
- Resnet-20
- Resnet-110

The networks are same with Tabel 6 in [paper](https://arxiv.org/pdf/1512.03385.pdf).

## Training
- Creating `./dataset` directory and downloading CIFAR10/CIFAR100 in it.
- Using the script `example_train_script.sh` to train various KD methods. You can simply specify the hyper-parameters listed in `train_xxx.py` or manually change them.
- The hyper-parameters I used can be found in the [training logs](https://pan.baidu.com/s/1A0-FCggjwnAtCCoSpGsjzA) (code: ezed).
- Some Notes:
	- Sobolev/LwM alone is unstable and may be used in conjunction with other KD methods.
   - BSS may occasionally destroy the training procedure, leading to poor results.
	- If not specified in the original papers, all the methods can be used on the middle feature maps or multiple feature maps are only employed after the last conv layer. It is simple to extend to multiple feature maps.
	- I assume the size (C, H, W) of features between teacher and student are the same. If not, you could employ 1\*1 conv, linear or pooling to rectify them.

## Results
- The trained baseline models are used as teachers. For fair comparison, all the student nets have same initialization with the baseline models.
- The initial models, trained models and training logs are uploaded [here](https://pan.baidu.com/s/1A0-FCggjwnAtCCoSpGsjzA) (code: ezed).
- The trade-off parameter `--lambda_kd` and other hyper-parameters are not chosen carefully. Thus the following results do not reflect which method is better than the others.
- Some relation based methods, e.g. PKT, RKD and CC, have less effectiveness on CIFAR100 dataset. It may be because there are more inter classes but less intra classes in one batch. You could increase the batch size, create memory bank or design advance batch sampling methods.

<table>
   <tr>
      <td>Teacher</td>
      <td>Student</td>
      <td>Name</td>
      <td>CIFAR10</td>
      <td>CIFAR100</td>
   </tr>
   <tr>
      <td>-</td>
      <td>resnet-20</td>
      <td>Baseline</td>
      <td>92.37%</td>
      <td>68.92%</td> 
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>Logits</td>
      <td>93.30%</td>
      <td>70.36%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>ST</td>
      <td>93.12%</td>
      <td>70.27%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>AT</td>
      <td>92.89%</td>
      <td>69.70%</td> 
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>Fitnet</td>
      <td>92.73%</td>
      <td>70.08%</td> 
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>NST</td>
      <td>92.79%</td>
      <td>69.21%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>PKT</td>
      <td>92.50%</td>
      <td>69.25%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>FSP</td>
      <td>92.76%</td>
      <td>69.61%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>FT</td>
      <td>92.98%</td>
      <td>69.90%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>RKD</td>
      <td>92.72%</td>
      <td>69.48%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>AB</td>
      <td>93.04%</td>
      <td>69.96%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>SP</td>
      <td>92.88%</td>
      <td>69.85%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>Sobolev</td>
      <td>92.78%</td>
      <td>69.39%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>BSS</td>
      <td>92.58%</td>
      <td>69.96%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>CC</td>
      <td>93.01%</td>
      <td>69.27%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>LwM</td>
      <td>92.80%</td>
      <td>69.23%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>IRG</td>
      <td>92.77%</td>
      <td>70.37%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>VID</td>
      <td>92.61%</td>
      <td>69.39%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>OFD</td>
      <td>92.82%</td>
      <td>69.93%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>AFD</td>
      <td>92.56%</td>
      <td>69.63%</td>
   </tr>
   <tr>
      <td>resnet-20</td>
      <td>resnet-20</td>
      <td>CRD</td>
      <td>92.96%</td>
      <td>70.33%</td> 
   </tr>
</table>


<table>
   <tr>
      <td>Teacher</td>
      <td>Student</td>
      <td>Name</td>
      <td>CIFAR10</td>
      <td>CIFAR100</td>
   </tr>
   <tr>
      <td>-</td>
      <td>resnet-20</td>
      <td>Baseline</td>
      <td>92.37%</td>
      <td>68.92%</td> 
   </tr>
   <tr>
      <td>-</td>
      <td>resnet-110</td>
      <td>Baseline</td>
      <td>93.86%</td>
      <td>73.15%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>Logits</td>
      <td>92.98%</td>
      <td>69.78%</td> 
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>ST</td>
      <td>92.82%</td>
      <td>70.06%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>AT</td>
      <td>93.21%</td>
      <td>69.28%</td> 
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>Fitnet</td>
      <td>93.04%</td>
      <td>69.81%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>NST</td>
      <td>92.83%</td>
      <td>69.31%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>PKT</td>
      <td>93.01%</td>
      <td>69.31%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>FSP</td>
      <td>92.78%</td>
      <td>69.78%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>FT</td>
      <td>93.01%</td>
      <td>69.49%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>RKD</td>
      <td>93.21%</td>
      <td>69.36%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>AB</td>
      <td>92.96%</td>
      <td>69.41%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>SP</td>
      <td>93.30%</td>
      <td>69.45%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>Sobolev</td>
      <td>92.60%</td>
      <td>69.23%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>BSS</td>
      <td>92.78%</td>
      <td>69.71%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>CC</td>
      <td>92.98%</td>
      <td>69.33%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>LwM</td>
      <td>92.52%</td>
      <td>69.11%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>IRG</td>
      <td>93.13%</td>
      <td>69.36%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>VID</td>
      <td>92.98%</td>
      <td>69.49%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>OFD</td>
      <td>93.13%</td>
      <td>69.81%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>AFD</td>
      <td>92.92%</td>
      <td>69.60%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-20</td>
      <td>CRD</td>
      <td>92.92%</td>
      <td>70.80%</td> 
   </tr>
</table>


<table>
   <tr>
      <td>Teacher</td>
      <td>Student</td>
      <td>Name</td>
      <td>CIFAR10</td>
      <td>CIFAR100</td>
   </tr>
   <tr>
      <td>-</td>
      <td>resnet-110</td>
      <td>Baseline</td>
      <td>93.86%</td>
      <td>73.15%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>Logits</td>
      <td>94.38%</td>
      <td>74.89%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>ST</td>
      <td>94.59%</td>
      <td>74.33%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>AT</td>
      <td>94.42%</td>
      <td>74.64%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>Fitnet</td>
      <td>94.43%</td>
      <td>73.63%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>NST</td>
      <td>94.43%</td>
      <td>73.55%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>PKT</td>
      <td>94.35%</td>
      <td>73.74%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>FSP</td>
      <td>94.39%</td>
      <td>73.59%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>FT</td>
      <td>94.30%</td>
      <td>74.72%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>RKD</td>
      <td>94.39%</td>
      <td>73.78%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>AB</td>
      <td>94.63%</td>
      <td>73.91%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>SP</td>
      <td>94.45%</td>
      <td>74.07%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>Sobolev</td>
      <td>94.26%</td>
      <td>73.14%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>BSS</td>
      <td>94.19%</td>
      <td>73.87%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>CC</td>
      <td>94.49%</td>
      <td>74.43%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>LwM</td>
      <td>94.19%</td>
      <td>73.28%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>IRG</td>
      <td>94.44%</td>
      <td>74.96%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>VID</td>
      <td>94.25%</td>
      <td>73.63%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>OFD</td>
      <td>94.38%</td>
      <td>74.11%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>AFD</td>
      <td>94.44%</td>
      <td>73.90%</td>
   </tr>
   <tr>
      <td>resnet-110</td>
      <td>resnet-110</td>
      <td>CRD</td>
      <td>94.30%</td>
      <td>75.44%</td>
   </tr>
</table>


<table>
   <tr>
      <td>Net1</td>
      <td>Net2</td>
      <td>Name</td>
      <td>CIFAR10</td>
      <td>CIFAR100</td>
   </tr>
   <tr>
      <td>-</td>
      <td>resnet-20</td>
      <td>baseline</td>
      <td>92.37%</td>
      <td>68.92%</td>
   </tr>
   <tr>
      <td>-</td>
      <td>resnet-110</td>
      <td>baseline</td>
      <td>93.86%</td>
      <td>73.15%</td>
   </tr>
   <tr>
      <td>resnet20</td>
      <td>resnet20</td>
      <td>DML</td>
      <td>93.07%/93.37%</td>
      <td>70.39%/70.22%</td>
   </tr>
   <tr>
      <td>resnet110</td>
      <td>resnet20</td>
      <td>DML</td>
      <td>94.45%/92.92%</td>
      <td>74.53%/70.29%</td>
   </tr>
   <tr>
      <td>resnet110</td>
      <td>resnet110</td>
      <td>DML</td>
      <td>94.74%/94.79%</td>
      <td>74.72%/75.55%</td>
   </tr>
</table>

## Todo List
- [ ] KDSVD (now has some bugs)
- [ ] QuEST: Quantized Embedding Space for Transferring Knowledge
- [ ] EEL: Learning an Evolutionary Embedding via Massive Knowledge Distillation
- [ ] OnAdvFD: Feature-map-level Online Adversarial Knowledge Distillation
- [ ] CS-KD: Regularizing Class-wise Predictions via Self-knowledge Distillation
- [ ] PAD: Prime-Aware Adaptive Distillation
- [ ] CD: Channel Distillation: Channel-Wise Attention for Knowledge Distillation
- [ ] DCM: Knowledge Transfer via Dense Cross-Layer Mutual-Distillation

## Requirements
- python 3.7
- pytorch 1.3.1
- torchvision 0.4.2

## Acknowledgements
This repo is partly based on the following repos, thank the authors a lot.
- [HobbitLong/RepDistiller](https://github.com/HobbitLong/RepDistiller)
- [bhheo/BSS_distillation](https://github.com/bhheo/BSS_distillation)
- [clovaai/overhaul-distillation](https://github.com/clovaai/overhaul-distillation)
- [passalis/probabilistic_kt](https://github.com/passalis/probabilistic_kt)
- [lenscloth/RKD](https://github.com/lenscloth/RKD)

If you employ the listed KD methods in your research, please cite the corresponding papers.
