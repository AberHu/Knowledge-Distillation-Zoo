# Knowledge-Distillation-Zoo
Pytorch implementation of various Knowledge Distillation methods. 

This repository is a simple reference, thus many tricks, such as step-by-step training, iterative training, ensemble of teachers,  etc. are not considered.

## Lists
  Filename| Method|  Link
  :----| :-----: | :----:    
  [train_baseline.py]() | basic cnn with softmax loss |   —    
  [train_logits.py]()   | mimic learning via regressing logits (logits) | [paper](http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf) 
  [train_st.py]()   | soft targets (st) | [paper](https://arxiv.org/pdf/1503.02531.pdf) 
  [train_fitnet.py]()   | hints for thin deep nets (fitnet) | [paper](https://arxiv.org/pdf/1412.6550.pdf) 
  [train_at.py]()   | attention transfer (at) | [paper](https://arxiv.org/pdf/1612.03928.pdf) 
  [train_fsp.py]()   | flow of solution procedure (fsp) | [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf) 
  [train_nst.py]()   | neural selective transfer (nst) | [paper](https://arxiv.org/pdf/1707.01219.pdf) 
  [train_pkt.py]()   | probabilistic knowledge transfer (pkt) | [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf) 
  [train_ft.py]()   | factor transfer (ft) | [paper](https://arxiv.org/pdf/1802.04977.pdf)
  [train_dml.py]()   | deep mutual learning (dml) | [paper](https://arxiv.org/pdf/1706.00384.pdf)

- Note, there are some differences between this repository and the original paper：
	- For `fitnet`: the training procedure is one stage without hint layer.
	- For `at`: I use the sum of absolute values with power p=2 as the attention.
	- For `nst`: I use squared mmd matching.
	- For `dml`: just two nets are employed.

## Datasets
- CIFAR10
- CIFAR100

## Networks
- Resnet-20
- Resnet-110

The networks are same with Tabel 6 in [paper](https://arxiv.org/pdf/1512.03385.pdf).

## Training
- Creating `./dataset` directory and downloading CIFAR10/CIFAR100 in it.
- Using the train script, simply specifying the parameters listed in  `train_xxx.py`  as a flag or manually changing them.
- The parameters I used can be found in the training [logs](https://pan.baidu.com/s/1OpNH0E8IcQkiv1tFWsQt_w?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#list/path=%2F).
- For `baseline`
```Shell
    python train_baseline.py
			--data_name=cifar10/cifar100 \
			--net_name=resnet20/resnet110 \
			--num_class=10/100
```
- For `logits,st,fitnet,at,fsp,nst,pkt,ft`
```Shell
    python train_xxx.py
		    --s_init=/path/to/your/student_initial_model \
		    --t_model=/path/to/your/teacher_model \
			--data_name=cifar10/cifar100  \
			--t_name=resnet20/resnet110 \
			--s_name=resnet20/resnet110 \
			--num_class=10/100
```
- For `dml`
```Shell
    python train_dml.py
		    --net1_init=/path/to/your/net1_initial_model \
		    --net2_init=/path/to/your/net2_initial_model \
			--data_name=cifar10/cifar100  \
			--net1_name=resnet20/resnet110 \
			--net2_name=resnet20/resnet110 \
			--num_class=10/100
```

## Results
- The trained baseline models are used as teachers. For fair comparison, all the student nets have same initialization with the baseline models.
- The initial models, trained models and training logs are uploaded [here](https://pan.baidu.com/s/1OpNH0E8IcQkiv1tFWsQt_w?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#list/path=%2F).
- The loss trade-off parameters `--lambda_xxx` are not chosen carefully. Thus the following results do not reflect which method is better than the others.

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
   <tr>
      <td></td>
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
   <tr>
      <td></td>
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
   <tr>
      <td></td>
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


## Requirements
- python 2.7
- pytorch 1.0.0
- torchvision 0.2.1
