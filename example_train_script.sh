# example scripts for running various KD methods
# use cifar10, resnet110 (teacher) and resnet20 (student) as examples

# Baseline
CUDA_VISIBLE_DEVICES=0 python -u train_base.py \
                           --save_root "./results/base/" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --net_name resnet20 \
                           --note base-c10-r20

# Logits
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/logits/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode logits \
                           --lambda_kd 0.1 \
                           --note logits-c10-r110-r20

# SoftTarget
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/st/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode st \
                           --lambda_kd 0.1 \
                           --T 4.0 \
                           --note st-c10-r110-r20

# AT
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/at/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode at \
                           --lambda_kd 1000.0 \
                           --p 2.0 \
                           --note at-c10-r110-r20

# Fitnet
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/fitnet/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode fitnet \
                           --lambda_kd 0.1 \
                           --note fitnet-c10-r110-r20

# NST
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/nst/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode nst \
                           --lambda_kd 10.0 \
                           --note nst-c10-r110-r20

# PKT
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/pkt/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode pkt \
                           --lambda_kd 10000.0 \
                           --note pkt-c10-r110-r20

# FSP
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/fsp/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode fsp \
                           --lambda_kd 1.0 \
                           --note fsp-c10-r110-r20

# FT
CUDA_VISIBLE_DEVICES=0 python -u train_ft.py \
                           --save_root "./results/ft/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --lambda_kd 200.0 \
                           --k 0.5 \
                           --note ft-c10-r110-r20

# RKD
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/rkd/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode rkd \
                           --lambda_kd 1.0 \
                           --w_dist 25.0 \
                           --w_angle 50.0 \
                           --note rkd-c10-r110-r20

# AB
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/ab/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode ab \
                           --lambda_kd 10.0 \
                           --m 2.0 \
                           --note ab-c10-r110-r20

# SP
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/sp/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode sp \
                           --lambda_kd 3000.0 \
                           --note sp-c10-r110-r20

# Sobolev
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/sobolev/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode sobolev \
                           --lambda_kd 20.0 \
                           --note sobolev-c10-r110-r20

# BSS
CUDA_VISIBLE_DEVICES=0 python -u train_bss.py \
                           --save_root "./results/bss/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --lambda_kd 2.0 \
                           --T 3.0 \
                           --attack_size 32 \
                           --note bss-c10-r110-r20

# CC
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/cc/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode cc \
                           --lambda_kd 100.0 \
                           --gamma 0.4 \
                           --P_order 2 \
                           --note cc-c10-r110-r20

# LwM
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/lwm/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode lwm \
                           --lambda_kd 0.4 \
                           --note lwm-c10-r110-r20

# IRG
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/irg/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode irg \
                           --lambda_kd 1.0 \
                           --w_irg_vert 0.1 \
                           --w_irg_edge 5.0 \
                           --w_irg_tran 5.0 \
                           --note irg-c10-r110-r20

# VID
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/vid/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode vid \
                           --lambda_kd 1.0 \
                           --sf 1.0 \
                           --init_var 5.0 \
                           --note vid-c10-r110-r20

# OFD
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/ofd/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode ofd \
                           --lambda_kd 0.5 \
                           --note ofd-c10-r110-r20

# AFD
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/afd/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --kd_mode afd \
                           --lambda_kd 500.0 \
                           --att_f 1.0 \
                           --note afd-c10-r110-r20

# CRD
# lambda_kd=0.2 for CIFAR10, lambda_kd=0.8 for CIFAR100.
CUDA_VISIBLE_DEVICES=0 python -u train_crd.py \
                           --save_root "./results/crd/" \
                           --t_model "./results/base/base-c10-r110/model_best.pth.tar" \
                           --s_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name resnet110 \
                           --s_name resnet20 \
                           --lambda_kd 0.2 \
                           --feat_dim 128 \
                           --nce_n 16384 \
                           --nce_t 0.1 \
                           --nce_mom 0.5 \
                           --mode 'exact' \
                           --note crd-c10-r110-r20

# DML
CUDA_VISIBLE_DEVICES=0 python -u train_dml.py \
                           --save_root "./results/dml/" \
                           --net1_init "./results/base/base-c10-r110/initial_r110.pth.tar" \
                           --net2_init "./results/base/base-c10-r20/initial_r20.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --net1_name resnet110 \
                           --net2_name resnet20 \
                           --lambda_kd 1.0 \
                           --note dml-c10-r110-r20



