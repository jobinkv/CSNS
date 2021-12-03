#!/bin/bash
#SBATCH -A research
#SBATCH -n 1
#	#SBATCH --reservation ndq
#	#SBATCH -C 2080ti
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END
echo " deeplab: $SLURM_NODELIST"

mkdir -p /ssd_scratch/cvit/jobinkv
# geting the image
mkdir -p /ssd_scratch/cvit/jobinkv/wise
# geting the image
rsync -avz jobinkv@ada:/share3/jobinkv/wise.zip /ssd_scratch/cvit/jobinkv/wise/
rsync -avz jobinkv@ada:/share3/jobinkv/wiseAllLabels.zip /ssd_scratch/cvit/jobinkv/wise/
cd /ssd_scratch/cvit/jobinkv/wise
unzip -n wise.zip
unzip -n wiseAllLabels.zip
#space data
mkdir -p /ssd_scratch/cvit/jobinkv/spase
rsync -avz jobinkv@ada:/share3/jobinkv/space.zip /ssd_scratch/cvit/jobinkv/spase/
rsync -avz jobinkv@ada:/share3/jobinkv/spaseLabel.zip /ssd_scratch/cvit/jobinkv/spase/
cd /ssd_scratch/cvit/jobinkv/spase/
unzip -n space.zip
unzip -n spaseLabel.zip
#synthetic slide
mkdir -p /ssd_scratch/cvit/jobinkv/pyTorchPreTrainedModels/
mkdir -p /ssd_scratch/cvit/jobinkv/data_check/
rsync -avz jobinkv@ada:/share3/jobinkv/pyTorchPreTrainedModels/resnet101-5d3b4d8f.pth /ssd_scratch/cvit/jobinkv/pyTorchPreTrainedModels/
rsync -avz jobinkv@ada:/share3/jobinkv/resnet101-imagenet.pth /ssd_scratch/cvit/jobinkv/

dataset='spaseAll' #'spase' #'wise' 
attn='LEANetv1'
tails='_final.pth'
mode='train' #'trainval'
model='deepv3'
dot='.'
arch='DeepR101V3PlusD_LEANet_OS8'
model_name=$attn-$arch-$dataset$tails
cd /home/jobinkv/attention_layers/final_run/$attn

#python -m torch.distributed.launch --nproc_per_node=1 trainslide.py --dataset $dataset\


python trainslide.py --dataset $dataset\
  --arch network.$model$dot$arch \
  --city_mode $mode  --lr 0.04 --poly_exp 0.9 \
  --hanet_lr 0.04 --hanet_poly_exp 0.9 \
  --crop_size 564  --color_aug 0.25  --max_iter 57000  \
  --bs_mult 2 --pos_rfactor 18 --dropout 0.1  \
  --best_model_name $model_name --jobid $SLURM_JOB_ID\
  --exp $SLURM_NODELIST_$SLURM_JOB_ID --ckpt /ssd_scratch/cvit/jobinkv/ \
  --tb_path "/ssd_scratch/cvit/jobinkv/$SLURM_JOB_ID" --syncbn --sgd --gblur --aux_loss \
  --snapshot "/ssd_scratch/cvit/jobinkv/$SLURM_JOB_ID/$model_name" \
  --template_selection_loss_contri 0.1 --backbone_lr 0.01 --multi_optim

rsync -avz /ssd_scratch/cvit/jobinkv/$SLURM_JOB_ID/$SLURM_JOB_ID 10.2.16.142:/mnt/1/pytorchTensorboard/runs/$dataset-$attn-$model
rsync -avz /ssd_scratch/cvit/jobinkv/$SLURM_JOB_ID/$model_name 10.2.16.142:/mnt/3/icdar21/slide/result4paper

#Arguments
#--hanet_set 3 64 3 ==> kernel_size=3, r_factor=64, layer=3
#--hanet_pos 2 1 ==>position injection parameter 
#pos_rfactor == > average pool 128//pos_rfactor



