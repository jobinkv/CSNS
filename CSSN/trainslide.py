"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch
from ipdb import set_trace as st
from config import cfg, assert_and_infer_cfg
from utils.evalNew import Eval
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss_multi as loss
import network
import optimizer
import time
import torchvision.utils as vutils
import torch.nn.functional as F
from network.mynn import freeze_weights, unfreeze_weights
import numpy as np
from PIL import Image,ImageDraw
from torchvision.utils import save_image
import cv2
import matplotlib.pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='cityscapes, mapillary, camvid, kitti')
parser.add_argument('--best_model_name', type=str, default='best_model',
                    help='in this name the model saved')
parser.add_argument('--jobid', type=str, default='98098',
                    help='job id assigned')
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')

parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--max_cu_epoch', type=int, default=10000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--multi_optim', action='store_true', default=False,
                    help='multi step optimization')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--template_snapshot', type=str, default=None)
parser.add_argument('--template_mask', type=str, default=None)
parser.add_argument('--snapshot_pe', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='/ssd_scratch/cvit/jobinkv/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', type=bool, default=False,
                    help='for selecting training or testing mode')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--num_cluster', type=int, default=5,
                    help='total number of templates')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--hanet', nargs='*', type=int, default=[0,0,0,0,0],
                    help='Row driven attention networks module')
parser.add_argument('--hanet_set', nargs='*', type=int, default=[0,0,0],
                    help='Row driven attention networks module')
parser.add_argument('--hanet_pos', nargs='*', type=int, default=[0,0,0],
                    help='Row driven attention networks module')
parser.add_argument('--pos_rfactor', type=int, default=16,
                    help='number of position information, if 0, do not use')
parser.add_argument('--aux_loss', action='store_true', default=False,
                    help='auxilliary loss on intermediate feature map')
parser.add_argument('--attention_loss', type=float, default=0.0)
parser.add_argument('--template_selection_loss_contri', type=float, default=0.4)
parser.add_argument('--auxi_loss_contri', type=float, default=0.4)
parser.add_argument('--template_loss_contri', type=float, default=0.0)
parser.add_argument('--hanet_poly_exp', type=float, default=0.0)
parser.add_argument('--backbone_lr', type=float, default=0.00,
                    help='different learning rate on backbone network')
parser.add_argument('--hanet_lr', type=float, default=0.01,
                    help='different learning rate on attention module')
parser.add_argument('--hanet_wd', type=float, default=0.0001,
                    help='different weight decay on attention module')                    
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--pos_noise', type=float, default=0.0)
parser.add_argument('--no_pos_dataset', action='store_true', default=False,
                    help='get dataset with position information')
parser.add_argument('--use_hanet', action='store_true', default=False,
                    help='use hanet')
parser.add_argument('--pooling', type=str, default='mean',
                    help='pooling methods, average is better than max')
args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
#select dataset
if args.dataset=='wiseAll':
    from datasets.wiseAll import labels_name_list
elif args.dataset=='spaseAll':
    from datasets.spaseAll import labels_name_list
# Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

# if args.apex:
# Check that we are running with cuda as distributed is only supported for cuda.
torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

torch.distributed.init_process_group(backend='nccl',
                                        init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.local_rank)

printing_args = ['lr','arch','dataset','sgd','trunk','max_epoch','pos_rfactor','max_iter','max_cu_epoch', 'weight_decay','snapshot', 'exp','hanet','aux_loss','attention_loss','template_loss_contri','hanet_poly_exp','backbone_lr','hanet_lr','use_hanet','date_str','train_batch_size','val_batch_size', 'multi_optim']
def args2table(args):
    table='<table>'
    for key in vars(args):
        if key in printing_args:
            table+='<tr><th>'+key+'</th>'
            table+='<td>'+str(getattr(args, key))+'</td></tr>'
            print (key,str(getattr(args, key)))
    return table+'</table>'

def dicts2table(dicts):
    table='<table>'
    for key in dicts:
        table+='<tr><th>'+key+'</th>'
        table+='<td>'+str(dicts[key])+'</td></tr>'
    return table+'</table>'

def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)
    if args.attention_loss>0 and args.hanet[4]==0:
        print("last hanet is not defined !!!!")
    train_loader, val_loader, test_loader, train_obj = datasets.setup_loaders(args)
    #create_cluster(train_loader,6)
    #templateAnalys()
    #refineCluster()
    criterion, criterion_val,criterion_cluster = loss.get_loss(args)
    if args.aux_loss:
        criterion_aux = loss.get_loss_aux(args)
        net = network.get_net(args, criterion, criterion_cluster, criterion_aux)
    else:
        net = network.get_net(args, criterion,criterion_cluster)      

    for i in range(5):
        if args.hanet[i] > 0:
            args.use_hanet = True
    if args.multi_optim:
        optim, scheduler, optim_at, scheduler_at = optimizer.get_optimizer_attention(args, net)
        print ('Multi optim triggered')
    else:
        optim, scheduler = optimizer.get_optimizer(args, net)
        print ('uniform optim')
  
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)
    epoch = 0
    i = 0
    writer.add_text('Experiment setup',args2table(args), 0)
    if args.test_mode:
        checkpoint = torch.load(args.snapshot, map_location=torch.device('cpu'))
        loaded_dict = checkpoint['state_dict']
        net_state_dict = net.state_dict()
        new_loaded_dict = {}
        for k in net_state_dict:
            if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
                new_loaded_dict[k] = loaded_dict[k]
            else:
                logging.info("Skipped loading parameter %s", k)
        net_state_dict.update(new_loaded_dict)
        net.load_state_dict(net_state_dict)
        logging.info("Checkpoint Load Compelete") 
        test(test_loader, net,  writer) 
        exit()
    print("#### iteration", i)
    torch.cuda.empty_cache()
    # Main Loop
    # for epoch in range(args.start_epoch, args.max_epoch):

    if (args.use_hanet and args.hanet_pos[1] == 0):  # embedding
        if args.hanet_lr > 0.0:
            validate(val_loader, net, criterion_val, optim, scheduler, epoch, writer, i, optim_at, scheduler_at)
        else:
            validate(val_loader, net, criterion_val, optim, scheduler, epoch, writer, i)
    best_MIoU=0
    best_PA=0
    best_epoch=0
    while i < args.max_iter:
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.ITER = i
        cfg.immutable(True)

        if (args.use_hanet and args.hanet_lr > 0.0):
            # validate(val_loader, net, criterion_val, optim, epoch, writer, i, optim_at)
            i = train(train_loader, net, optim, epoch, writer, scheduler, args.max_iter, optim_at, scheduler_at)
            #train_loader.sampler.set_epoch(epoch + 1)
            val_loss, PA, MIoU = validate(val_loader, net, criterion_val, optim, scheduler, 
                   epoch+1, writer, i, optim_at, scheduler_at)
        else:
            # validate(val_loader, net, criterion_val, optim, epoch, writer, i)
            i = train(train_loader, net, optim, epoch, writer, scheduler, args.max_iter)
            #train_loader.sampler.set_epoch(epoch + 1)
            val_loss, PA, MIoU = validate(val_loader, net, criterion_val, optim, scheduler, epoch+1, writer, i)
        is_best = MIoU>best_MIoU
        if is_best:
            best_MIoU=MIoU
            best_PA = PA
            best_epoch = epoch
            torch.save({
             'state_dict': net.state_dict(),
             'epoch': epoch,
             'mean_iu': MIoU,
             'Pixel_ac': PA,
             }, os.path.join(args.ckpt,args.jobid,args.best_model_name))
            logging.info("=>best_MIoU {} at {}".format(best_MIoU, epoch))
            logging.info("=>saving the final checkpoint to " + os.path.join(args.ckpt,args.jobid,args.best_model_name))
        else:
            logging.info("=>MIoU {} is not improved from {} at {}".format(MIoU, best_MIoU, epoch))


        if args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.build_epoch(cut=True)
                # if args.apex:
                train_loader.sampler.set_num_samples()
            else:
                train_obj.build_epoch()
        epoch += 1
    for pnt in range(0,best_epoch):
        writer.add_scalar('Val_best/MIoU', (best_MIoU), pnt)
        writer.add_scalar('Val_best/pixel_accuracy', (best_PA), pnt)
    if 1==1:
        checkpoint = torch.load(os.path.join(args.ckpt,args.jobid,args.best_model_name), map_location=torch.device('cpu'))
        loaded_dict = checkpoint['state_dict']
        net_state_dict = net.state_dict()
        new_loaded_dict = {}
        for k in net_state_dict:
            if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
                new_loaded_dict[k] = loaded_dict[k]
            else:
                logging.info("Skipped loading parameter %s", k)
        net_state_dict.update(new_loaded_dict)
        net.load_state_dict(net_state_dict)
        logging.info("Checkpoint Load Compelete") 
        test(test_loader, net,  writer) 
        exit()


def refineCluster():
    templates = torch.load('cluster.pth', map_location=torch.device('cpu'))
    cluster_out=[]
    for i in range(0,25):
        for j in range (1,6):
            cluster_out.append(templates[j]['centr'][i,:,:])
    cluster = torch.stack(cluster_out)
    cluster = F.interpolate(cluster.unsqueeze(0), (68, 90), mode='nearest')
    torch.save(cluster.squeeze(0),'refined_cluster3.pth')
    print ('done')
    exit()
def refineCluster_old():
    templates = torch.load('cluster.pth', map_location=torch.device('cpu'))
    cluster = torch.cat((templates[1]['centr'],
                 templates[2]['centr'],templates[3]['centr'],
                 templates[4]['centr'],templates[5]['centr']),0)
    cluster = F.interpolate(cluster.unsqueeze(0), (68, 90), mode='nearest')
    torch.save(cluster.squeeze(0),'refined_cluster2.pth')
    exit()
    cluster_out=[]
    st()
    for j in range(1,len(templates)):
        cluster_out.append(templates[j]['centr'])
    allCenter = torch.stack(cluster_out)
    allCenter = F.interpolate(allCenter, (68, 90), mode='nearest')
    st()
    #torch.save({'cluster':downSampled_cen,'noCluster':5},'refined_cluster.pth') 
    torch.save(allCenter,'refined_cluster1.pth') 

def templateAnalys():
   
    templates = torch.load('cluster.pth', map_location=torch.device('cpu'))
    images_out=[]
    allCenter = torch.stack([item['centr'] for item in templates])
    allCenter = F.interpolate(allCenter, (68, 90), mode='nearest')
    downSampled_cen = F.interpolate(allCenter, (200,400), mode='nearest')
    for j in range(1,len(templates)):
        image1 = labelWritenp('cluster'+str(j),400)
        for i in range(0,templates[j]['centr'].shape[0]):
            tmp = downSampled_cen[j,i,:,:]
            #if torch.sum(tmp)>0:
            tmp = tmp*255/torch.max(tmp)
            tmp = np.uint8(tmp)
            tmp = cv2.applyColorMap(tmp, cv2.COLORMAP_JET)
            image1 = np.concatenate((image1,labelWritenp(labels_name_list[i],400), tmp), axis=0)
        images_out.append(image1)
    max_len = images_out[4].shape[0]
    final = np.zeros((max_len,7,3),dtype=int)
    for item in images_out:
       final = np.concatenate((final,item,np.zeros((max_len,7,3),dtype=int)),axis=1) 
    cv2.imwrite('cluster_downNup.jpg',final)
    print ('Done')
    exit()


def templateAnalys_old():
   
    templates = torch.load('cluster.pth', map_location=torch.device('cpu'))
    images_out=[]
    max_len = 14210 # maximum length of image
    for j in range(1,len(templates)):
        image1 = labelWritenp('cluster'+str(j),720)
        for i in range(0,templates[j]['centr'].shape[0]):
            tmp = templates[j]['centr'][i,:,:]
            st()
            if torch.sum(tmp)>0:
                tmp = tmp*255/torch.max(tmp)
                tmp = np.uint8(tmp)
                tmp = cv2.applyColorMap(tmp, cv2.COLORMAP_JET)
                image1 = np.concatenate((image1,labelWritenp(labels_name_list[i],720), tmp), axis=0)
        image1 = np.concatenate((image1,np.zeros((max_len-image1.shape[0], 720,3),dtype=int)), axis=0)
        images_out.append(image1)
    final = np.zeros((max_len,7,3),dtype=int)
    for item in images_out:
       final = np.concatenate((final,item,np.zeros((max_len,7,3),dtype=int)),axis=1) 
    cv2.imwrite('cluster.jpg',final)
    print ('Done')
    exit()

def create_cluster(train_loader,num_cluster):
    cluster = {"centr":torch.zeros((25, 540, 720), dtype=torch.float),"count":1}
    templates = []
    for j in range(0,num_cluster):
        templates.append(cluster.copy())
    for i, data in enumerate(train_loader):
        if args.no_pos_dataset:
            inputs, gts, _img_name = data
        elif args.pos_rfactor > 0:
            inputs, gts, _img_name, aux_gts, (pos_h, pos_w), clr = data
        else:
            inputs, gts, _img_name, aux_gts = data
        for k in range(0,gts.shape[0]):
            templates[int(clr[k])]['centr'] = templates[int(clr[k])]['centr']+gts[k,:,:,:]
            templates[int(clr[k])]['count']+=1

        if i % 50 == 50:
            print ('Done', i,'/',len(train_loader))

    for j in range(0,num_cluster):
        templates[j]['centr']/=templates[j]['count']
    torch.save(templates,'cluster.pth') 
    print ('done')
    exit()
def train(train_loader, net, optim, curr_epoch, writer, scheduler, max_iter, optim_at=None, scheduler_at=None):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()
    requires_attention = False

    if args.attention_loss>0:
        get_attention_gt = Generate_Attention_GT(args.dataset_cls.num_classes)
        criterion_attention = loss.get_loss_bcelogit(args)
        requires_attention = True
    
    train_total_loss = AverageMeter()
    train_template_loss = AverageMeter()
    train_template_att_loss = AverageMeter()
    time_meter = AverageMeter()

    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):
        # inputs = (2,3,713,713)
        # gts    = (2,713,713)
        if curr_iter >= max_iter:
            break
        start_ts = time.time()
        if args.no_pos_dataset:
            inputs, gts, _img_name = data
        elif args.pos_rfactor > 0:
            inputs, gts, _img_name, aux_gts, (pos_h, pos_w), clr = data
        else:
           
            inputs, gts, _img_name, aux_gts = data
        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gts = inputs.cuda(), gts.cuda()
        batch_size = inputs.shape[0] 
        optim.zero_grad()
        if optim_at is not None:
            optim_at.zero_grad() #best
        if args.no_pos_dataset:
            main_loss = net(inputs, gts=gts)        
            del inputs, gts
        else:
            if args.pos_rfactor > 0:
                outputs = net(inputs, gts=gts, aux_gts=aux_gts, pos=(pos_h, pos_w), attention_loss=requires_attention, cluster =clr) #best
            else:
                outputs = net(inputs, gts=gts, aux_gts=aux_gts, attention_loss=requires_attention)
            
            if args.aux_loss:
                main_loss,tempt_loss, aux_loss,tempt_att_loss = outputs[0], outputs[1],outputs[2],outputs[3] #best
                if args.attention_loss>0:
                    attention_map = outputs[2]
                    attention_labels = get_attention_gt(aux_gts, attention_map.shape)
                    # print(attention_map.shape, attention_labels.shape)
                    attention_loss = criterion_attention(input=attention_map.transpose(1,2), target=attention_labels.transpose(1,2))
            else:
                if args.attention_loss>0:
                    main_loss = outputs[0]
                    attention_map = outputs[1]
                    attention_labels = get_attention_gt(aux_gts, attention_map.shape)
                    # print(attention_map.shape, attention_labels.shape)
                    attention_loss = criterion_attention(input=attention_map.transpose(1,2), target=attention_labels.transpose(1,2))
                else:
                    main_loss = outputs 

            del inputs, gts, aux_gts #best

        if args.no_pos_dataset:
            total_loss = main_loss
        elif args.attention_loss>0:
            if args.aux_loss:
                total_loss = main_loss + (0.4 * aux_loss) + (args.attention_loss * attention_loss)
            else:
                total_loss = main_loss + (args.attention_loss * attention_loss)
        else:
            if args.aux_loss:
                total_loss = main_loss + (args.template_selection_loss_contri*tempt_loss)+ \
                             (args.auxi_loss_contri * aux_loss)+(args.template_loss_contri*tempt_att_loss)
            else:
                total_loss = main_loss

        log_total_loss = total_loss.clone().detach_()
        log_template_loss = tempt_loss.clone().detach_()
        log_template_att_loss = tempt_att_loss.clone().detach_()

        torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(log_template_loss, torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(log_template_att_loss, torch.distributed.ReduceOp.SUM)

        log_total_loss = log_total_loss / args.world_size
        log_template_loss = log_template_loss / args.world_size
        log_template_att_loss = log_template_att_loss / args.world_size

        train_total_loss.update(log_total_loss.item(), batch_pixel_size)
        train_template_loss.update(log_template_loss.item(), batch_pixel_size)
        train_template_att_loss.update(log_template_att_loss.item(), batch_pixel_size)

        total_loss.backward()
        optim.step()
        if optim_at is not None:
            optim_at.step()

        scheduler.step()
        if scheduler_at is not None:
            scheduler_at.step()

        time_meter.update(time.time() - start_ts)

        del total_loss, log_total_loss

        curr_iter += 1

        if args.local_rank == 0:
            if i % 50 == 49:
                if optim_at is not None:
                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [lr_at {:0.6f}], [time {:0.4f}]'.format(                   curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss.avg,
                    optim.param_groups[-1]['lr'], optim_at.param_groups[-1]['lr'], time_meter.avg / args.train_batch_size)
                else:
                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                        curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss.avg,
                        optim.param_groups[-1]['lr'], time_meter.avg / args.train_batch_size)
    
                logging.info(msg)
                #logging.info('batch size: {}'.format(batch_size))
                # Log tensorboard metrics for each iteration of the training phase
                writer.add_scalar('Train/main_loss', (train_total_loss.avg),  curr_iter)
                writer.add_scalar('Train/template_loss', (train_template_loss.avg),  curr_iter)
                writer.add_scalar('Train/template_att_loss', (train_template_att_loss.avg),  curr_iter)
                train_total_loss.reset()
                train_template_loss.reset()
                time_meter.reset()

        if i > 5 and args.test_mode:
            return curr_iter

    return curr_iter

def validate(val_loader, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, optim_at=None, scheduler_at=None):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    tmplate_acc = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []
    myEval = Eval(labels_name_list)
    myEval.reset()
    for val_idx, data in enumerate(val_loader):
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        if args.no_pos_dataset:
            inputs, gt_image, img_names = data
        elif args.pos_rfactor > 0:
            inputs, gt_image, img_names, _, (pos_h, pos_w),template_gt = data
        else:
            inputs, gt_image, img_names, template_gt = data
        assert len(inputs.size()) == 4 and len(gt_image.size()) == 4
        assert inputs.size()[2:] == gt_image.size()[2:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            if args.pos_rfactor > 0:
                if args.use_hanet and args.hanet_pos[0] > 0:  # use hanet and position
                    #output, attention_map, pos_map = net(inputs, pos=(pos_h, pos_w), attention_map=True)
                    output,template_out  = net(inputs, pos=(pos_h, pos_w), attention_map=True)
                else:
                    output,template_out = net(inputs, pos=(pos_h, pos_w))
            else:
                output,template_out = net(inputs)

        del inputs

        assert output.size()[2:] == gt_image.size()[2:]
        assert output.size()[1] == args.dataset_cls.num_classes
        # calculating template selection accuracy
        tmplate_acc.update( Classification_accuracy(template_gt,template_out),1)
        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)
        #label = gt_image.cpu().numpy()
        del gt_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        
        output = output.data.cpu().numpy()
        argpred = output.copy()
        argpred[output>=0.5] = 1
        argpred[output<0.5]  = 0
        #predictions = output.data.max(1)[1].cpu()
        myEval.add_batch(gt_image.numpy(), argpred)
        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        #if val_idx < 10:
        #    dump_images.append([gt_image, predictions, img_names])

        #iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
        #                     args.dataset_cls.num_classes)
        del output, val_idx, data

    #iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    #torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    #iou_acc = iou_acc_tensor.cpu().numpy()

    #if args.local_rank == 0:
    #    if args.use_hanet and args.hanet_pos[0] > 0:  # use pos and hanet
    #        visualize_attention(writer, attention_map, curr_iter)
            #if args.hanet_pos[1] == 0:  # embedding
            #    visualize_pos(writer, pos_map, curr_iter)
    PA = myEval.Pixel_Accuracy()
    MIoU = myEval.Pixel_Intersection_over_Union()
    print (PA,MIoU)
    writer.add_scalar('validation/pixel_accuracy', (PA), curr_iter)
    writer.add_scalar('validation/MIoU', (MIoU), curr_iter)
    writer.add_scalar('validation/Template_acc', (tmplate_acc.avg), curr_iter)
    
    logging.info("Template selection accuracy: %f ", val_loss.avg)
    return val_loss.avg, PA, MIoU
def Classification_accuracy(template_gt,template_out):
    batch_size = template_gt.size(0)
    _, pred = template_out.topk(1, 1, True, True)
    template_out = pred.data.cpu()
    correct = template_out.eq(template_gt.view(1, -1).expand_as(template_out))
    correct_k = correct.view(-1).float().sum(0)
    return correct_k.mul_(100.0 / batch_size)

def test(test_loader, net,  writer):
    """
    Runs after the training
    net: thet network
    writer: tensorboard writer
    """
    selectedImgList=['132810-ZRRDZD73S8-540_frame9540','140105-HFH3KIYANZ-540_frame15330','140603-1I2W1LJXSJ-540_frame240', '141004-SFJDBON87Z-540_frame5460','142204-HYTHBBJ5WA-540_frame32430']
    net.eval()
    dump_images = []
    myEval = Eval(labels_name_list)
    myEval.reset()
    for test_idx, data in enumerate(test_loader):
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        if args.no_pos_dataset:
            inputs, gt_image, img_names = data
        elif args.pos_rfactor > 0:
            inputs, gt_image, img_names, _, (pos_h, pos_w),_ = data
        else:
            inputs, gt_image, img_names, _ = data
        assert len(inputs.size()) == 4 and len(gt_image.size()) == 4
        assert inputs.size()[2:] == gt_image.size()[2:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            if args.pos_rfactor > 0:
                if args.use_hanet and args.hanet_pos[0] > 0:  # use hanet and position
                    #output, attention_map, pos_map = net(inputs, pos=(pos_h, pos_w), attention_map=True)
                    output,t_out  = net(inputs, pos=(pos_h, pos_w), attention_map=True)
                else:
                    output,t_out = net(inputs, pos=(pos_h, pos_w))
            else:
                output, t_out = net(inputs)

        #del inputs

        assert output.size()[2:] == gt_image.size()[2:]
        assert output.size()[1] == args.dataset_cls.num_classes

        #del gt_cuda

        
        output = output.data.cpu().numpy()
        argpred = output.copy()
        argpred[output>=0.5] = 1
        argpred[output<0.5]  = 0
        #predictions = output.data.max(1)[1].cpu()
        myEval.add_batch(gt_image.numpy(), argpred)
        # Logging
        if img_names[0] in selectedImgList :
            if args.local_rank == 0:
                logging.info("Testing: %d / %d", test_idx + 1, len(test_loader))
                plotSampleImges(writer, inputs, gt_image, argpred,img_names,category='Test')
        # Image Dumps
        #if test_idx < 10:
        #    dump_images.append([gt_image, predictions, img_names])

        #iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
        #                     args.dataset_cls.num_classes)
        del output, test_idx, data

    #iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    #torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    #iou_acc = iou_acc_tensor.cpu().numpy()

    #if args.local_rank == 0:
    #    if args.use_hanet and args.hanet_pos[0] > 0:  # use pos and hanet
    #        visualize_attention(writer, attention_map, 0)
            #if args.hanet_pos[1] == 0:  # embedding
            #    visualize_pos(writer, pos_map, curr_iter)
    PA = myEval.Pixel_Accuracy()
    MIoU = myEval.Pixel_Intersection_over_Union()
    class_wise_miou = myEval.Classwise_Intersection_over_Union()
    writer.add_text('Classwise MIoU',dicts2table(class_wise_miou), 0)
    for pnt in range(0,10):
        writer.add_scalar('Testing/pixel_accuracy', (PA), pnt)
        writer.add_scalar('Testing/MIoU', (MIoU), pnt)
    logging.info("Testing Result: MIoU= %f, and Pixel accuracy= %f", MIoU , PA)



num_vis_pos = 0

def visualize_pos(writer, pos_maps, iteration):
    global num_vis_pos
    #if num_vis_pos % 5 == 0:
    #    save_pos_numpy(pos_maps, iteration)
    num_vis_pos += 1

    stage = 'valid'
    for i in range(len(pos_maps)):
        pos_map = pos_maps[i]
        if isinstance(pos_map, tuple):
            num_pos = 2
        else:
            num_pos = 1

        for j in range(num_pos):
            if num_pos == 2:
                pos_embedding = pos_map[j]
            else:
                pos_embedding = pos_map

            H, D = pos_embedding.shape
            pos_embedding = pos_embedding.unsqueeze(0)  # 1 X H X D
            if H > D:   # e.g. 32 X 8
                pos_embedding = F.interpolate(pos_embedding, H, mode='nearest') # 1 X 32 X 8
                D = H
            elif H < D:   # H < D, e.g. 32 X 64
                pos_embedding = F.interpolate(pos_embedding.transpose(1,2), D, mode='nearest').transpose(1,2) # 1 X 32 X 64
                H = D
            if args.hanet_pos[1]==1: # pos encoding
                pos_embedding = torch.cat((torch.ones(1, H, D).cuda(), pos_embedding/2, pos_embedding/2), 0)
            else:   # pos embedding
                pos_embedding = torch.cat((torch.ones(1, H, D).cuda(), torch.sigmoid(pos_embedding*20),
                                        torch.sigmoid(pos_embedding*20)), 0)
            pos_embedding = vutils.make_grid(pos_embedding, padding=5, normalize=False, range=(0,1))
            writer.add_image(stage + '/Pos/layer-' + str(i) + '-' + str(j), pos_embedding, iteration)

def save_pos_numpy(pos_maps, iteration):
    file_fullpath = '/home/userA/shchoi/Projects/visualization/pos_data/'
    file_name = str(args.date) + '_' + str(args.hanet_pos[0]) + '_' + str(args.exp) + '_layer'

    for i in range(len(pos_maps)):
        pos_map = pos_maps[i]
        if isinstance(pos_map, tuple):
            num_pos = 2
        else:
            num_pos = 1

        for j in range(num_pos):
            if num_pos == 2:
                pos_embedding = pos_map[j]
            else:
                pos_embedding = pos_map

            H, D = pos_embedding.shape
            pos_embedding = pos_embedding.data.cpu().numpy()   # H X D
            file_name_post = str(i) + '_' + str(j) + '_' + str(H) + 'X' + str(D) + '_' + str(iteration)
            np.save(file_fullpath + file_name + file_name_post, pos_embedding)

# plot image on tensorboard
def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
             else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
             for i in range(x.dim()))
    return x[inds]

def inv_preprocess(imgs):
    #if numpy_transform:
    #imgs = flip(imgs, 1)
    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)
    norm_ip(imgs, float(imgs.min()), float(imgs.max()))
    return imgs

def decode_labels(mask, num_classes=2):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    label_colours = [
        (0,   0,   0  ), # bacground
        (0,   255, 0  ), # currect
        (255, 0,   0  ) # list
        ]
    if isinstance(mask, torch.Tensor):
        mask = mask.data.cpu().numpy()
    n,c,h, w = mask.shape
    #if n < num_images:
    num_images = n
    #outputs = np.zeros((num_images*26, h, w, 3), dtype=np.uint8)
    result =[]
    for i in range(num_images):
      outputs = np.zeros((c, h, w, 3), dtype=np.uint8)
      for jj in range(c):
          img = Image.new('RGB', (w, h))
          pixels = img.load()
          new_mask = mask[i,jj,:,:]
          for j_, j in enumerate(new_mask[:, :]):
              for k_, k in enumerate(j):
                  if k < num_classes:
                      pixels[k_,j_] = label_colours[int(k)]
          outputs[jj] = np.array(img)
      result.append(torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0))
    return result

def labelWrite(labelName,width):
    img = Image.new('RGB', (width, 30), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10,10), labelName, fill=(255,255,0))
    border_image = torch.from_numpy(np.asarray(img, np.float32)).div_(255.0)
    border_image = torch.transpose(border_image,2,1)
    border_image = torch.transpose(border_image,0,1)
    return border_image

def labelWritenp(labelName,width):
    img = Image.new('RGB', (width, 30), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((100,10), labelName, fill=(255,255,0))
    return np.asarray(img)
def verticalLine(height):
    img = Image.new('RGB', (10,height), color = (173, 9, 37))
    border_image = torch.from_numpy(np.asarray(img, np.float32)).div_(255.0)
    border_image = torch.transpose(border_image,2,1)
    border_image = torch.transpose(border_image,0,1)
    return border_image

def verticalLine(height):
    img = Image.new('RGB', (10,height), color = (173, 9, 37))
    border_image = torch.from_numpy(np.asarray(img, np.float32)).div_(255.0)
    border_image = torch.transpose(border_image,2,1)
    border_image = torch.transpose(border_image,0,1)
    return border_image

def plotSampleImges(writer, x, label, argpred,img_name, category='Test'):
    images_inv = inv_preprocess(x.clone().cpu())
    img_sample = F.interpolate(images_inv, (200,400), mode='bilinear',align_corners=True)# batch
    labels_colors = decode_labels(label) # batch
    preds_colors = decode_labels(argpred)
    for ind, (image, label_img, pred_img,name) in enumerate(zip(img_sample,labels_colors,preds_colors,img_name)):
        label_sam = F.interpolate(label_img, (200,400), mode='nearest')
        pred_sam = F.interpolate(pred_img, (200,400), mode='nearest')
        image1 = image.detach().clone()
        image2 = image.detach().clone()
        for idx, (lab, pre) in enumerate(zip(label_sam,pred_sam)):
            if torch.sum(lab) == 0:
                continue
            image1 = torch.cat((image1, labelWrite(labels_name_list[idx],400) ,lab),1)
            image2 = torch.cat((image2, labelWrite(labels_name_list[idx],400) ,pre),1)
        #final_out = torch.cat((image1,verticalLine(image1.shape[1]),image2),2)
        #save_image(final_out,img_name[0]+'.png')
        writer.add_image(category+'/'+ name+'/Images', torch.cat((image1,verticalLine(image1.shape[1]),image2),2), 0)


def visualize_attention(writer, attention_map, iteration, threshold=0):
    stage = 'valid'
    for i in range(len(attention_map)):
        C = attention_map[i].shape[1]
        #H = alpha[2].shape[2]
        attention_map_sb = F.interpolate(attention_map[i], C, mode='nearest')
        attention_map_sb = attention_map_sb[0].transpose(0,1).unsqueeze(0)  # 1 X H X C X 1, 
        attention_map_sb = torch.cat((torch.ones(1, C, C).cuda(), torch.abs(attention_map_sb - 1.0),
                        torch.abs(attention_map_sb - 1.0)), 0)
        attention_map_sb = vutils.make_grid(attention_map_sb, padding=5, normalize=False, range=(threshold,1))
        writer.add_image(stage + '/Attention/Row-wise-' + str(i), attention_map_sb, iteration)

from threading import Thread
#import cupy as cp
    
class Generate_Attention_GT(object):   # 34818
    def __init__(self, n_classes=19):
        self.channel_weight_factor = 0   # TBD
        self.ostride = 0
        self.labels = 0
        self.attention_labels = 0
        self.n_classes = n_classes

    def rows_hasclass(self, B, C):
        rows = cp.where(self.labels[B]==C)[0]
        if len(rows) > 0:
            row = cp.asnumpy(cp.unique((rows//self.ostride), return_counts=False))
            print("channel", C, "row", row)
            self.attention_labels[B][C][row] = 1

    def __call__(self, labels, attention_size):
        B, C, H = attention_size
        # print(labels.shape, attention_size)
        self.labels = cp.asarray(labels)
        self.attention_labels = torch.zeros(B, self.n_classes, H).cuda()
        self.ostride = labels.shape[1] // H

        # threads = []
        for j in range(0, labels.shape[0]):
            for k in range(0, self.n_classes):
                rows = cp.where(self.labels[j]==k)[0]
                if len(rows) > 0:
                    row = cp.asnumpy(cp.unique((rows//self.ostride), return_counts=False))
                    # print("channel", k, "row", row)
                    self.attention_labels[j][k][row] = 1

        return self.attention_labels


if __name__ == '__main__':
    main()
