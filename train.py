#-*-coding:utf-8-*-
# date:2020-12-30
# Author: X.L.Eric
## function: train

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import  sys
from utils.model_utils import *
from utils.common_utils import *
from data_iter.datasets import *

from network.resnet import resnet50,resnet34
from loss.loss import *
import cv2
import time
import json
from datetime import datetime

def trainer(ops,f_log):
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

        if ops.log_flag:
            sys.stdout = f_log

        set_seed(ops.seed)
        #---------------------------------------------------------------- 构建模型
        print('use model : %s'%(ops.model))

        if ops.model == 'resnet_50':
            model_ = resnet50(pretrained = True,landmarks_num = ops.num_classes,img_size = ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_34':
            model_ = resnet34(pretrained = True,landmarks_num = ops.num_classes,img_size = ops.img_size[0],dropout_factor=ops.dropout)

        use_cuda = torch.cuda.is_available()

        device = torch.device("cuda:0" if use_cuda else "cpu")
        model_ = model_.to(device)

        # print(model_)# 打印模型结构
        # Dataset
        dataset = LoadImagesAndLabels(ops= ops,img_size=ops.img_size,flag_agu=ops.flag_agu,fix_res = ops.fix_res,vis = False)
        print('len train datasets : %s'%(dataset.__len__()))
        # Dataloader
        dataloader = DataLoader(dataset,
                                batch_size=ops.batch_size,
                                num_workers=ops.num_workers,
                                shuffle=True,
                                pin_memory=False,
                                drop_last = True)
        # 优化器设计
        optimizer_SGD = optim.SGD(model_.parameters(), lr=ops.init_lr, momentum=ops.momentum, weight_decay=ops.weight_decay)# 优化器初始化
        optimizer = optimizer_SGD
        # 加载 finetune 模型
        if os.access(ops.fintune_model,os.F_OK):# checkpoint
            chkpt = torch.load(ops.fintune_model, map_location=device)
            model_.load_state_dict(chkpt)
            print('load fintune model : {}'.format(ops.fintune_model))

        print('/**********************************************/')
        # 损失函数
        if ops.loss_define != 'wing_loss':
            criterion = nn.MSELoss(reduce=True, reduction='mean')

        criterion_gender = nn.CrossEntropyLoss()#CrossEntropyLoss() 是 softmax 和 负对数损失的结合

        step = 0
        idx = 0

        # 变量初始化
        best_loss = np.inf
        loss_mean = 0. # 损失均值
        loss_idx = 0. # 损失计算计数器
        flag_change_lr_cnt = 0 # 学习率更新计数器
        init_lr = ops.init_lr # 学习率

        epochs_loss_dict = {}

        for epoch in range(0, ops.epochs):
            if ops.log_flag:
                sys.stdout = f_log
            print('\nepoch %d ------>>>'%epoch)
            model_.train()
            # 学习率更新策略
            if loss_mean!=0.:
                if best_loss > (loss_mean/loss_idx):
                    flag_change_lr_cnt = 0
                    best_loss = (loss_mean/loss_idx)
                else:
                    flag_change_lr_cnt += 1

                    if flag_change_lr_cnt > 20:
                        init_lr = init_lr*ops.lr_decay
                        set_learning_rate(optimizer, init_lr)
                        flag_change_lr_cnt = 0

            loss_mean = 0. # 损失均值
            loss_idx = 0. # 损失计算计数器

            for i, (imgs_,pts_,gender_,age_) in enumerate(dataloader):
                # print('imgs_, pts_,gender_,age_ : ',imgs_.size(), pts_.size(),gender_.size(),age_.size())
                # continue
                if use_cuda:
                    imgs_ = imgs_.cuda()  # pytorch 的 数据输入格式 ： (batch, channel, height, width)
                    pts_ = pts_.cuda()
                    gender_ = gender_.cuda()
                    age_ = age_.cuda()
                output_landmarks,output_gender,output_age = model_(imgs_.float())
                # print("output_gender,output_age : ",output_gender.size(),output_age.size())
                if ops.loss_define == 'wing_loss':
                    loss_pts = got_total_wing_loss(output_landmarks, pts_.float())
                    loss_age = got_total_wing_loss(output_age, age_.float())
                else:
                    loss_pts = criterion(output_landmarks, pts_.float())
                    loss_age = criterion(output_age, age_.float())

                loss_gender = criterion_gender(output_gender,gender_)

                loss = loss_pts + 0.2*loss_age + 0.3*loss_gender

                loss_mean += loss.item()
                loss_idx += 1.
                if i%10 == 0:
                    loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print('  %s - %s - epoch [%s/%s] (%s/%s):'%(loc_time,ops.model,epoch,ops.epochs,i,int(dataset.__len__()/ops.batch_size)),\
                    'Mean Loss : %.6f - Loss: %.6f'%(loss_mean/loss_idx,loss.item()),\
                    " loss_pts:{:.4f},loss_age:{:.4f},loss_gender:{:.4f}".format(loss_pts.item(),loss_age.item(),loss_gender.item()),\
                    ' lr : %.5f'%init_lr,' bs:',ops.batch_size,\
                    ' img_size: %s x %s'%(ops.img_size[0],ops.img_size[1]),' best_loss:%.5f'%best_loss)
                # 计算梯度
                loss.backward()
                # 优化器对模型参数更新
                optimizer.step()
                # 优化器梯度清零
                optimizer.zero_grad()
                step += 1


            torch.save(model_.state_dict(), ops.model_exp + 'model_epoch-{}.pth'.format(epoch))

    except Exception as e:
        print('Exception : ',e) # 打印异常
        print('Exception  file : ', e.__traceback__.tb_frame.f_globals['__file__'])# 发生异常所在的文件
        print('Exception  line : ', e.__traceback__.tb_lineno)# 发生异常所在的行数

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Classification Train')
    parser.add_argument('--seed', type=int, default = 123,
        help = 'seed') # 设置随机种子
    parser.add_argument('--model_exp', type=str, default = './model_exp',
        help = 'model_exp') # 模型输出文件夹
    parser.add_argument('--model', type=str, default = 'resnet_50',
        help = 'model : resnet_50') # 模型类型
    parser.add_argument('--num_classes', type=int , default = 196,
        help = 'num_classes') #  landmarks 个数*2
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择

    parser.add_argument('--train_path', type=str,
        default = './wiki_crop_face/label/',
        help = 'train_path')# 训练集标注信息

    parser.add_argument('--pretrained', type=bool, default = True,
        help = 'imageNet_Pretrain') # 初始化学习率
    parser.add_argument('--fintune_model', type=str, default = './model_exp/2021-01-05_14-59-26/model_epoch-4.pth',
        help = 'fintune_model') # fintune model
    parser.add_argument('--loss_define', type=str, default = 'wing_loss',
        help = 'define_loss') # 损失函数定义
    parser.add_argument('--init_lr', type=float, default = 1e-4,
        help = 'init_learningRate') # 初始化学习率
    parser.add_argument('--lr_decay', type=float, default = 0.1,
        help = 'learningRate_decay') # 学习率权重衰减率
    parser.add_argument('--weight_decay', type=float, default = 5e-4,
        help = 'weight_decay') # 优化器正则损失权重
    parser.add_argument('--momentum', type=float, default = 0.9,
        help = 'momentum') # 优化器动量
    parser.add_argument('--batch_size', type=int, default = 32,
        help = 'batch_size') # 训练每批次图像数量
    parser.add_argument('--dropout', type=float, default = 0.5,
        help = 'dropout') # dropout
    parser.add_argument('--epochs', type=int, default = 2000,
        help = 'epochs') # 训练周期
    parser.add_argument('--num_workers', type=int, default = 8,
        help = 'num_workers') # 训练数据生成器线程数
    parser.add_argument('--img_size', type=tuple , default = (256,256),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--flag_agu', type=bool , default = True,
        help = 'data_augmentation') # 训练数据生成器是否进行数据扩增
    parser.add_argument('--fix_res', type=bool , default = False,
        help = 'fix_resolution') # 输入模型样本图片是否保证图像分辨率的长宽比
    parser.add_argument('--clear_model_exp', type=bool, default = False,
        help = 'clear_model_exp') # 模型输出文件夹是否进行清除
    parser.add_argument('--log_flag', type=bool, default = False,
        help = 'log flag') # 是否保存训练 log

    #--------------------------------------------------------------------------
    args = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)
    loc_time = time.localtime()
    args.model_exp = args.model_exp + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)+'/'
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)

    f_log = None
    if args.log_flag:
        f_log = open(args.model_exp+'/train_{}.log'.format(time.strftime("%Y-%m-%d_%H-%M-%S",loc_time)), 'a+')
        sys.stdout = f_log

    print('---------------------------------- log : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", loc_time)))
    print('\n/******************* {} ******************/\n'.format(parser.description))

    unparsed = vars(args) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    unparsed['time'] = time.strftime("%Y-%m-%d %H:%M:%S", loc_time)

    fs = open(args.model_exp+'train_ops.json',"w",encoding='utf-8')
    json.dump(unparsed,fs,ensure_ascii=False,indent = 1)
    fs.close()

    trainer(ops = args,f_log = f_log)# 模型训练

    if args.log_flag:
        sys.stdout = f_log
    print('well done : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
