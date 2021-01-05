#-*-coding:utf-8-*-
# date:2020-04-25
# Author: X.L.Eric
# function: inference

import os
import argparse
import torch
import torch.nn as nn
from data_iter.datasets import letterbox
import numpy as np


import time
import datetime
import os
import math
from datetime import datetime
import cv2
import torch.nn.functional as F

from network.resnet import resnet50,resnet34
from utils.common_utils import *
import copy
# from heatmap import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Landmarks Test')

    parser.add_argument('--test_model', type=str, default = './model_exp/2021-01-05_13-19-59/model_epoch-10.pth',
        help = 'test_model') # 模型路径
    parser.add_argument('--model', type=str, default = 'resnet_50',
        help = 'model : resnet_50') # 模型类型
    parser.add_argument('--num_classes', type=int , default = 196,
        help = 'num_classes') #  分类类别个数
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择
    parser.add_argument('--test_path', type=str, default = './samples/',
        help = 'test_path') # 测试集路径
    parser.add_argument('--img_size', type=tuple , default = (256,256),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--fix_res', type=bool , default = False,
        help = 'fix_resolution') # 输入模型样本图片是否保证图像分辨率的长宽比
    parser.add_argument('--vis', type=bool , default = True,
        help = 'vis') # 是否可视化图片

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    ops = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    print('----------------------------------')

    unparsed = vars(ops) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    #---------------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

    test_path =  ops.test_path # 测试图片文件夹路径

    #---------------------------------------------------------------- 构建模型
    print('use model : %s'%(ops.model))

    if ops.model == 'resnet_50':
        model_ = resnet50(landmarks_num = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == 'resnet_34':
        model_ = resnet34(landmarks_num = ops.num_classes,img_size=ops.img_size[0])

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval() # 设置为前向推断模式

    # print(model_)# 打印模型结构

    # 加载测试模型
    if os.access(ops.test_model,os.F_OK):# checkpoint
        chkpt = torch.load(ops.test_model, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(ops.test_model))

    #---------------------------------------------------------------- 预测图片
    font = cv2.FONT_HERSHEY_SIMPLEX
    with torch.no_grad():
        idx = 0
        for file in os.listdir(ops.test_path):
            if '.jpg' not in file:
                continue
            idx += 1
            print('{}) image : {}'.format(idx,file))
            img = cv2.imread(ops.test_path + file)
            img_width = img.shape[1]
            img_height = img.shape[0]
            # 输入图片预处理
            if ops.fix_res:
                img_ = letterbox(img,size_=ops.img_size[0],mean_rgb = (128,128,128))
            else:
                img_ = cv2.resize(img, (ops.img_size[1],ops.img_size[0]))

            img_ = img_.astype(np.float32)
            img_ = (img_-128.)/256.

            img_ = img_.transpose(2, 0, 1)
            img_ = torch.from_numpy(img_)
            img_ = img_.unsqueeze_(0)

            if use_cuda:
                img_ = img_.cuda()  # (bs, 3, h, w)

            output_landmarks,output_gender,output_age = model_(img_.float())
            # print(pre_.size())
            output_landmarks = output_landmarks.cpu().detach().numpy()


            print(output_gender,output_age)
            output_landmarks = np.squeeze(output_landmarks)
            # print(output.shape)
            dict_landmarks = draw_landmarks(img,output_landmarks,draw_circle = False)

            draw_contour(img,dict_landmarks)
            #----------------------
            output_gender = F.softmax(output_gender,dim = 1)
            output_gender = output_gender[0]
            output_gender = output_gender.cpu().detach().numpy()
            output_gender = np.array(output_gender)
            gender_max_index = np.argmax(output_gender)#概率最大类别索引
            score_gender = output_gender[gender_max_index]# 最大概率
            print(gender_max_index,score_gender)

            # hm = get_heatmap(img, output,radius=7,img_size = 256,gaussian_op = True)
            output_age = output_age.cpu().detach().numpy()[0][0]
            output_age = (output_age*100.+50.)

            if gender_max_index == 1.:
                cv2.putText(img, 'gender:{}'.format("male"), (2,20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)
                cv2.putText(img, 'gender:{}'.format("male"), (2,20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,20,0),1)
            else:
                cv2.putText(img, 'gender:{}'.format("female"), (2,20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)
                cv2.putText(img, 'gender:{}'.format("female"), (2,20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 20, 0),1)
            cv2.putText(img, 'age:{:.2f}'.format(output_age), (2,50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)
            cv2.putText(img, 'age:{:.2f}'.format(output_age), (2,50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,20, 0),1)

            if ops.vis:
                # cv2.namedWindow('r',0)
                # cv2.imshow('r',img_c)
                cv2.namedWindow('image',0)
                cv2.imshow('image',img)
                # cv2.namedWindow('heatmap',0)
                # cv2.imshow('heatmap',hm)
                if cv2.waitKey(1) == 27 :
                    break

    cv2.destroyAllWindows()

    print('well done ')
