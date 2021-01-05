#-*-coding:utf-8-*-
# date:2019-05-20
# Author: X.L.Eric
# function: data iter
import glob
import math
import os
import random
import shutil
from pathlib import Path
from PIL import Image
# import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_iter.data_agu import *
from draw_utils.draw_utils import draw_global_contour
import json
# 图像白化
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

# 图像亮度、对比度增强
def contrast_img(img, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img.shape
    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1-c, b)
    return dst

class LoadImagesAndLabels(Dataset):
    def __init__(self, ops, img_size=(224,224), flag_agu = False,fix_res = True,vis = False):

        # for f_ in os.listdir("./WFLW_Faces/label/"):
        #     f = open("./WFLW_Faces/label/" + f_, encoding='utf-8')#读取 json文件
        #     dict = json.load(f)
        #     f.close()
        #
        #     img_path = "./WFLW_Faces/images/" + f_.replace(".json",".jpg")
        #     img = cv2.imread(img_path)
        #     draw_global_contour(img,dict)
        #
        #     cv2.namedWindow("wflw",0)
        #     cv2.imshow("wflw",img)
        #     cv2.waitKey(10)

        print('img_size (height,width) : ',img_size[0],img_size[1])
        print("train_path : {}".format(ops.train_path))
        max_age = 0
        min_age = 65535.
        # idx = 0
        file_list = []
        landmarks_list = []
        age_list = []
        gender_list = []
        idx = 0
        for f_ in os.listdir(ops.train_path):
            f = open(ops.train_path + f_, encoding='utf-8')#读取 json文件
            dict = json.load(f)
            f.close()

            if dict["age"]>90 or dict["age"]<3:
                continue
            idx += 1
            #-------------------------------------------------------------------
            img = cv2.imread(dict["path"])
            file_list.append(dict["path"])

            print("------> Author : {} ,age:{:.3f}, <{}> {}".format(dict["author"],dict["age"],idx,dict["path"]))
            # print(dict["landmarks"].keys())

            pts = []
            for k_ in dict["landmarks"].keys():
                for pt_ in dict["landmarks"][k_]:
                    x,y = pt_
                    pts.append([x,y])
                    # print("x,y : ",x,y)
                    # cv2.circle(img, (int(x),int(y)), 5, (0,255,0),-1)
            # print(len(pts))
            landmarks_list.append(pts)
            if dict["gender"] == "male":
                gender_list.append(1)
            else:
                gender_list.append(0)
            age_list.append(dict["age"])
            if max_age < dict["age"]:
                max_age = dict["age"]
            if min_age > dict["age"]:
                min_age = dict["age"]
            # if idx > 1000:
            #     break
            if False:
                # print(img.shape)
                x1,y1,x2,y2 = dict["loc"]
                # print("x1,y1,x2,y2",x1,y1,x2,y2)
                cv2.rectangle(img, (int(x1), int(y1)),(int(x2), int(y2)), (255, 255, 0),3)

                draw_global_contour(img,dict["landmarks"])

                cv2.putText(img, 'age:{:.2f}'.format(dict["age"]), (2,20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)
                cv2.putText(img, 'gender:{}'.format(dict["gender"]), (2,40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)
                cv2.putText(img, 'age:{:.2f}'.format(dict["age"]), (2,20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 10, 25),1)
                cv2.putText(img, 'gender:{}'.format(dict["gender"]), (2,40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 10, 25),1)

                x1_ = x1 + random.randint(-5,5)
                y1_ = y1 + random.randint(-5,5)
                x2_ = x2 + random.randint(-5,5)
                y2_ = y2 + random.randint(-5,5)

                x1_ = max(0,x1_)
                x1_ = min(img.shape[1]-1,x1_)
                x2_ = max(0,x2_)
                x2_ = min(img.shape[1]-1,x2)
                y1_ = max(0,y1_)
                y1_ = min(img.shape[0]-1,y1_)
                y2_ = max(0,y2_)
                y2_ = min(img.shape[0]-1,y2_)

                cv2.namedWindow("result",0)
                cv2.imshow("result",img)
                cv2.waitKey(1)
        cv2.destroyAllWindows()

        print("max_age : {:.3f} ,min_age : {:.3f}".format(max_age,min_age))
        self.files = file_list
        self.landmarks = landmarks_list
        self.ages = age_list
        self.genders = gender_list
        self.img_size = img_size
        self.flag_agu = flag_agu
        self.fix_res = fix_res
        self.vis = vis

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]
        gender = self.genders[index]
        age = self.ages[index]

        img = cv2.imread(img_path)  # BGR
        if self.flag_agu == True:
            left_eye = np.average(pts[60:68], axis=0)
            right_eye = np.average(pts[68:76], axis=0)

            angle_random = random.randint(-22,22)
            # 返回 crop 图 和 归一化 landmarks
            img_, landmarks_  = face_random_rotate(img, pts, angle_random, left_eye, right_eye,
                fix_res = self.fix_res,img_size = self.img_size,vis = False)
        if self.flag_agu == True:
            if random.random() > 0.5:
                c = float(random.randint(80,120))/100.
                b = random.randint(-10,10)
                img_ = contrast_img(img_, c, b)
        if self.flag_agu == True:
            if random.random() > 0.9:
                # print('agu hue ')
                img_hsv=cv2.cvtColor(img_,cv2.COLOR_BGR2HSV)
                hue_x = random.randint(-10,10)
                # print(cc)
                img_hsv[:,:,0]=(img_hsv[:,:,0]+hue_x)
                img_hsv[:,:,0] =np.maximum(img_hsv[:,:,0],0)
                img_hsv[:,:,0] =np.minimum(img_hsv[:,:,0],180)#范围 0 ~180
                img_=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
        if self.flag_agu == True:
            if random.random() > 0.95:
                img_ = img_agu_channel_same(img_)
        # cv2.imwrite("./samples/{}.jpg".format(index),img_)
        if self.vis == True:
        # if True:

            cv2.putText(img_, 'age:{:.2f}'.format(age), (2,20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)
            if gender == 1.:
                cv2.putText(img_, 'gender:{}'.format("male"), (2,40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)
            else:
                cv2.putText(img_, 'gender:{}'.format("female"), (2,40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),2)

            cv2.namedWindow('crop',0)
            cv2.imshow('crop',img_)
            cv2.waitKey(1)
        img_ = img_.astype(np.float32)
        img_ = (img_-128.)/256.
        img_ = img_.transpose(2, 0, 1)
        landmarks_ = np.array(landmarks_).ravel()

        # gender = np.expand_dims(np.array(gender),axis=0)
        age = np.expand_dims(np.array(((age-50.)/100.)),axis=0) # 归一化年龄
        return img_,landmarks_,gender,age
