
import json
import os
import cv2
import random
from draw_utils.draw_utils import draw_global_contour

path = "./wiki_crop_face/label/"
idx = 0
for f_ in os.listdir(path):
    f = open(path + f_, encoding='utf-8')#读取 json文件
    dict = json.load(f)
    f.close()
    idx += 1
    print("-------------------------Author : {} , {})  {}".format(dict["author"],idx,dict["path"]))
    img = cv2.imread(dict["path"])
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

    face_crop = img[y1_:y2_,x1_:x2_,:]
    # print("face_crop shape : ",face_crop.shape)
    M = cv2.getRotationMatrix2D((face_crop.shape[1]//2,face_crop.shape[0]//2), random.randint(-15,15), 1.0) #12
    rotated = cv2.warpAffine(face_crop, M, (face_crop.shape[1],face_crop.shape[0])) #13
    cv2.namedWindow("Rotated",0)
    cv2.imshow("Rotated", rotated) #14

    # for k in dict.keys():
    #
    #     print("{} : {}".format(k,dict[k]))

    cv2.namedWindow("result",0)
    cv2.imshow("result",img)
    cv2.waitKey(1)
