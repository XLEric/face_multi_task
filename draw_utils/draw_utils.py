import random
import time
from collections import defaultdict
from PIL import Image,ImageDraw,ImageFont
import cv2
import numpy as np
import base64
def add_chinese(img_,str_,id):
    pil_img = cv2.cvtColor(img_,cv2.COLOR_BGR2RGB)#cv2和PIL中颜色的hex码的储存顺序不同，需转RGB模式
    pilimg = Image.fromarray(pil_img)#Image.fromarray()将数组类型转成图片格式，与np.array()相反
    draw = ImageDraw.Draw(pilimg)#PIL图片上打印汉字
    font = ImageFont.truetype("./cfg_font/simhei.ttf",20,encoding="utf-8")#参数1：字体文件路径，参数2：字体大小；Windows系统“simhei.ttf”默认存储在路径：C:\Windows\Fonts中
    draw.text((0,int(20*id)),str_,(255,0,0),font=font)
    img_ = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)#将图片转成cv2.imshow()可以显示的数组格
    return img_

def draw_person(img,data):
    for bbox in data:
        plot_box(bbox, img, color=(255,0,255), label="person", line_thickness=2)

def draw_bdudu_mask(img_,data):
    img_fusion = None
    if data['data'] is not None:
        pass
        res = data['data']['info']

        labelmap = base64.b64decode(res['labelmap'])    # res为通过接口获取的返回json
        nparr = np.fromstring(labelmap, np.uint8)
        labelimg = cv2.imdecode(nparr, 1)
        # width, height为图片原始宽、高
        labelimg = cv2.resize(labelimg, (img_.shape[1], img_.shape[0]), interpolation=cv2.INTER_NEAREST)
        im_new = np.where(labelimg==1, 255, labelimg)
        cv2.imwrite('./outputfile/mask.jpg', (im_new).astype(np.uint8))

        #
        color_map = (np.zeros((img_.shape[0], img_.shape[1],3))).astype(np.uint8)
        index = np.where(im_new == 255)# 获得对应分类的的像素坐标
        color_map[index[0], index[1],1] = 255

        img_fusion = cv2.addWeighted(img_, 1., color_map, 0.6, 0)

    return img_,img_fusion

def draw_bdudu_body_attr(img_,data):

    if data['data'] is not None:
        res_ = data['data']['info']
        for i in range(res_['person_num']):
            r_ = res_['person_info'][i]

            print(r_)
            r_attributes_age = r_['attributes']['age']['name']
            r_attributes_male = r_['attributes']['gender']['name']

            r_age_str = ''
            if r_attributes_male == '女性':
                r_male_str = 'woman'
            elif r_attributes_male == '男性':
                r_male_str = 'man'

            # 幼儿、青少年、青年、中年、老年
            # Young children, youth, middle age and old age
            r_age_str = ''
            if r_attributes_age == '幼儿':
                r_age_str = 'children'
            elif r_attributes_age == '青少年':
                r_age_str = 'teenagers'
            elif r_attributes_age == '青年':
                r_age_str = 'youth'
            elif r_attributes_age == '中年':
                r_age_str = 'middle'
            elif r_attributes_age == '老年':
                r_age_str = 'old'

            print(r_attributes_age)

            x1,y1,x2,y2 = int(r_['location']['left']),int(r_['location']['top']),int(r_['location']['left']+r_['location']['width']),int(r_['location']['top']+r_['location']['height'])

            cv2.rectangle(img_, (x1,y1), (x2,y2), (25,190,222), 2)

            cv2.putText(img_, '{} {}'.format(r_age_str,r_male_str), (x1,y1+25),cv2.FONT_HERSHEY_COMPLEX, 1.1, (0, 255, 0),9)
            cv2.putText(img_, '{} {}'.format(r_age_str,r_male_str), (x1,y1+25),cv2.FONT_HERSHEY_COMPLEX, 1.1, (0, 0, 255),3)


    return img_


def draw_bdudu_detect(img_,data):
    if data['data'] is not None:
        pass
        response = data['data']['info']


        x1,y1,x2,y2 = int(response['result']['left']),int(response['result']['top']),int(response['result']['left']+response['result']['width']),int(response['result']['left']+response['result']['height'])
        cv2.rectangle(img_, (x1,y1), (x2,y2), (25,190,222), 2)
    return img_

def draw_bdudu_object(img_,data):
    if data['data'] is not None:
        pass
        response = data['data']['info']


        print (response)

        for i in range(len(response['result'])):
            print("{} : {}".format(i+1,response['result'][i]))
            str_ = response['result'][i]['root'] +'-'+ response['result'][i]['keyword'] + '-' + '{:.2f}'.format(response['result'][i]['score'])
            img_ = add_chinese(img_,str_,i)
    return img_

def draw_bdudu_plant(img_,data):
    if data['data'] is not None:
        pass
        response = data['data']['info']


        print (response)

        for i in range(len(response['result'])):
            print("{} : {}".format(i+1,response['result'][i]))
            str_ = response['result'][i]['name'] + '-' + '{:.2f}'.format(response['result'][i]['score'])
            img_ = add_chinese(img_,str_,i)
    return img_

def draw_bdudu_animal(img_,data):
    if data['data'] is not None:
        pass
        response = data['data']['info']


        # print (response)

        for i in range(len(response['result'])):
            print("{} : {}".format(i+1,response['result'][i]))
            str_ = response['result'][i]['name'] + '-' + '{:.2f}'.format(float(response['result'][i]['score']))
            img_ = add_chinese(img_,str_,i)
    return img_

def draw_face(img_,detections):

    if detections is not None:

        if len(detections)>0:
            for m in detections:
                # print(m)
                plot_box(m["xyxy"], img_, label="face"+"{:.2f}".format(float(m["score"]))+" pyr:{}".format(m["euler_angle"]), color=(255,0,255))
                draw_global_contour(img_,m["landmarks"])
                # idx = 0
                # for k in m["landmarks"]:
                #     idx += 1
                #     for xy in  m["landmarks"][k]:
                #         x_,y_ = xy
                #         cv2.circle(img_, (int(x_),int(y_)), 1, rgbs[idx],-1)

BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]

# BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
#                       [11, 12], [12, 13]]
BODY_PARTS_PAF_IDS = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
                      [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27])


def draw_one_pose(img,keypoints,color_x = [255, 0, 0]):
    # assert keypoints.shape == (Pose.num_kpts, 2)

    color = [0, 224, 255]

    for part_id in range(len(BODY_PARTS_PAF_IDS) - 2-7):
        kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
        global_kpt_a_id = keypoints[kpt_a_id, 0]
        if global_kpt_a_id != -1:
            x_a, y_a = keypoints[kpt_a_id]
            cv2.circle(img, (int(x_a), int(y_a)), 3, color, -1)
        kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
        global_kpt_b_id = keypoints[kpt_b_id, 0]
        if global_kpt_b_id != -1:
            x_b, y_b = keypoints[kpt_b_id]
            cv2.circle(img, (int(x_b), int(y_b)), 3, color, -1)
        if global_kpt_a_id != -1 and global_kpt_b_id != -1:
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (255,60,60), 9)
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), color_x, 4)


def draw_pose(im0,data):

    for pose in data["data"]:
        bbox = pose['bbox']
        cv2.rectangle(im0, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 255, 0),3)

        cv2.putText(im0, 'idd: {}'.format(pose['id']), (int(bbox[0]), int(bbox[1]) - 16),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0),4)
        cv2.putText(im0, 'idd: {}'.format(pose['id']), (int(bbox[0]), int(bbox[1] - 16)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        draw_one_pose(im0,np.array(pose['keypoints']),(int(pose['color'][0]),int(pose['color'][1]),int(pose['color'][2])))


def draw_bbox(img,data):
    msg = data['data']['info']
    print('\n---------------->>> draw_bbox')
    colors = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, 200 + 1)][::-1]
    idx = 0
    for m in msg:
        print(m)
        plot_one_box
        xyxy = int(m['xyxy'][0]),int(m['xyxy'][1]),int(m['xyxy'][2]),int(m['xyxy'][3])
        label = "{}_{:.2f}".format(m['label'],m['score'])
        plot_one_box(xyxy, img, label=label, color=colors[idx])
        idx += 1
    # cv2.namedWindow('image',0)
    # cv2.imshow('image',img)
    # cv2.waitKey(800)

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def plot_box(bbox, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)# 目标的bbox
    if label:
        tf = max(tl - 2, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0] # label size
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3 # 字体的bbox
        cv2.rectangle(img, c1, c2, color, -1)  # label 矩形填充
        # 文本绘制
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255],thickness=tf, lineType=cv2.LINE_AA)

import random
rgbs = []
for j in range(100):
    rgb = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    rgbs.append(rgb)
def draw_global_contour(image,dict):


    x0,y0 = 0,0
    idx = 0
    for key in dict.keys():
        idx += 1
        # print(key)
        # _,_ = dict[key][0]

        if 'left_eye' == key:
            eye_x = np.mean([dict[key][i][0]+x0 for i in range(len(dict[key]))])
            eye_y = np.mean([dict[key][i][1]+y0 for i in range(len(dict[key]))])
            cv2.circle(image, (int(eye_x),int(eye_y)), 3, (255,255,55),-1)
        if 'right_eye' == key:
            eye_x = np.mean([dict[key][i][0]+x0 for i in range(len(dict[key]))])
            eye_y = np.mean([dict[key][i][1]+y0 for i in range(len(dict[key]))])
            cv2.circle(image, (int(eye_x),int(eye_y)), 3, (255,215,25),-1)

        if 'basin' == key or 'wing_nose' == key:
            pts = np.array([[dict[key][i][0]+x0,dict[key][i][1]+y0] for i in range(len(dict[key]))],np.int32)
            # print(pts)
            cv2.polylines(image,[pts],False,rgbs[idx],thickness = 2)

        else:
            points_array = np.zeros((1,len(dict[key]),2),dtype = np.int32)
            for i in range(len(dict[key])):
                x,y = dict[key][i]
                points_array[0,i,0] = x+x0
                points_array[0,i,1] = y+y0

            # cv2.fillPoly(image, points_array, color)
            cv2.drawContours(image,points_array,-1,rgbs[idx],thickness=2)

def get_bbox(img,pts,edge_expand = (0.,0.),flag_ave = False):
    print(pts)
    x_min = 65535
    y_min = 65535
    x_max = 0
    y_max = 0
    for i in range(len(pts)):
        x_min = min(pts[i][0],x_min)
        y_min = min(pts[i][1],y_min)
        x_max = max(pts[i][0],x_max)
        y_max = max(pts[i][1],y_max)

    expand_w = (x_max-x_min)*edge_expand[0]
    expand_h = (y_max-y_min)*edge_expand[1]

    if flag_ave:
        expand_w = min(expand_w,expand_h)
        expand_h = min(expand_w,expand_h)

    x_min -= expand_w
    y_min -= expand_h
    x_max += expand_w
    y_max += expand_h

    x_min = max(0,x_min)
    y_min = max(0,y_min)

    x_max = min((img.shape[1]-1),x_max)
    y_max = min((img.shape[0]-1),y_max)

    return (int(x_min),int(y_min),int(x_max),int(y_max))

def draw_skeleton(img,person_):
    colors = [(0,215,255),(255,115,55),(5,255,55),(25,15,255),(225,15,55),(225,115,55),(225,215,55),(0,0,255)]
    #
    cv2.line(img, (int(person_['body_parts']['right_wrist']['x']), int(person_['body_parts']['right_wrist']['y'])),\
     (int(person_['body_parts']['right_elbow']['x']), int(person_['body_parts']['right_elbow']['y'])), colors[0], 9)

    cv2.line(img, (int(person_['body_parts']['right_shoulder']['x']), int(person_['body_parts']['right_shoulder']['y'])),\
     (int(person_['body_parts']['right_elbow']['x']), int(person_['body_parts']['right_elbow']['y'])), colors[0], 9)

    cv2.line(img, (int(person_['body_parts']['neck']['x']), int(person_['body_parts']['neck']['y'])),\
      (int(person_['body_parts']['right_shoulder']['x']), int(person_['body_parts']['right_shoulder']['y'])), colors[0], 9)

    #
    cv2.line(img, (int(person_['body_parts']['left_wrist']['x']), int(person_['body_parts']['left_wrist']['y'])),\
      (int(person_['body_parts']['left_elbow']['x']), int(person_['body_parts']['left_elbow']['y'])), colors[1], 9)

    cv2.line(img, (int(person_['body_parts']['left_shoulder']['x']), int(person_['body_parts']['left_shoulder']['y'])),\
      (int(person_['body_parts']['left_elbow']['x']), int(person_['body_parts']['left_elbow']['y'])), colors[1], 9)

    cv2.line(img, (int(person_['body_parts']['neck']['x']), int(person_['body_parts']['neck']['y'])),\
      (int(person_['body_parts']['left_shoulder']['x']), int(person_['body_parts']['left_shoulder']['y'])), colors[1], 9)
    #

    cv2.line(img, (int(person_['body_parts']['neck']['x']), int(person_['body_parts']['neck']['y'])),\
      (int(person_['body_parts']['right_hip']['x']), int(person_['body_parts']['right_hip']['y'])), colors[2], 9)
    cv2.line(img, (int(person_['body_parts']['right_hip']['x']), int(person_['body_parts']['right_hip']['y'])),\
      (int(person_['body_parts']['right_knee']['x']), int(person_['body_parts']['right_knee']['y'])), colors[2], 9)
    cv2.line(img, (int(person_['body_parts']['right_knee']['x']), int(person_['body_parts']['right_knee']['y'])),\
      (int(person_['body_parts']['right_ankle']['x']), int(person_['body_parts']['right_ankle']['y'])), colors[2], 9)

    #
    cv2.line(img, (int(person_['body_parts']['neck']['x']), int(person_['body_parts']['neck']['y'])),\
      (int(person_['body_parts']['left_hip']['x']), int(person_['body_parts']['left_hip']['y'])), colors[3], 9)
    cv2.line(img, (int(person_['body_parts']['left_hip']['x']), int(person_['body_parts']['left_hip']['y'])),\
      (int(person_['body_parts']['left_knee']['x']), int(person_['body_parts']['left_knee']['y'])), colors[3], 9)
    cv2.line(img, (int(person_['body_parts']['left_knee']['x']), int(person_['body_parts']['left_knee']['y'])),\
      (int(person_['body_parts']['left_ankle']['x']), int(person_['body_parts']['left_ankle']['y'])), colors[3], 9)

    #
    cv2.line(img, (int(person_['body_parts']['neck']['x']), int(person_['body_parts']['neck']['y'])),\
      (int(person_['body_parts']['nose']['x']), int(person_['body_parts']['nose']['y'])), colors[4], 9)

    #
    cv2.line(img, (int(person_['body_parts']['nose']['x']), int(person_['body_parts']['nose']['y'])),\
      (int(person_['body_parts']['left_eye']['x']), int(person_['body_parts']['left_eye']['y'])), colors[5], 9)
    cv2.line(img, (int(person_['body_parts']['left_eye']['x']), int(person_['body_parts']['left_eye']['y'])),\
      (int(person_['body_parts']['left_ear']['x']), int(person_['body_parts']['left_ear']['y'])), colors[5], 9)
    #
    cv2.line(img, (int(person_['body_parts']['nose']['x']), int(person_['body_parts']['nose']['y'])),\
      (int(person_['body_parts']['right_eye']['x']), int(person_['body_parts']['right_eye']['y'])), colors[6], 9)
    cv2.line(img, (int(person_['body_parts']['right_eye']['x']), int(person_['body_parts']['right_eye']['y'])),\
      (int(person_['body_parts']['right_ear']['x']), int(person_['body_parts']['right_ear']['y'])), colors[5], 9)

    cv2.line(img, (int(person_['body_parts']['nose']['x']), int(person_['body_parts']['nose']['y'])),\
      (int(person_['body_parts']['top_head']['x']), int(person_['body_parts']['top_head']['y'])), colors[7], 9)


    pts_ = []
    pts_.append((person_['body_parts']['nose']['x'],person_['body_parts']['nose']['y']))
    pts_.append((person_['body_parts']['top_head']['x'],person_['body_parts']['top_head']['y']))
    pts_.append((person_['body_parts']['neck']['x'],person_['body_parts']['neck']['y']))
    pts_.append((person_['body_parts']['left_ear']['x'],person_['body_parts']['left_ear']['y']))
    pts_.append((person_['body_parts']['right_ear']['x'],person_['body_parts']['right_ear']['y']))

    face_bbox = get_bbox(img,pts_,edge_expand = (0.3,0.07),flag_ave = True)
    print('---->>>',face_bbox)
    cv2.rectangle(img, (face_bbox[0],face_bbox[1]), (face_bbox[2],face_bbox[3]), (25,190,222), 2)


    pts_ = []
    h_ = abs(person_['body_parts']['right_wrist']['y']-person_['body_parts']['right_elbow']['y'])
    w_ = abs(person_['body_parts']['right_wrist']['x']-person_['body_parts']['right_elbow']['x'])

    pts_.append((person_['body_parts']['right_wrist']['x']+w_,person_['body_parts']['right_wrist']['y']))
    pts_.append((person_['body_parts']['right_wrist']['x']+w_,person_['body_parts']['right_wrist']['y']))

    pts_.append((person_['body_parts']['right_wrist']['x']-w_,person_['body_parts']['right_wrist']['y']))
    pts_.append((person_['body_parts']['right_wrist']['x']-w_,person_['body_parts']['right_wrist']['y']))

    pts_.append((person_['body_parts']['right_wrist']['x'],person_['body_parts']['right_wrist']['y']-h_))
    pts_.append((person_['body_parts']['right_wrist']['x'],person_['body_parts']['right_wrist']['y']-h_))

    pts_.append((person_['body_parts']['right_wrist']['x'],person_['body_parts']['right_wrist']['y']+h_))
    pts_.append((person_['body_parts']['right_wrist']['x'],person_['body_parts']['right_wrist']['y']+h_))

    right_bbox = get_bbox(img,pts_,edge_expand = (0.02,0.02))
    print('---->>>',right_bbox)
    cv2.rectangle(img, (right_bbox[0],right_bbox[1]), (right_bbox[2],right_bbox[3]), (0,255,189), 2)


    pts_ = []
    h_ = abs(person_['body_parts']['left_wrist']['y']-person_['body_parts']['left_elbow']['y'])
    w_ = abs(person_['body_parts']['left_wrist']['x']-person_['body_parts']['left_elbow']['x'])

    pts_.append((person_['body_parts']['left_wrist']['x']+w_,person_['body_parts']['left_wrist']['y']))
    pts_.append((person_['body_parts']['left_wrist']['x']+w_,person_['body_parts']['left_wrist']['y']))

    pts_.append((person_['body_parts']['left_wrist']['x']-w_,person_['body_parts']['left_wrist']['y']))
    pts_.append((person_['body_parts']['left_wrist']['x']-w_,person_['body_parts']['left_wrist']['y']))

    pts_.append((person_['body_parts']['left_wrist']['x'],person_['body_parts']['left_wrist']['y']-h_))
    pts_.append((person_['body_parts']['left_wrist']['x'],person_['body_parts']['left_wrist']['y']-h_))

    pts_.append((person_['body_parts']['left_wrist']['x'],person_['body_parts']['left_wrist']['y']+h_))
    pts_.append((person_['body_parts']['left_wrist']['x'],person_['body_parts']['left_wrist']['y']+h_))

    left_bbox = get_bbox(img,pts_,edge_expand = (0.02,0.02))
    print('---->>>',left_bbox)
    cv2.rectangle(img, (left_bbox[0],left_bbox[1]), (left_bbox[2],left_bbox[3]), (0,255,189), 2)
def draw_bdudu_body(img_,data):
    frame_info=[]
    if data['data'] is not None:

        msg = data['data']['info']
        frame_info = msg['person_info']
        for i in range(msg['person_num']):
            person_ = msg['person_info'][i]



            print('{}) person {} : {}'.format(i+1,person_,person_.keys()))
            print()

            # bbox
            x,y,w,h,score = person_['location']['left'],person_['location']['top'],person_['location']['width'],person_['location']['height'],person_['location']['score']
            if float(score) <0.8:
                continue
            area = w*h/(img_.shape[0]*img_.shape[1])
            if area>0.9:
                frame_info= []
                continue

            flag_have_person = True


            cv2.putText(img_, "cof {:.2f}".format(float(score)), (int(x),int(y)-5),cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 0), 9)
            cv2.putText(img_, "cof {:.2f}".format(float(score)), (int(x),int(y)-5),cv2.FONT_HERSHEY_PLAIN, 1.6, (15, 55, 255), 2)

            cv2.rectangle(img_, (int(x),int(y)), (int(x+w),int(y+h)), (0,255,225), 2)
            # skeleton
            draw_skeleton(img_,person_)

            # key points
            for part_ in person_['body_parts']:
                print('pt:',part_,person_['body_parts'][part_])
                x,y,score = person_['body_parts'][part_]['x'],person_['body_parts'][part_]['y'],person_['body_parts'][part_]['score']
                cv2.circle(img_, (int(x),int(y)), 5, (255,0,30),-1)

    return img_,frame_info
