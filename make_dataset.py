import cv2
import sys
import string
import numpy as np
np.set_printoptions(threshold= np.inf)
import math


def get_mark_point(pic_num,dirr):
    #with open("C:\\Users\\SQY\\Desktop\\y\\"+str(pic_num)+".txt", "r") as f:
    with open(dirr, "r") as f:
        linenum = 0  # 表头两行不要
        point_mark_a = []
        point_mark_b = []
        point_mark_c = []
        point_mark_d = []
        #aaaa=list(f)
        #print(aaaa)
        for line in f.readlines():
            #line0 = line[0]
            if linenum != 0:
                linenum -= 1
                continue
            pointline = line.strip().split("\t")
            #pointline = line.strip()
            #print(list(pointline))
            if pointline[0] == "":
                break

            pointa_1 = []
            pointb_1 = []
            pointc_1 = []
            pointd_1 = []

            # a,b 是1和3对角线，c，d是2，4对角线
            pointa_1.append(int(float(pointline[0])))
            pointa_1.append(int(float(pointline[1])))
            pointb_1.append(int(float(pointline[4])))
            pointb_1.append(int(float(pointline[5])))

            pointc_1.append(int(float(pointline[2])))
            pointc_1.append(int(float(pointline[3])))
            pointd_1.append(int(float(pointline[6])))
            pointd_1.append(int(float(pointline[7])))


            '''
            pointa_1.append(int(float(pointline[0][:6])))
            pointa_1.append(int(float(pointline[1][:6])))
            pointb_1.append(int(float(pointline[2][:6])))
            pointb_1.append(int(float(pointline[3][:6])))
            '''

            point_mark_a.append(pointa_1)
            point_mark_b.append(pointb_1)

            point_mark_c.append(pointc_1)
            point_mark_d.append(pointd_1)

        # print(point_mark_a)
    print("获取坐标")
    return point_mark_a, point_mark_b,point_mark_c, point_mark_d

def draw_ellipse(ww,hh,img_dir, pic_num,txt_dir):
    global point_kuan1, distance, point_kuan2, ju_xing_chang, ju_xing_kuan
    pointa, pointb,pointc, pointd = get_mark_point(pic_num,txt_dir)
    # Gimg = np.zeros([3744, 5616, 1], np.uint8)
    Gimg = np.zeros([hh,ww, 1], np.uint8)  #高，宽
    print('目标个数  =  ',len(pointa))

    for i in range(0, len(pointa)):
        ellipse_size_scale = 0.52  # 控制椭圆大小0.51

        # 求出矩形和四个边的中心点
        center_point = (int((pointa[i][0] + pointb[i][0]) / 2), int((pointa[i][1] + pointb[i][1]) / 2))
        pointac = (((pointa[i][0] + pointc[i][0]) / 2), ((pointa[i][1] + pointc[i][1]) / 2))
        pointad=(((pointa[i][0] + pointd[i][0]) / 2),((pointa[i][1] + pointd[i][1]) / 2))
        pointbc = (((pointb[i][0] + pointc[i][0]) / 2), ((pointb[i][1] + pointc[i][1]) / 2))
        pointbd = (((pointb[i][0] + pointd[i][0]) / 2), ((pointb[i][1] + pointd[i][1]) / 2))

        # 判断矩形的长和宽
        distance_ac=int(np.sqrt(np.square(abs(pointa[i][0] - pointc[i][0])) + np.square(abs(pointa[i][1] - pointc[i][1]))))
        distance_ad = int(np.sqrt(np.square(abs(pointa[i][0] - pointd[i][0])) + np.square(abs(pointa[i][1] - pointd[i][1]))))
        if distance_ad>distance_ac:
            point_kuan1=pointac
            point_kuan2=pointbd
            ju_xing_chang=distance_ad
            ju_xing_kuan = distance_ac
            distance=distance_ad
        if distance_ad<distance_ac:
            point_kuan1 = pointad
            point_kuan2 = pointbc
            ju_xing_chang = distance_ac
            ju_xing_kuan = distance_ad
            distance = distance_ac

        #distance=int(np.sqrt(np.square(abs(pointac[0] - pointbd[0])) + np.square(abs(pointac[1] - pointbd[1]))))
        a = ellipse_size_scale *distance   # 半长轴

        #for n in range(0,Gimg.shape[1]):
            #for m in range(0,Gimg.shape[0]):
        min_h = min(pointa[i][1], pointb[i][1], pointc[i][1], pointd[i][1])
        max_h = max(pointa[i][1], pointb[i][1], pointc[i][1], pointd[i][1])
        min_w = min(pointa[i][0], pointb[i][0], pointc[i][0], pointd[i][0])
        max_w = max(pointa[i][0], pointb[i][0], pointc[i][0], pointd[i][0])
        for w in range(min_w-3,max_w+3):
            for h in range(min_h-3,max_h+3):
                point = (w, h) #txt中坐标存的是w,h
                PF1=int(np.sqrt(np.square(abs(point_kuan1[0] - point[0])) + np.square(abs(point_kuan1[1] - point[1]))))
                PF2 =int(np.sqrt(np.square(abs(point_kuan2[0] - point[0])) + np.square(abs(point_kuan2[1] - point[1]))))
                PF = PF1 + PF2
                if PF < (2*a+0) and w<ww and h<hh:
                    #print(ju_xing_chang,  ju_xing_kuan)
                    mean, sigma1,sigma2 = 0,  ju_xing_chang//2,  ju_xing_kuan//1    #0,15,15   # 0, 1
                    #mean, sigma1, sigma2 = 0, 9,6
                    #Gimg[m][n] = (0, int(pix_value), 0)
                    pix_value=255*np.exp(-1 * ((h-center_point[1])**2/(2 * (sigma1 ** 2))+(w-center_point[0])**2/(2 * (sigma2 ** 2))))
                    print(pix_value/255.0)
                    Gimg[h][w] = (int(pix_value))
            #print('%0.7f' %pix_value)
    Gimg = cv2.GaussianBlur(Gimg, (3,3), 0)  #(5,5)
    mask_dir = img_dir.replace('images', 'mask')
    cv2.imwrite(mask_dir, Gimg)
    print("mask完成_%d" % (pic_num ))

    # cv2.namedWindow("draw_ellipse_img",cv2.WINDOW_FREERATIO)
    # cv2.imshow("draw_ellipse_img", Gimg)
    # cv2.waitKey()
    return pic_num


def get_distance(point1, point2):
    return int(np.sqrt(np.square(abs(point1[0] - point2[0])) + np.square(abs(point1[1] - point2[1]))))
    pass


img_num = 1
import json
from PIL import Image, ImageFilter, ImageDraw
with open("C:\\Users\\SQY\\Desktop\\AOD.json", 'r') as outfile:
    train_list = json.load(outfile)
for i in range(0,len(train_list)):
    # image_dir = str(i+1) + ".jpg"
    #image_dir = "C:\\Users\\SQY\\Desktop\\MU\\" + str(i) + ".png"
    #gaussimg_dir = "C:\\Users\\SQY\\Desktop\\MUM\\" + str(i) + "_mask.png"
    img_dir = train_list[i]
    gaussimg_dir = img_dir.replace('.png', '.txt').replace('images', 'txt')
    # gaussimg_dir = "C:\\Users\\SQY\\Desktop\\cai\\" + str(i ) + "_mask.jpg"
    # mix_dir = "E:/database/data/2/" + "2_" + str(i ) + ".jpg"
    # img = cv2.imread("E:/database/data/" + image_dir)
    img = Image.open(img_dir).convert('RGB')
    print(img.size)
    w=img.size[0]
    h= img.size[1]
    draw_ellipse(w,h,img_dir, i,gaussimg_dir)  # 绘制高斯椭圆

