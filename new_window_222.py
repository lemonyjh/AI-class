import time
import serial  #导入串口模块
import serial.tools.list_ports

from collections import OrderedDict
import cv2
import argparse
import random
import torch
import numpy as np
from matplotlib import font_manager
my_font = font_manager.FontProperties(fname=r"c:\windows\fonts\msyh.ttc")

import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from utils.torch_utils import select_device, time_synchronized
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box2

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from new_window_ui_222 import Ui_MainWindow
from PyQt5.QtCore import Qt, QPointF, QDateTime,QTime
from PyQt5.QtGui import QPainter
from PyQt5 import QtCore, QtGui, QtWidgets


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        # 串口无效
        self.ser = None
        self.send_num = 0
        self.receive_num = 0
        self.model_init()#加载模型
        self.refresh()# 刷新一下串口的列表
        self.timer_video = QtCore.QTimer()  # 创建视频定时器
        self.timer_ckjs = QtCore.QTimer()  # 创建串口接受定时器
        self.timer_now = QtCore.QTimer()  # 创建时间显示定时器
        self.timer_now.start(1000)
        self.video=1
        self.ck = 1
        self.wdx=[]
        self.wdy=[]
        self.sdx=[]
        self.sdy=[]
        self.a=0
        # 显示发送与接收的字符数量
        # dis = '发送：' + '{:d}'.format(self.send_num) + '  接收:' + '{:d}'.format(self.receive_num)
        # self.statusBar.showMessage(dis)


        # 波特率控件
        self.comboBox_2.addItem('9600')
        self.comboBox_2.addItem('115200')
        self.comboBox_2.addItem('57600')
        self.comboBox_2.addItem('56000')
        self.comboBox_2.addItem('38400')
        self.comboBox_2.addItem('19200')
        self.comboBox_2.addItem('14400')
        self.comboBox_2.addItem('9600')
        self.comboBox_2.addItem('4800')
        self.comboBox_2.addItem('2400')
        self.comboBox_2.addItem('1200')

        self.pushButton_2.clicked.connect(self.open_ck)   #打开串口
        self.timer_ckjs.timeout.connect(self.ckjs)        #串口接收
        self.timer_now.timeout.connect(self.now_label)
        self.pushButton_class_1.clicked.connect(self.class_number)
        self.pushButton_class_2.clicked.connect(self.button_camera_open)
        self.timer_video.timeout.connect(self.show_video_frame)  # 定时器超时
        self.pushButton_4.clicked.connect(self.ckfs)   #发送数据
        self.pushButton_zxt_1.clicked.connect(self.zxt_show)
        self.pushButton_zxt_2.clicked.connect(self.zxt_not)


    def refresh(self):
        # 查询可用的串口
        plist = list(serial.tools.list_ports.comports())

        if len(plist) <= 0:
            self.comboBox.addItem("无可用串口")


        else:
            # 把所有的可用的串口输出到comboBox中去
            self.comboBox.clear()

            for i in range(0, len(plist)):
                plist_0 = list(plist[i])
                self.comboBox.addItem(str(plist_0[0]))

    def open_ck(self):
        if self.ck==1:
            self.pushButton_2.setText('关闭串口')
            try:
                # 打开串口，并且获得串口对象
                self.SCI1 = serial.Serial(self.comboBox.currentText(), int(self.comboBox_2.currentText()), timeout=0.5)
                # SCI1 = serial.Serial("com14", 9600, timeout=0.5)
                # 判断是否打开成功
                if (self.SCI1.isOpen() == True):
                    print("串口已经打开！")

                self.timer_ckjs.start(1000)
                self.ck=0
            except :
                print("串口打开异常:", exc)
        else:
            self.pushButton_2.setText('打开串口')
            self.timer_ckjs.stop()
            self.ck=1
            print("close")

    # def jieshou(self):
    def ckjs(self):
            # commandFromECU = 'r'  # 从键盘上输入一个命令
            # self.SCI1.write(str(commandFromECU).encode("utf-8"))  # 将键盘输入的控制命令发送给ECU上的单片机
            # 如果是读取数据，则从串口中接收数据。
            time.sleep(1)  # 等待1秒,等待接收ECU上单片机返回的数据. ECU会依次发送0x30-0x39等10个数据
            bufferSize = self.SCI1.in_waiting  # 接收ECU上单片机发过来的数据，并且返回数据的大小
            print("接收到", str(bufferSize), '个数据')

            data = self.SCI1.read_all().hex()  # 将接收缓冲区中数据读取到data中
            print(data)  # 将接收到的数据按照16进制打印出来
            if  bufferSize== 67 and data[-9]!='d':
                wd0 = data[-9]
                wd1 = data[-7]
                wd = wd0 + wd1
                sd0 = data[-5]
                sd1 = data[-3]
                sd = sd0 + sd1
                print(wd)
                print(sd)
                new = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())

                '''================================================================='''

                if self.a>0:
                    plt.cla()
                time_any = time.strftime("%H:%M:%S",time.localtime())
                print(time_any)
                wdy_1 = int(wd)
                self.wdx.append(time_any)
                self.wdy.append(wdy_1)
                print(self.wdx)
                print(self.wdy)
                sdy_1 = int(sd)
                self.sdx.append(time_any)
                self.sdy.append(sdy_1)
                print(self.sdx)
                print(self.sdy)


                plt.xlabel("时间", fontproperties=my_font)
                plt.ylabel("温度 单位（℃）\n 温度 单位（RH）", fontproperties=my_font)
                plt.title("温湿度折线图", fontproperties=my_font)
                plt.plot(self.wdx, self.wdy, label="温度",color='r')
                plt.plot(self.sdx, self.sdy, label="湿度",color='b')

                plt.legend(prop=my_font, loc=0)
                plt.savefig("img/wsdt.jpg")
                self.chartView_1.setPixmap(QtGui.QPixmap("img/wsdt.jpg"))
                self.chartView_1.setScaledContents(True)
                self.a=self.a+1
                print(self.a)

                '''=================================================================='''
                self.textEdit.append(new)
                self.textEdit.append('9553号温度值：'+wd+'℃'+'\t'+'9553号湿度值：'+sd+'RH')
                wd1 = wd
                self.listView_3_label_11.setText(wd1)
                wd2 = str(32+int(int(wd)*1.8))
                self.listView_3_label_21.setText(wd2)
                self.listView_4_label_11.setText(sd)
                wdj=int(wd)+50
                sdj=int(sd)
                self.wdj.setValue(wdj)
                self.sdj.setValue(sdj)
                if int(wd)>28:
                    self.yujing.setPixmap(QtGui.QPixmap("img/hot.png"))
                    self.yujing.setScaledContents(True)
                    self.yujing_1.setText("请开\n冷气")
                elif int(wd)<20:
                    self.yujing.setPixmap(QtGui.QPixmap("img/cold.png"))
                    self.yujing.setScaledContents(True)
                    self.yujing_1.setText("请开\n暖气")

                else:
                    self.yujing.clear()
                    self.yujing_1.clear()






    def ckfs(self):
        # commandFromECU = self.lineEdit.text()  # 从键盘上输入一个命令
        commandFromECU = 'g'
        print(commandFromECU)
        self.SCI1.write(commandFromECU.encode("utf-8"))

    def now_label(self):
        hour_min = time.strftime("%H:%M",time.localtime())
        self.listView_2_hour_min.setText(hour_min)
        mon_day = time.strftime("%m-%d", time.localtime())
        self.listView_2_mon_day.setText(mon_day)
        year = time.strftime("%Y", time.localtime())
        self.listView_2_year.setText(year)

    def class_number(self):
        number = self.lineEdit_class.text()  # 获取lineedit中的值
        self.listView_2_yingdao_label.setText(number)

    def zxt_show(self):
        self.chartView_1.setHidden(False)




    def zxt_not(self):
        self.chartView_1.setHidden(True)








    # 加载相关参数，并初始化模型
    def model_init(self):
        # 模型相关参数配置
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='source')
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        print(self.opt)
        # 默认使用opt中的设置（权重等）来对模型进行初始化
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size

        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'
        cudnn.benchmark = True
        self.model = attempt_load(weights, map_location=self.device)
        stride = int(self.model.stride.max())
        self.imgsz = check_img_size(imgsz, s=stride)
        if self.half:
            self.model.half()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        print("model initial done")

        # 目标检测
    def detect(self,  img):
        '''
        :param name_list: 文件名列表
        :param img: 待检测图片
        :return: info_show:检测输出的文字信息
        '''
        showimg = img
        with torch.no_grad():

            img = letterbox(img, new_shape=self.opt.img_size)[0]

            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t1 = time_synchronized()
            pred = self.model(img, augment=self.opt.augment)[0]

            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)

            h = 0
            for i, det in enumerate(pred):

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    i = 1
                    label = '%s %.2f' % (self.names[int(cls)], conf)


                    a = plot_one_box2(xyxy, showimg, label=label, color=self.colors[int(cls)],
                                                line_thickness=2)

                    h = h+1

        return h

    # 打开摄像头检测
    def button_camera_open(self):
        if self.video==1:
            print("Open camera to detect")
            # 设置使用的摄像头序号，系统自带为0
            camera_num = 0
            self.pushButton_class_2.setText('关闭监控')
            # 打开摄像头
            self.cap = cv2.VideoCapture(camera_num)

            # 判断摄像头是否处于打开状态
            bool_open = self.cap.isOpened()


            if not bool_open:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_video.start(100)  # 以ms为间隔，启动或重启定时器
                self.video = 0
        else:
            self.pushButton_class_2.setText('打开监控')
            self.timer_video.stop()
            self.cap.release()
            self.label_img.setPixmap(QtGui.QPixmap("img/test.jpg"))
            self.label_img.setScaledContents(True)
            self.video = 1


    def show_video_frame(self):
        flag, img = self.cap.read()
        if img is not None:

            # global fdata
            a = self.detect(img)  # 检测结果写入到原始img上
            sd=str(a)
            qq1=self.listView_2_yingdao_label.text()
            qq2=int(qq1)-int(a)
            qq=str(qq2)
            # 检测信息显示在界面
            self.listView_2_shidao_label.setText(sd)
            self.listView_2_queqin_label.setText(qq)
            show = cv2.resize(img, (640, 480))  # 直接将原始img上的检测结果进行显示
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label_img.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.label_img.setScaledContents(True)  # 设置图像自适应界面大小


        else:
            self.timer_video.stop()
            # 读写结束，释放资源
            self.cap.release()  # 释放video_capture资源







"""============================================================================================="""
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
