# -*- coding: utf-8 -*- 
import sys
import os
import V6UI
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import threading
import cv2
import numpy as np
import natsort
from PyQt5 import QtGui

start_folder = str()

class ThreadClass(QThread):
    changed = pyqtSignal(str)
    changed2 = pyqtSignal(str)
    totalprogress = pyqtSignal(int, int)
    imageprogress = pyqtSignal(int, int)

    def __init__(self, parent): 
        super(ThreadClass,self).__init__(parent)
        # self.parent = finlabel
    def run(self):
        global start_folder

        def search(dirname):
            print('start')
            now_directory_num = 0
            
            location = []
            
            for path, direct, files in os.walk(dirname):
                print('path:',path)
            
                if path.split("/")[-1] == 'Images' or path.split("/")[-1] == 'images' or path.split("\\")[-1] == 'Images' or path.split("\\")[-1] == 'images':
                    first_path = path.replace("\\",'/').split("/")
                    print('first_path:',first_path)
                    location.append(first_path)

                if path.split("/")[-1] == 'Full Image' or path.split("\\")[-1] == 'Full Image':
                    first_path = path.replace("\\",'/').split("/")
                    del location[-1]
                    location.append(first_path)

            location = natsort.natsorted(location)
            print(location)
            len_location = len(location)
            # print("len(location):",location)

            for se_path in location:
                folder = "/".join(se_path)
                self.changed2.emit('{0}'.format(folder))
                print('folder',folder)
                save_point = se_path
                if save_point[-1] == 'Full Image':
                    del save_point[-1:]
                    # print('del Full Image save_point',save_point)
                elif save_point[-1] == 'Images' or save_point[-1] == 'images':
                    pass
                save_point = "/".join(save_point)

                # print('save_point',save_point)
                # if os.path.isdir(save_point+'/Image_deconvolution') == False:
                #     aa = os.mkdir(save_point+'/Image_deconvolution')

                os.makedirs(save_point+'/Image_deconvolution', exist_ok = True)
                findfiles = save_point+'/Image_deconvolution'

                image_files = sorted(os.listdir(folder))
                for img in image_files:
                    if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png", "bmp"]:
                        image_files.remove(img)

                print(len(image_files))

                if image_files == []:
                    print("clean")
                    now_directory_num = now_directory_num + 1
                    self.totalprogress.emit(now_directory_num,len_location)

                else:
                    image_files = natsort.natsorted(image_files)
                    # print('image_files',image_files)
                    chk_image_files = image_files[0].split('_')
                    # if chk_image_files[-1] == chk_image_files[2]:
                    #     stackHDRs(image_files, folder, findfiles)
                
                    try:
                        chk_image_files[-1] != chk_image_files[2]
                    except IndexError:
                        now_directory_num = now_directory_num + 1
                    else:
                        if chk_image_files[-1] == chk_image_files[2]:
                            stackHDRs(image_files, folder, findfiles, now_directory_num, len_location)
                            now_directory_num = now_directory_num + 1


        def stackHDRs(image_files, folder, findfiles, now_directory_num, len_location):
            focusimages = []
            countnum = 0
            chknum = []
            firstset = image_files[0].split('_')

            first_num, sec_num = int(firstset[0]), int(firstset[1])

            for i in image_files:
                # s = copy_image_files.pop(0)
                s = i.split('_')

                if first_num == int(s[0]) and sec_num == int(s[1]):
                    countnum +=1
                    
                elif first_num == int(s[0]) and sec_num != int(s[1]):
                    chknum.append(countnum)
                    sec_num = int(s[1])
                    countnum = 1
                    
                elif first_num != int(s[0]):
                    chknum.append(countnum)
                    first_num = int(s[0])
                    sec_num = int(s[1])
                    countnum = 1

            chknum.append(countnum) #------------c#

            print('len(chknum): ',len(chknum))
            print('chknum: ',chknum)
            
            totalImage_number = len(chknum)

            for i in range(len(chknum)):

                self.totalprogress.emit(now_directory_num,len_location)
                self.imageprogress.emit(i,totalImage_number)
                self.changed.emit('Folder:{0}/{1} Image:{2}/{3}'.format(now_directory_num, len_location, i, totalImage_number))

                count = chknum.pop(0)
                a = image_files[:count:]
                number = str(a[0])#------------c#생략
                numbers = number.split('_')#------------c#생략
                finnumber = numbers[0]+'_'+numbers[1]#------------c#생략
                for ii in a:
                    focusimages.append(cv2.imdecode(np.fromfile(folder+"/{}".format(ii), np.uint8), cv2.IMREAD_COLOR))
                print('초기 세팅 이미지 수:', len(focusimages))
                merged = imagemaking(focusimages)
                mergedfin = cv2.medianBlur(merged, 3)
                # cv2.imshow('12', mergedfin)
                # cv2.waitKey(0)

                del(focusimages[:count:])
                del(image_files[:count:])
                
                is_success, im_buf_arr = cv2.imencode(".jpg",mergedfin)
                im_buf_arr.tofile((findfiles+'/'+str(finnumber)+ ".jpg"))
                print(str(finnumber))

        def imagemaking(focusimages):
            laps = []
            thrsum = []
            number_chk = 0

            for i in focusimages:
                thrsum.append(doLapchk(cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)))

            print('thrsum = ', thrsum)

            maxnum = max(thrsum)
            print('maxnum:', maxnum)

            savepoint = []

            if maxnum == 0:
                pass

            else:
                for ii in thrsum: 
                    k = int((ii*100)/maxnum)
                    savepoint.append(k)
                    nohun = sum(savepoint)-100
                    print('차이퍼센트: ',k)
                    if k <= int(nohun / len(thrsum)):
                        del(focusimages[number_chk])
                    else:
                        number_chk +=1
            
            hun = sum(savepoint)
            print("100프로 수치 삭제 : ", int(nohun / len(thrsum)))
            print("전체 수치 평균:", int(hun / len(thrsum)))
            del(savepoint)

            for i in range(len(focusimages)):
                laps.append(doLap(cv2.cvtColor(focusimages[i],cv2.COLOR_BGR2GRAY)))
            laps = np.asarray(laps)
            output = np.zeros(shape=focusimages[0].shape, dtype=focusimages[0].dtype)
            abs_laps = np.absolute(laps)
            maxima = abs_laps.max(axis=0)
            bool_mask = abs_laps == maxima
            mask = bool_mask.astype(np.uint8)
            
            for j in range(0,len(focusimages)):
                output = cv2.bitwise_not(focusimages[j],output, mask=mask[j])
            return 255-output

        def doLap(image):
            kernel_size = 5
            blur_size = 7

            smooth=cv2.GaussianBlur(image, (15, 15), 0, 0)
            w = image - smooth
            ww = image + w*2

            blurred = cv2.GaussianBlur(ww, (blur_size,blur_size), 0)
            return cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)

        # def doLap(image):
        #     kernel_size = 5
        #     blur_size = 7
        #     blurred = cv2.GaussianBlur(image, (blur_size,blur_size), 0)
        #     lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)

        #     total = 1900*1400
        #     totalsum = np.sum(lap)
        #     print('totalsum=',totalsum)
        #     print('B평균=',totalsum/total)
        #     print('B평균int=',int(totalsum/total))

        #     ret, showthr = cv2.threshold(lap, int(totalsum/total), 256, cv2.THRESH_TOZERO)
        #     return showthr

        def doLapchk(image):
        
            kernel_size = 5
            blur_size = 7
            # ret, thr = cv2.threshold(image, 100, 255, cv2.THRESH_TOZERO)
            # print('thr num:',sum(thr))
            # thr = cv2.resize(thr,(0,0),fx=0.3, fy=0.3)
            # cv2.imshow('1', thr+125)
            # cv2.waitKey(0)

            smooth=cv2.GaussianBlur(image, (15, 15), 0, 0)
            w = image - smooth
            ww = image + w*2

            blurred = cv2.GaussianBlur(ww, (blur_size,blur_size), 0)
            # blurred = cv2.GaussianBlur(image, (blur_size,blur_size), 0)
            lap = cv2.Laplacian(blurred, cv2.CV_8UC1, ksize=kernel_size)
            # print('Lap',lap)
            cv2.imshow('1', lap)
            cv2.waitKey(0)

            return np.sum(lap)
                        
        search(start_folder)
        self.imageprogress.emit(100,100)
        self.totalprogress.emit(100,100)
        self.changed.emit('Deconvolution Finish')

#1/384*100
class test1(QMainWindow, V6UI.Ui_Dialog):

    def __init__(self):
        super(test1,self).__init__()
        # self.threadclass = ThreadClass() #
        self.ThreadClass_start = ThreadClass(parent=self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.addfile_click)
        self.pushButton_2.clicked.connect(self.start_click)
        self.pushButton_3.clicked.connect(self.finish_click)
        self.ThreadClass_start.changed.connect(self.point)
        self.ThreadClass_start.changed2.connect(self.point2)
        self.ThreadClass_start.totalprogress.connect(self.totalprogress)
        self.ThreadClass_start.imageprogress.connect(self.imageprogress)
    def addfile_click(self):
        global start_folder

        start_folder = str(QFileDialog.getExistingDirectory(None, "Select Directory"))
        self.lineEdit.setText(start_folder)
        start_folder = start_folder + '/'
        return start_folder
    
    def start_click(self):
        global start_folder

        # thread = threading.Thread(target=focuss, args=())
        self.finlabel.setText("Deconvolution Start")
        self.ThreadClass_start.daemon = True
        self.ThreadClass_start.start()
        # ThreadClass_start.join()
        # self.finlabel.setText("Deconvolution Finish")

    def point(self, pointnum):
        self.finlabel.setText(pointnum)

    def point2(self, pointnum2):
        self.finlabel_2.setText(pointnum2)

    def totalprogress(self, setval,maxsetval):
        self.progressBar.setMaximum(maxsetval)
        self.progressBar.setValue(setval)
    
    def imageprogress(self, setval2,maxsetval2):
        self.progressBar_2.setMaximum(maxsetval2)
        self.progressBar_2.setValue(setval2)

    def finish_click(self):
        self.progressBar.setValue(0)
        self.progressBar_2.setValue(0)
        self.ThreadClass_start.terminate()
        # self.ThreadClass_start.stop()
        self.ThreadClass_start.quit()
        self.ThreadClass_start.wait()
        self.finlabel_2.setText("")
        self.finlabel.setText("Deconvolution Stop")
        # self.QCoreApplication.instance().quit

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = test1()
    myWindow.show()
    app.exec_()
