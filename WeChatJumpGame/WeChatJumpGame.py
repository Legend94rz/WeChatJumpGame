import os
import matplotlib.pyplot as plt
import subprocess
import cv2
import numpy as np
from Model import CoarseModel, FineModel

SCREENSHOT_WAY = 2
capFileName = 'cap.png'
chessFileName = 'chess.png'

def pull_screenshot():
    """
    获取屏幕截图，目前有 0 1 2 3 四种方法，未来添加新的平台监测方法时，
    可根据效率及适用性由高到低排序
    """
    global SCREENSHOT_WAY
    if 1 <= SCREENSHOT_WAY <= 3:
        process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
        binary_screenshot = process.stdout.read()
        binary_screenshot = binary_screenshot.replace(b'\r\n', b'\n')
        with open(capFileName, 'wb') as f:
            f.write(binary_screenshot)
    elif SCREENSHOT_WAY == 0:
        os.system('adb shell screencap -p /sdcard/'+capFileName)
        os.system('adb pull /sdcard/'+capFileName+' .')

def preprocess(src):
    resolution = src.shape[:2]
    scale = resolution[1] / 720.
    src = cv2.resize( src, (720, int(src.shape[0]/scale)),interpolation = cv2.INTER_NEAREST)
    if src.shape[0]>1280:
        s = (src.shape[0] - 1280) // 2
        src = src[ s:(s+1280) ,:,:]
    elif src.shape[0]<1280:
        s1 = (1280-src.shape[0])//2
        s2 = (1280-src.shape[0]) - s1
        pad1 = 255*np.ones((s1,720,3),dtype = np.uint8)
        pad2 = 255*np.ones((s2,720,3),dtype = np.uint8)
        src = np.concatenate((pad1,src,pad2),0)
    return src

def findStartLocation(src,template):
    result = cv2.matchTemplate(src,template,cv2.TM_CCOEFF)
    loc = cv2.minMaxLoc(result)
    return np.array([ loc[-1], (template.shape[1],template.shape[0]) ])

def findTargetLocation(src):
    m1 = CoarseModel()
    m1.train()
    m2 = FineModel()
    m2.train()

    out1 = m1.predict(np.expand_dims( src[320:-320], 0))
    coarPos = out1[0].astype(int)

    x1 = coarPos[0]-160
    x2 = coarPos[0]+160
    y1 = coarPos[1]-160
    y2 = coarPos[1]+160
    if y1<0:
        y1 = 0
        y2 =320
    if y2>src.shape[1]:
        y2 = src.shape[1]
        y1 = y2 - 320
    fineImg = src[x1:x2,y1:y2,:]
    out2 = m2.predict(np.expand_dims(fineImg,0))
    return np.array([x1,y1]) + out2[0].astype(int)

if __name__=="__main__":
    #pull_screenshot()
    template = cv2.imread(chessFileName)
    src = cv2.imread(capFileName)
    src = preprocess(src)
    s = findStartLocation(src,template)
    t = findTargetLocation(src)
    print(s,t)
    cv2.circle(src,tuple(s[0]),3,(255,0,0),-1)
    cv2.circle(src,tuple(t),3,(0,255,0),-1)

    cv2.namedWindow('1',cv2.WINDOW_NORMAL)
    cv2.imshow('1',src)
    cv2.waitKey(0)
    cv2.destroyWindow('1')
