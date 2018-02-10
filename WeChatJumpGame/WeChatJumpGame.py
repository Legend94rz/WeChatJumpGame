import os
import matplotlib.pyplot as plt
import subprocess
#todo: 优化nextBatch的stack，并重构
#todo: 删除Data中不必要的图片，并优化initFileList
import cv2
import numpy as np
from Model import CoarseModel, FineModel
import argparse
import time
from LinearResidual import PressTimerCalculator

SCREENSHOT_WAY = 2
capFileName = 'cap.png'
chessFileName = 'chess-sm.png'
backupPreScreen = True
chessCenterOff = [25,125]
m1 = CoarseModel()
m2 = FineModel()

def getDistance(s,t):
    print(s,t)
    return np.linalg.norm(np.array(s)-np.array( t ));

def pull_screenshot():
    """
    获取屏幕截图，目前有 0 1 2 3 四种方法，未来添加新的平台监测方法时，
    可根据效率及适用性由高到低排序
    """
    if backupPreScreen and os.path.exists(capFileName):
        os.rename(capFileName,capFileName+'.bak')
    if 1 <= SCREENSHOT_WAY <= 3:
        process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
        binary_screenshot = process.stdout.read()
        binary_screenshot = binary_screenshot.replace(b'\r\n', b'\n')
        with open(capFileName, 'wb') as f:
            f.write(binary_screenshot)
    elif SCREENSHOT_WAY == 0:
        os.system('adb shell screencap -p /sdcard/' + capFileName)
        os.system('adb pull /sdcard/' + capFileName + ' .')

def preprocess(src):
    resolution = src.shape[:2]
    scale = resolution[1] / 720.
    src = cv2.resize(src, (720, int(src.shape[0] / scale)),interpolation = cv2.INTER_NEAREST)
    if src.shape[0] > 1280:
        s = (src.shape[0] - 1280) // 2
        src = src[s:(s + 1280) ,:,:]
    elif src.shape[0] < 1280:
        s1 = (1280 - src.shape[0]) // 2
        s2 = (1280 - src.shape[0]) - s1
        pad1 = 255 * np.ones((s1,720,3),dtype = np.uint8)
        pad2 = 255 * np.ones((s2,720,3),dtype = np.uint8)
        src = np.concatenate((pad1,src,pad2),0)
    return src

def findStartLocation(src,template):
    result = cv2.matchTemplate(src,template,cv2.TM_CCOEFF_NORMED)
    loc = cv2.minMaxLoc(result)
    s = loc[-1]
    return (s[0] + chessCenterOff[0],s[1] + chessCenterOff[1]), loc[1]

def findTargetLocation(src):
    out1 = m1.predict(np.expand_dims(src[320:-320,:,:], 0))
    coarPos = out1[0].astype(int)

    x1 = coarPos[0] - 160
    x2 = coarPos[0] + 160
    y1 = coarPos[1] - 160
    y2 = coarPos[1] + 160
    if y1 < 0:
        y1 = 0
        y2 = 320
    if y2 > src.shape[1]:
        y2 = src.shape[1]
        y1 = y2 - 320
    fineImg = src[x1:x2,y1:y2,:]
    out2 = m2.predict(np.expand_dims(fineImg,0))
    out2 = np.array([x1,y1]) + out2[0].astype(int)
    return (out2[1],out2[0])

def extractShoutcut(src,p,w,h):
    return src[p[1] - h // 2:p[1] + h // 2, p[0] - w // 2:p[0] + w // 2, :]

def getResidual(src, template):
    print(src.shape,template.shape)
    try:
        result = cv2.matchTemplate(src,template,cv2.TM_CCOEFF)
    except:
        return np.Infinity
    loc = cv2.minMaxLoc(result)
    p = loc[-1]
    s = (src.shape[1] // 2 , src.shape[0] // 2)                         #棋子位置
    t = (p[0] + template.shape[1] // 2, p[1] + template.shape[0] // 2)  #实际目标位置
    a = getDistance(s,t)
    if t[1] < s[1]:
        return -a
    return a

if __name__ == "__main__":
    template = cv2.imread(chessFileName)
    resm = PressTimerCalculator()
    I = 0
    m1.train()
    m2.train()
    while True:
        pull_screenshot()
        src = cv2.imread(capFileName)
        src = preprocess(src)
        pressX, pressY = int(0.8 * src.shape[0])+np.random.randint(-20,20), int(src.shape[1]//2)+np.random.randint(-20,20)
        s,confidence = findStartLocation(src,template)
        print('======================================')
        print('source loc: %r, confidence: %r' % (s,confidence))
        if confidence > 0.6:
            print('Jump: %d' % I)
            t = findTargetLocation(src)
            #todo hard code
            scS = extractShoutcut(src,s,320,300)
            if I > 0 and resm.canAdd():
                res = getResidual(scS,scT)
                #resm.add(getDistance(olds,oldt),res,oldtm)
            scT = extractShoutcut(src,t,180,200)
            tm = resm.predict(getDistance(s,t))
            os.system('adb shell input swipe %d %d %d %d %d' % (pressX,pressY,pressX,pressY,tm))
            olds,oldt,oldtm = s,t,tm
            I = I + 1
        else:
            pass
            #pressX,pressY = 564,1593
            #os.system('adb shell input tap %d %d ' % (pressX,pressY))
            #I = 0
            #resm = PressTimerCalculator()
        time.sleep(1)
