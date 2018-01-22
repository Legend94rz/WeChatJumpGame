import os
import matplotlib.pyplot as plt
import subprocess
import cv2
import numpy as np
from Model import CoarseModel

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

def findStartLocation(src,template):
    result = cv2.matchTemplate(src,template,cv2.TM_CCOEFF)
    loc = cv2.minMaxLoc(result)
    return np.array([ loc[-1], (template.shape[1],template.shape[0]) ])

def findTargetLocation(src):
    return np.array([0,0])

if __name__=="__main__":
    '''
    pull_screenshot()
    template = cv2.imread(chessFileName)
    src = cv2.imread(capFileName)
    p = findStartLocation(src,template)

    print('find: %r,%r'%(p[0,0],p[0,1]))
    cv2.rectangle(src, tuple( p[0,:] ), tuple( np.sum(p,axis = 0) ),(255,0,0),thickness = 3)

    cv2.namedWindow('1',cv2.WINDOW_NORMAL)
    cv2.imshow('1',src)
    cv2.waitKey(0)
    cv2.destroyWindow('1')
    '''