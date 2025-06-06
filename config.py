class CameraConfig:
    INDEX = 0
    WIDTH = 640
    HEIGHT = 480
class SerialConfig:
    INDEX = '/dev/ttyAMA0'
    TIMEOUT = 0.1
    


# 将原图像切分成左中右3块
LEFT_EDGE_RATIO = 1/4
RIGHT_EDGE_RATIO = 3/4

# canny边缘检测用到的参数
CANNY_THRESH1 = 40
CANNY_THRESH2 = 100

# 根据轮廓的面积判断是否是棋盘时，进行比较的面积阈值
MIN_AREA = 2000

# 判断棋子时设定的半径阈值
CHESS_RADIUS_MIN = 40
CHESS_RADIUS_MAX = 60


# 逼近多边形时用到的参数
EPSILON_RATIO = 0.01

# 判断每个小方格的状态
# SQUARE_LIGHTNESS_HIGH_THRESH = 140
# SQUARE_LIGHTNESS_LOW_THRESH = 95
SQUARE_GREEN_HIGH_THRESH = 140
SQUARE_GREEN_LOW_THRESH = 95
# SQUARE_RED_HIGH_THRESH = 130
# SQUARE_RED_LOW_THRESH = 90

# g通道和r通道的差值的阈值
SQUARE_G_DIFF_R_THRESH = 45

# 判断是否发生旋转的水平和竖直误差
# HOR_DEVIATION = 120
# VER_DEVIATION = 100