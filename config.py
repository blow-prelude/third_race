class CameraConfig:
    INDEX = 1
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

# 判断每个小方格的状态，一个是黑棋，一个是白棋
WHITE_CHESS_THRESH = 180
BLACK_CHESS_THRESH = 50