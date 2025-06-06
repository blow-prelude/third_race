
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,QWidget, QLabel,QTextEdit,QSizePolicy
from PyQt5.QtCore import Qt,QThread, pyqtSignal, pyqtSlot, QMutex
from PyQt5.QtGui import QImage, QPixmap

import numpy as np
import cv2
import sys

# 导入 config 模块
import config

import globals

# 开启多线程

# from concurrent.futures import ThreadPoolExecutor
# import asyncio


from typing import List, Tuple

# 导入随机库、
import random

# 导入串口库
# import serial

class CameraThread(QThread):

    # 定义一个信号，用来把处理完的 numpy.ndarray 帧发给主线程
    frame_ready = pyqtSignal(QImage)

    # 约定：1 表示“第一手玩家”赢，2 表示“另一个玩家”赢，0 表示平局
    game_over_sign = pyqtSignal(int)


    # 检测到作弊信号
    cheat_sign = pyqtSignal(bool)

    def __init__(self,parent=None):
        super().__init__(parent)

        # 用于控制线程循环
        self._running = True
        
        self._lock = QMutex()

         # 9个方格的中心点坐标     
        self.square_centers: List[Tuple[np.ndarray, int]] = []
        self.square_center :List[Tuple] = []

        # 该元组用于存储每个方格的状态，0，1，2分别为无，黑，白
        self.board_status : List[int] = [0] * 9
        
        # 保存的棋盘状态
        self.save_board_status = [0] * 9

        # 场外黑白棋的位置
        self.outside_black_chess_location : List[List[float]]= []
        self.outside_white_chess_location : List[List[float]]= []

          # 场外黑白棋的随机序列
        self.pick_black_index: List[int] = random.sample(list(range(0,5)), 5)
        self.pick_white_index: List[int] = random.sample(list(range(5,10)), 5)


         # 系统的下一次落点
        self.move_index = -1

        # 计算像素信息的平均值，每计算一次像素值就+1
        self.pixel_count = [0] * 9 

        self.mean_lightness_arr = np.zeros((9,10),dtype=np.float32)
        self.mean_g_arr = np.zeros((9,10),dtype=np.float32)
        self.mean_b_arr = np.zeros((9,10),dtype=np.float32)
        self.mean_r_arr = np.zeros((9,10),dtype=np.float32)

        
        # 胜利条件：行或列或对角线有3个相同的棋子
        self.win_conditions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # 行
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # 列
            (0, 4, 8), (2, 4, 6)  # 对角线
        ]
        
        # 作弊检测标志位
        self.start_cheat_detect = False



        # 初始化串口
        # self.ser = SerialInit()
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(config.CameraConfig.INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,config.CameraConfig.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,config.CameraConfig.HEIGHT)
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G')) 
        if not self.cap.isOpened():
            print('Cannot open camera!')
        # 用于控制线程循环
        self._running = True


    def stop(self):
        """外部调用，用来停止循环"""
        self._lock.lock()
        self._running = False
        self._lock.unlock()


 
    def run(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # 1. 预处理：灰度 + 闭操作 + Canny
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kernels = np.ones((5,5), dtype=np.uint8)
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernels, iterations=5)
            edges = cv2.Canny(closed, config.CANNY_THRESH1, config.CANNY_THRESH2)

            # 2. 计算左右边缘区域的 x 坐标
            left_edge = int(frame.shape[1] * config.LEFT_EDGE_RATIO)
            right_edge = int(frame.shape[1] * config.RIGHT_EDGE_RATIO)

            # 同步执行棋盘和盘外棋子的检测
            obgr = frame.copy()
            # self.detect_outside_chess(edges, left_edge, right_edge)
            board_info = self.detect_board(edges, left_edge, right_edge, obgr)

            # 4. 如果成功检测到棋盘，则检测棋盘上的棋子
            if board_info is not None:
                self.check_board(
                    obgr,
                    board_info['square_centers'],
                    board_info['square_radius']
                )

                # 判断棋盘是否旋转
                board_corners = board_info['board_corners']

                # 左上和右上如果水平上近似相等，就认为没有旋转
                if abs(board_corners[0][0]-board_corners[1][0]) <= config.HOR_DEVIATION :
                    globals.isrotate = False
                # 左上和右下如果竖直上近似相等，就认为旋转
                elif abs(board_corners[0][1]-board_corners[2][1]) <= config.VER_DEVIATION :
                    globals.isrotate = True

                # print(f'isrotate : {globals.isrotate}')


                # 如果用户已经下完棋子，那么系统开始规划下棋

                # 更完美的做法是电机回到原点后发送数据，接收到该信号后检测是否有赢家
                if globals.user_ready == True and globals.game_finished == False:
                
                    # 作弊检测
                    print('last status:',self.save_board_status)
                    print('current status:',self.board_status)
                    if self.start_cheat_detect == True:
                        # current_board_status = self.board_status
                        if self.save_board_status == self.board_status:
                            print('you are integrity!')
                            self.cheat_sign.emit(False)
                        else:
                            print('you have cheated ! !')
                            self.cheat_sign.emit(True)
                            # 检测哪些区域发生改变
                            detect_difference = [idx for idx,(current,save) in
                                                enumerate(zip(self.board_status,self.save_board_status))
                                                if current != save]

                            if len(detect_difference)==2:
                                for i in range(len(detect_difference)):
                                    # 获取原先不为空且发生改变的区域,需要将棋子运回该方格
                                    if self.save_board_status[i] != 0:
                                        cheat_start_idx = detect_difference[i]
                                        cheat_end_idx = detect_difference[1-i]
                                        print(f'recover from {cheat_end_idx} to {cheat_start_idx}\n')
                                        # self.ser.sendString(cheat_end_idx,cheat_start_idx,is_cheat=True)
                    print(self.board_status)
                    # 5. 检查赢家，如果胜负未分就用算法规划下一步的落子
                    self.winner = self.check_winner(self.board_status)
                    
                    if self.winner == -1:
                        print("game is not over yet!")
                        self.move_index = self.find_best_move(self.board_status)

                        # 用户先手，系统取白子
                        if globals.first_turn==1:
                            if self.pick_white_index:
                                pick_index = self.pick_white_index.pop(0)
                                print(f'[INFO] white from {pick_index} to {self.move_index}\n')
                                # self.ser.sendString(pick_index,self.move_index)
                                # 重新等待用户
                                globals.user_ready = False

                        # 系统先手，取黑子
                        elif globals.first_turn==2:
                            if self.pick_black_index:
                                pick_index = self.pick_black_index.pop(0)
                                print(f'[INFO] black from {pick_index} to {self.move_index}\n')
                                # self.ser.sendString(pick_index,self.move_index)
                                # 重新等待用户
                                globals.user_ready = False


                    elif self.winner != -1:
                        print("result has been sent!")
                        self.game_over_sign.emit(self.winner)
                        globals.game_finished = True            


            # 转为 QImage 并发射
            rgb = cv2.cvtColor(obgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_ready.emit(img)
            
        self.cap.release()
        


    # 计算给定的2点之间的欧几里得距离
    def compute_distance(self,pt1,pt2):
        return ((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**0.5
    


    # 识别盘外棋子
    # 在图像左右侧检测，如果外界矩形近似于正方形且边长在一定范围内,就认为检测到圆形
    # 然后根据左黑右白，分别把圆的中心点加入到列表中
    def detect_outside_chess(self,edges:np.ndarray, left_edge:int, right_edge:int) -> None:
        mask = np.zeros_like(edges)
        # 中间区域全为0,左右边缘为255
        mask[: , :left_edge] = 255
        mask[: , right_edge:] = 255
         # 在左右边缘找到棋子轮廓
        chess_cons = cv2.findContours(cv2.bitwise_and(edges,edges,mask),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        black_list = []
        white_list = []
        for con in chess_cons:
            x, y, w, h = cv2.boundingRect(con)
            chess_radius = min(w,h)/2
            if config.CHESS_RADIUS_MIN<=w<=config.CHESS_RADIUS_MAX and config.CHESS_RADIUS_MIN<=h<=config.CHESS_RADIUS_MAX:
                # 进行颜色判断,暂时不写

                # 左侧为黑棋，否则是白棋
                if x<=320:
                    black_list.append([x + chess_radius, y + chess_radius])
                else:
                    white_list.append([x + chess_radius, y + chess_radius])
                 # 更新成员变量
        self.outside_black_chess_location = black_list
        self.outside_white_chess_location = white_list
        # 打印数量
        # print(f'outside has {len(black_list)} black chess, {len(white_list)} white chess')



    # 异步协程：检测棋盘（中间区域），找到最大的四边形轮廓，计算出棋盘参数并返回
    # 返回值是字典类型，分别是小方格中心坐标，小方格半径，棋盘4个角点
    def detect_board(self, edges: np.ndarray, left_edge: int, right_edge: int, frame: np.ndarray):
        mask = np.zeros_like(edges)
        # 中间区域全为255,左右边缘为0
        mask[: , left_edge:right_edge] = 255

        # 面积最大的就是棋盘
        biggest_area : int = 0
        biggest = []
        board_cons = cv2.findContours(cv2.bitwise_and(edges,edges,mask),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        if board_cons is not None:
            for con in board_cons:
                area = cv2.contourArea(con)
                if area>=config.MIN_AREA and area>biggest_area:
                    biggest_area = area
                    biggest = con
        
        if len(biggest) != 0:
             # 使用最小外接矩形
            rect = cv2.minAreaRect(biggest)  # ((cx, cy), (w, h), angle)
            box = cv2.boxPoints(rect)        # 得到 4 个角点，顺时针
            box = np.array(box, dtype=np.float32)

            # 手动排序为：top-left, top-right, bottom-right, bottom-left
            # 方法：先按照 y 排序取出上两个点，再按 x 排序确定左右
            box_sorted = sorted(box, key=lambda p: (p[1], p[0]))
            top_points = box_sorted[:2]
            bottom_points = box_sorted[2:]

            if top_points[0][0] < top_points[1][0]:
                top_left, top_right = top_points
            else:
                top_right, top_left = top_points

            if bottom_points[0][0] < bottom_points[1][0]:
                bottom_left, bottom_right = bottom_points
            else:
                bottom_right, bottom_left = bottom_points

            # 再组合为有序四边形
            sorted_box = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


                # 宽高计算
            w1 = self.compute_distance(top_left, top_right)
            w2 = self.compute_distance(bottom_left, bottom_right)
            h1 = self.compute_distance(top_left, bottom_left)
            h2 = self.compute_distance(top_right, bottom_right)
            self.board_width = float((w1 + w2) / 2.0)
            self.board_height = float((h1 + h2) / 2.0)

            # 中心点
            self.board_center = (top_left + bottom_right) / 2.0

            # 单格内接圆半径
            self.square_radius = float(np.mean([self.board_width, self.board_height]) / 6.0)
            # print(f'square_radius = {self.square_radius}')

            # 棋盘角度（顺时针为负）
            delta = ((bottom_right - bottom_left)+(top_right - top_left)) / 2.0
            rotate_angle = np.arctan2(delta[1], delta[0])  # 弧度
            rotation_matrix = np.array([
                [np.cos(rotate_angle), -np.sin(rotate_angle)],
                [np.sin(rotate_angle),  np.cos(rotate_angle)]
            ], dtype=np.float32)

            # 9 格子圆心计算
            square_order = [
                (0, 0), (0, 1), (0, 2),
                (1, 0), (1, 1), (1, 2),
                (2, 0), (2, 1), (2, 2)
            ]
            square_centers = [0] * 9
            for row, col in square_order:
                rel_x = (2 * col - 2) * self.square_radius
                rel_y = (2 * row - 2) * self.square_radius
                rel_vec = np.array([rel_x, rel_y], dtype=np.float32)

                rotated = np.dot(rotation_matrix, rel_vec.reshape(2, 1)).flatten()
                abs_center = self.board_center + rotated
                square_centers[row * 3 + col] = abs_center

                # 可视化圆心
                cv2.circle(frame, (int(abs_center[0]), int(abs_center[1])), int(self.square_radius), (255, 0, 0), 2)

                cv2.putText(frame, str(row * 3 + col), 
                    (int(abs_center[0]), int(abs_center[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,255,0), 1)


            
            self.square_centers = square_centers

                # 将结果打包返回
            result = {
                    'square_centers': self.square_centers,
                    'square_radius': self.square_radius,
                    'board_corners': (top_left, top_right, bottom_right, bottom_left)
            }
            return result


    # 重置棋盘状态
    def reset_board_status(self):
        self.pick_black_index = random.sample(list(range(5)),5)
        self.pick_white_index = random.sample(list(range(5,10)),5)
        self.board_status = [0] * 9

    # 识别每个方格内是否有棋子，状态分别是0，1，2
    # 暂且先试试灰度图像的处理效果
    def check_board(self, bgr: np.ndarray, square_centers: List[Tuple[np.ndarray, int]], square_radius: float):
        
        for seq , center in enumerate(square_centers):
            # if 0 <= seq-1 < len(self.board_status):
                # 创建掩膜：一个和 gray 一样大小的全零矩阵
                h, w = bgr.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask,(int(center[0]), int(center[1])),int(square_radius), 1, -1)

                # 计算该圆内的平均亮度
                mean_b = cv2.mean(bgr, mask=mask)[0]
                mean_g = cv2.mean(bgr, mask=mask)[1]
                mean_r = cv2.mean(bgr, mask=mask)[2]
                mean_lightness = 0.114 * mean_b + 0.587 * mean_g + 0.299 * mean_r
            
                self.mean_lightness_arr[seq][self.pixel_count[seq-1]] = mean_lightness
                self.mean_g_arr[seq][self.pixel_count[seq]] = mean_g
                self.mean_b_arr[seq][self.pixel_count[seq]] = mean_b
                self.mean_r_arr[seq][self.pixel_count[seq]] = mean_r
                self.pixel_count[seq] += 1  # seq从1开始，所以seq-1
                

            
                # 每检测10次就进行一次判断
                if self.pixel_count[seq] >= 10:
                    avarage_lightness = np.mean(self.mean_lightness_arr[seq-1])
                    avarage_g = np.mean(self.mean_g_arr[seq])
                    avarage_b = np.mean(self.mean_b_arr[seq])
                    avarage_r = np.mean(self.mean_r_arr[seq])
                    # print(f' seq {seq} lightness : {avarage_lightness:.2f}, g : {avarage_g:.2f}, b : {avarage_b:.2f}, r : {avarage_r:.2f}')
                    self.pixel_count[seq] = 0
                    # 如果亮度大于阈值，认为是白棋
                    if avarage_r >= config.SQUARE_RED_HIGH_THRESH and avarage_g >= config.SQUARE_GREEN_HIGH_THRESH:
                        self.board_status[seq] = 2
                        # print(f'seq {seq} is white chess (2)')

                    # 如果亮度小于阈值且g通道偏小，认为是黑棋
                    elif avarage_r <= config.SQUARE_RED_LOW_THRESH and avarage_g <= config.SQUARE_GREEN_LOW_THRESH:
                        self.board_status[seq] = 1
                        # print(f'seq {seq} is black chess (1)')

                    # 否则认为是空格
                    else:
                        self.board_status[seq] = 0
                        # print(f'seq {seq} is no chess (0)')


    # 检查赢家
    # 如果有赢家，返回赢家的编号1或2
    # 如果平局，返回0
    # 如果没有下完，返回-1
    def check_winner(self, board_status):
        
        for a, b, c in self.win_conditions:
            if len(self.board_status) == 9:
                if board_status[a] == board_status[b] == board_status[c] != 0:
                    return board_status[a]
        # 所有方框都被填满，平局
        if 0 not in board_status:
            return 0
        return -1



    # 评估当前局面,如果有赢家，返回10或-10
    # 如果没有赢家，会尽量获胜或阻止对手获胜
    def evaluate(self, board_status):
        winner = self.check_winner(board_status)
        if winner == globals.first_turn:
            return -10
        elif winner == 3 - globals.first_turn:
            return 10

        score = 0
        for (a, b, c) in self.win_conditions:
            line = [board_status[a], board_status[b], board_status[c]]
            # 如果这一条线对方占2个空一个，-4分
            if line.count(globals.first_turn) == 2 and line.count(0) == 1:
                score -= 4
            # 如果这一条线我方占2个空一个，+5分
            if line.count(3 - globals.first_turn) == 2 and line.count(0) == 1:
                score += 5

        return score


    # minimax算法
    # alpha-beta剪枝
    # 传入当前棋盘状态，深度，alpha，beta，是否是最大化玩家
    # 返回当前局面对自己的评分
    def minimax(self, board_status, depth, alpha, beta, is_maximizing):
        winner = self.check_winner(board_status)
        if winner != -1:
            if winner == globals.first_turn:
                return -10 + depth  # 加depth是为了让AI优先选择较快的胜利或较晚的失败
            elif winner == 3 - globals.first_turn:
                return 10 - depth
            else:  # 平局
                return 0

        if is_maximizing:
            best = -float('inf')
            for i in range(9):
                if board_status[i] == 0:
                    board_status[i] = 3 - globals.first_turn
                    best = max(best, self.minimax(board_status, depth + 1, alpha, beta, False))
                    board_status[i] = 0
                    alpha = max(alpha, best)
                    if beta <= alpha:
                        break
            return best
        else:
            best = float('inf')
            for i in range(9):
                if board_status[i] == 0:
                    board_status[i] = globals.first_turn
                    best = min(best, self.minimax(board_status, depth + 1, alpha, beta, True))
                    board_status[i] = 0
                    beta = min(beta, best)
                    if beta <= alpha:
                        break
            return best



    # 遍历所有可能的落子点，返回评分最高的点
    def find_best_move(self, board_status):
        best_val = -float('inf')
        move_index = -1

        for i in range(9):
            # 在每个空的方框内假设自己落子
            # 然后用minimax算法模拟对手反应以及后续的可能性
            if board_status[i] == 0:
                board_status[i] = 3 - globals.first_turn
                move_val = self.minimax(board_status, 0, -float('inf'), float('inf'), False)
                board_status[i] = 0

                # 如果当前评分更好，或者在九宫格中心，就更新最优落点
                if move_val > best_val or (move_val == best_val and i == 4):
                    move_index = i
                    best_val = move_val

        return move_index



    @pyqtSlot()
    def on_save_board(self):
        self.save_board_status = self.board_status.copy()
        print('board_status has been saved! ', self.save_board_status)

    @pyqtSlot(bool)
    def on_cheat_detect_status(self, status):
        self.start_cheat_detect = status





       



# 3个子窗口的共性功能
class BaseFunctionWindow(QMainWindow):
    def __init__(self, main_window, title):
        super().__init__()
        # self.title = title
        self.main_window = main_window
        self.setWindowTitle(title)
        self.resize(640, 480)
        self.title = title

        # 初始化窗口
        # self.ser = SerialInit()


        # 返回按钮
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self.return_to_main_window)
        self.back_button.setFixedSize(100,50)

        # 将返回按钮设置在界面右侧

    def return_to_main_window(self):
        """返回主界面"""
        self.main_window.show()
        self.close()


# 主窗口，中间竖直排列3个按钮，点击后进入不同的功能界面
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(600,400)

        # 在主界面竖直居中排列3个button

        # 分别创建3个按钮
        self.button1 = QPushButton("deploy", self)
        self.button1.clicked.connect(lambda: self.switch_to_deploy(self.deploy_window))
        self.button1.setFixedSize(200, 50)
        # self.button1.setStyleSheet("background-color: white; color: black; font-size: 20px;")

        self.button2 = QPushButton("first_gamer", self)
        self.button2.clicked.connect(lambda: self.switch_to_first(self.first_gamer_window))
        self.button2.setFixedSize(200, 50)

        self.button3 = QPushButton("second_gamer", self)
        self.button3.clicked.connect(lambda: self.switch_to_second(self.second_gamer_window))
        self.button3.setFixedSize(200, 50)

        # 创建主界面布局
        layout = QVBoxLayout()
        layout.addWidget(self.button1)  
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)

        # 设置主界面布局
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


        # 创建子界面
        self.deploy_window = DeployPage(self, "delpoy")
        self.first_gamer_window = Gamer(self, "first_gamer")
        self.second_gamer_window = Gamer(self, "second_gamer")


        # 启动摄像头线程并绑定信号 
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.first_gamer_window.update_image)
        self.camera_thread.frame_ready.connect(self.second_gamer_window.update_image)
        self.camera_thread.game_over_sign.connect(self.first_gamer_window.on_game_over)
        self.camera_thread.game_over_sign.connect(self.second_gamer_window.on_game_over)
        self.camera_thread.cheat_sign.connect(self.first_gamer_window.on_cheat_found)
        self.camera_thread.cheat_sign.connect(self.second_gamer_window.on_cheat_found)

        self.first_gamer_window.save_board_signal.connect(self.camera_thread.on_save_board)
        self.second_gamer_window.save_board_signal.connect(self.camera_thread.on_save_board)
        self.first_gamer_window.start_cheat_detect_signal.connect(self.camera_thread.on_cheat_detect_status)
        self.first_gamer_window.start_cheat_detect_signal.connect(self.camera_thread.on_cheat_detect_status)
        # 启动run（）方法
        self.camera_thread.start()



    def switch_to_deploy(self, target_window):
        """切换到deploy窗口"""
        target_window.show()
        self.hide()

    def switch_to_first(self, target_window):
        """切换到first窗口"""
        # 用户先手，执黑棋
        globals.first_turn = 1
        # 系统执白棋，初始化取白棋的列表
        self.camera_thread.reset_board_status()
        # 重新开始游戏
        globals.game_finished = False

        target_window.show()
        self.hide()


    def switch_to_second(self, target_window):
        """切换到second窗口"""
        # 系统先手
        globals.first_turn = 2
        self.camera_thread.reset_board_status()
        # 重新开始游戏
        globals.game_finished = False

        target_window.show()
        self.hide()


    def closeEvent(self, event):
        """
        关闭主窗口时，一定要先停止摄像头线程，再调用父类 closeEvent
        """
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread.wait()
        super().closeEvent(event)



# 自定义放置棋子界面
class DeployPage(BaseFunctionWindow):
    def __init__(self, main_window, lecture_title):
        super().__init__(main_window, lecture_title)


         # 当前选中落子模式： 0=none, 1=black, 2=white
        self.current_mode = 0

        # --------- 1. 左侧：3x3 格子区域 --------- #
        # 使用一个 3x3 的 QPushButton 矩阵
        self.grid_buttons: List[List[QPushButton]] = []
        grid_widget = QWidget()
        grid_layout = QVBoxLayout()  # 先垂直包一层，再在内部用 3 个横排布局
        grid_layout.setSpacing(5)
        grid_layout.setContentsMargins(5, 5, 5, 5)

        for row in range(3):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(5)
            temp_row: List[QPushButton] = []
            for col in range(3):
                btn = QPushButton("")  # 初始文本为空
                btn.setFixedSize(100, 100)
                btn.setStyleSheet("font-size: 24px;")  # 字号稍大
                btn.clicked.connect(self.make_cell_click_handler(row, col))
                temp_row.append(btn)
                row_layout.addWidget(btn)
            self.grid_buttons.append(temp_row)
            grid_layout.addLayout(row_layout)

        grid_widget.setLayout(grid_layout)

        # --------- 2. 右侧：功能按钮区域 --------- #
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # 2.1 Black 按钮
        self.black_button = QPushButton("Black")
        self.black_button.setFixedSize(100, 40)
        self.black_button.clicked.connect(self.on_black_clicked)
        right_layout.addWidget(self.black_button, alignment=Qt.AlignTop)

        # 2.2 White 按钮
        self.white_button = QPushButton("White")
        self.white_button.setFixedSize(100, 40)
        self.white_button.clicked.connect(self.on_white_clicked)
        right_layout.addWidget(self.white_button, alignment=Qt.AlignTop)

        # 2.3 Deploy 按钮
        self.deploy_button = QPushButton("Deploy")
        self.deploy_button.setFixedSize(100, 40)
        self.deploy_button.clicked.connect(self.on_deploy_clicked)
        right_layout.addWidget(self.deploy_button, alignment=Qt.AlignTop)

        # 2.4 Reset 按钮
        self.reset_button = QPushButton("Reset")
        self.reset_button.setFixedSize(100, 40)
        self.reset_button.clicked.connect(self.on_reset_clicked)
        right_layout.addWidget(self.reset_button, alignment=Qt.AlignTop)

        # 2.5 Back 按钮（继承自 BaseFunctionWindow）
        #    BaseFunctionWindow 已经创建了 self.back_button 并连接 return_to_main_window()
        self.back_button.setText("Back")
        self.back_button.setFixedSize(100, 40)
        right_layout.addWidget(self.back_button, alignment=Qt.AlignBottom)

        # 把右侧布局设置到 right_widget
        right_widget.setLayout(right_layout)

        # --------- 3. 整体布局：左右并排 --------- #
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(grid_widget, stretch=2)
        main_layout.addWidget(right_widget, stretch=1)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def make_cell_click_handler(self, row: int, col: int):
        """
        返回一个槽函数引用，绑定到 grid_buttons[row][col] 的 clicked 信号上。
        """
        def handler():
            if self.current_mode not in (1, 2):
                # 没有选中 Black/White 模式时，点击不做任何事
                return

            btn = self.grid_buttons[row][col]
            # # 如果该格子已经被标记，就不允许再重复标
            # if btn.text() != "":
            #     return

            # 根据模式在按钮上标记 “1” 或 “2”
            if self.current_mode == 1:
                btn.setText("1")
                btn.setEnabled(False)  # 禁止重复点击
            elif self.current_mode == 2:
                btn.setText("2")
                btn.setEnabled(False)

            # 标记完毕后，你也可以自动把 current_mode 置为 0，让用户每次都要重复按 Black/White
            # 这取决于你的需求。若要保留模式，可注释下面一行：
            # self.clear_mode_selection()

        return handler

    def on_black_clicked(self):
        """
        进入“黑棋落子”模式：按钮变暗，White 解暗
        """
        self.current_mode = 1
        # 把 Black 字暗一点，White 恢复默认
        self.black_button.setStyleSheet("background-color: lightgray;")
        self.white_button.setStyleSheet("")

    def on_white_clicked(self):
        """
        进入“白棋落子”模式：按钮变暗，Black 解暗
        """
        self.current_mode = 2
        self.white_button.setStyleSheet("background-color: lightgray;")
        self.black_button.setStyleSheet("")

    def clear_mode_selection(self):
        """
        将 Black/White 的“暗色”恢复到默认，current_mode 置 0
        """
        self.current_mode = 0
        self.black_button.setStyleSheet("")
        self.white_button.setStyleSheet("")

    def on_reset_clicked(self):
        """
        Reset：清空所有格子上的标记，并恢复 Black/White 按钮状态
        """
        for row in range(3):
            for col in range(3):
                btn = self.grid_buttons[row][col]
                btn.setText("")
                btn.setEnabled(True)
        self.clear_mode_selection()

     # 通过串口把黑白棋落点(序号)发送出去
    def on_deploy_clicked(self):

        move_black_cmd : List = []
        move_white_cmd : List = []
        
        # TODO: 串口发送逻辑放在这里

        # 获得落子的序号
        black_idx, white_idx = self.get_grid_info()
        count_black_chess = len(black_idx)
        count_white_chess = len(white_idx)

        print(f"total {count_black_chess} black chess ,they are in {black_idx}")
        print(f"total {count_white_chess} black chess ,they are in {white_idx}")



        # 从 [0,1,2,3,4] 中随机取不重复编号用于“取子编号”
        pick_black_src = random.sample(range(5), count_black_chess)
        pick_white_src = random.sample(range(5), count_white_chess)

        # black_messages: List[str] = [
        #     f"from {src} to {dst}\n" for src, dst in zip(pick_black_src, black_idx)
        # ]
        # white_messages: List[str] = [
        #     f"from {src} to {dst}\n" for src, dst in zip(pick_white_src, white_idx)
        # ]

        # # 串口发送
        # for msg in black_messages:
        #     print(f"发送黑棋指令: {msg.strip()}")
        #     # self.ser.ser.write(msg.encode())
        #
        # for msg in white_messages:
        #     print(f"发送白棋指令: {msg.strip()}")
        #     # self.ser.ser.write(msg.encode())


        # 发送坐标序号
        move_black_cmd = [(src,dst) for src,dst in zip(pick_black_src, black_idx)]
        for src, dst in move_black_cmd:
            print(f"move black from {src} to {dst}\n")
            # self.ser.sendString(src,dst)

        move_white_cmd = [(src, dst) for src, dst in zip(pick_white_src, white_idx)]
        for src, dst in move_white_cmd:
            print(f"move white from {src} to {dst}\n")
            # self.ser.sendString(src,dst)



    # 获得棋盘上分别有个黑棋白棋以及他们的位置
    def get_grid_info(self) -> Tuple[List[int], List[int]]:
        """
        遍历 3x3 按钮矩阵，返回两个列表   
        九宫格序号定义：从左到右、从上到下依次为 0,1,2,3,4,5,6,7,8
        """
        black_idx = []
        white_idx = []
        for row in range(3):
            for col in range(3):
                idx = row * 3 + col
                text = self.grid_buttons[row][col].text().strip()
                if text == "1":
                    black_idx.append(idx)
                elif text == "2":
                    white_idx.append(idx)
        return black_idx, white_idx
       
    # BaseFunctionWindow 已经把 return_to_main_window() 连接到 self.back_button
    # 这里不需要 override closeEvent，除非你在 DeployPage 里有其他额外资源需要释放。




# 人机对弈界面
class Gamer(BaseFunctionWindow):

    # save被按下发送该信号，保存棋盘状态
    save_board_signal = pyqtSignal()
     # 开启作弊检测
    start_cheat_detect_signal = pyqtSignal(bool)  

    def __init__(self, main_window, lecture_title):
        super().__init__(main_window, lecture_title)
        # self.lecture_title = lecture_title
        # save按键是否被点击，即是否开启反作弊模式
        self.cheat_mode_on = False

        # 界面左侧放一个 QLabel，用来显示视频帧 
        self.video_label = QLabel()
        # 给一个初始大小，后面可以根据需要调整
        self.video_label.setFixedSize(320, 240)
        self.video_label.setStyleSheet("background-color: black;")

         # ———————— 2. 右侧：信息区（状态 + 作弊 + Save + Back） ———————— #

        # 2.1 状态区：用一个只读的 QTextEdit 或 QLabel 也可以
        self.status_area = QTextEdit()
        self.status_area.setReadOnly(True)
        self.status_area.setFixedHeight(80)
        self.status_area.setPlaceholderText("Status: ")
        #（以后可以用 self.status_area.append("your turn") 之类更新）

        # 2.2 作弊提示区：用 QLabel 即可
        self.cheat_area = QLabel("Cheat Alert: ")
        self.cheat_area.setFixedHeight(40)
        self.cheat_area.setStyleSheet("color: red;")  # 用红色文字提示
        self.cheat_area.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # 2.3 Save 按钮（槽函数先占位）
        self.save_button = QPushButton("Save")
        self.save_button.setFixedSize(100, 40)
        self.save_button.clicked.connect(self.save_board)  # 槽函数下面再定义

        # 2.4 ready 按钮
        self.ready_button = QPushButton("Ready")
        self.ready_button.setFixedSize(100, 40)
        self.ready_button.clicked.connect(self.user_have_ready)

        # 2.5 Back 按钮（继承自 BaseFunctionWindow，已经在 __init__ 里创建好）
        # 这里我们设置它的大小，并且放到最底部
        self.back_button.setFixedSize(100, 40)

        # 2.6 把 2.1、2.2、2.3、2.4 2.5 按纵向顺序放到一个 QVBoxLayout 里
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.status_area)
        right_layout.addWidget(self.cheat_area)
        right_layout.addStretch(1)              
         # 在按钮和状态区之间占个“弹性”空白，让按钮靠底下
        right_layout.addWidget(self.save_button, alignment=Qt.AlignRight)
        right_layout.addWidget(self.ready_button, alignment=Qt.AlignRight)
        right_layout.addWidget(self.back_button, alignment=Qt.AlignRight)

        # 用一个 container 包裹右侧区域
        right_container = QWidget()
        right_container.setLayout(right_layout)

        # ———————— 3. 总布局：左右并排 ———————— #
        content_layout = QHBoxLayout()
        content_layout.addWidget(self.video_label, stretch=2)
        content_layout.addWidget(right_container, stretch=1)

        # ———————— 4. 整个窗口的 central 布局 ———————— #
        container = QWidget()
        container.setLayout(content_layout)
        self.setCentralWidget(container)

        

    @pyqtSlot(QImage)
    def update_image(self, img: QImage):
        # print(f"{self.lecture_title} received frame") 
        self.video_label.setPixmap(QPixmap.fromImage(img))


    # 接收整数的槽，即接收最终赢家的信号
    @pyqtSlot(int)
    def on_game_over(self, winner: int):
        if winner == 0:
            self.status_area.append("Draw!")
        
        else:
            if winner == 3 - globals.first_turn:
                self.status_area.append("you lose!")
            else:
                self.status_area.append("you win!")

    # 接收作弊信号
    @pyqtSlot(bool)
    def on_cheat_found(self,cheat_flag: bool):
        if cheat_flag:
            self.cheat_area.setText("Cheat Alert: You cheated!!")

    # 保存当前棋盘的状态
    def save_board(self):
       # 切换状态
        self.cheat_mode_on = not self.cheat_mode_on

        if self.cheat_mode_on:
            self.save_button.setStyleSheet("background-color: gray; color: white;")
            self.save_button.setText("Cheat ON")
            self.save_board_signal.emit()  # 发出保存信号
            self.start_cheat_detect_signal.emit(True)
        else:
            self.save_button.setStyleSheet("")
            self.save_button.setText("Save")
            self.start_cheat_detect_signal.emit(False)


    # 表示用户已经下完棋，此时系统可以开始规划如何下棋
    def user_have_ready(self):
        if globals.user_ready == False:
            globals.user_ready = True



    def closeEvent(self, event):
        # 拦截关闭事件，隐藏窗口而不是关闭
        self.hide()
        event.ignore()
        # super().closeEvent(event)

    
# 初始化串口
class SerialInit:
    def __init__(self):
        # self.ser = serial.Serial(config.SerialConfig.INDEX,115200,timeout = config.SerialConfig.TIMEOUT)
        pass

    def sendString(self, dst :int ,src :int,is_cheat=False):
        # 
        # # 发送字符串到串口
        # 
        # if self.ser.is_open:
        #     if not is_cheat:
        #         if globals.isrotate:
        #             format_string = f'deploy from {dst} to {src}\n'
        #             self.ser.write(format_string.encode())
        #         else:
        #             format_string = f'isrotate from {dst} to {src}\n'
        #             self.ser.write(format_string.encode())
        #     else:
        #         format_string = f'recover from {dst} to {src}\n'
        #         self.ser.write(format_string.encode())
        # else:
        #     print("Serial port is not open!")
        pass
        


def main():
    app = QApplication(sys.argv)

    # 创建并显示主窗口
    main_win = MainWindow()
    main_win.show()

    # 进入 Qt 事件循环
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()





