import sys

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QHBoxLayout,
    QStackedWidget, QLabel, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer,QRect
from PyQt5.QtGui import QImage, QPixmap,QPainter, QPen, QBrush

import numpy as np
import cv2
import sys

# 导入 config 模块
import config

# 开启多线程
# from concurrent.futures import ThreadPoolExecutor
import asyncio


from typing import List, Tuple


class CameraThread():

    def __init__(self):

         # 9个方格的中心点坐标     
        self.square_centers: List[Tuple[np.ndarray, int]] = []
        self.square_center :List[Tuple] = []
        # 该元组用于存储每个方格的状态，0，1，2分别为无，黑，白
        self.board_status : List[int] = []

        # 场外黑白棋的位置
        self.outside_black_chess_location : List[List[float]]= []
        self.outside_white_chess_location : List[List[float]]= []
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(config.CameraConfig.INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,config.CameraConfig.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,config.CameraConfig.HEIGHT)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G')) 
        if not self.cap.isOpened():
            print('Cannot open camera!')

    # 视频处理主函数
    # 主循环：不断读帧并处理
    def run(self):
        """
        将 detect_outside_chess 与 detect_board 并行运行，等 detect_board 返回后再做 check_board
        """
        # 创建一个 asyncio 事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while True:
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

                # 3. 并行启动两个协程：盘外棋子检测 & 棋盘检测
                #    注意：detect_board 需要原始的 frame 用于可视化
                tasks = [
                    self.detect_outside_chess(edges, left_edge, right_edge),
                    self.detect_board(edges, left_edge, right_edge, frame)
                ]
                done, pending = loop.run_until_complete(asyncio.wait(tasks))

                board_info = None
                for task in done:
                    res = task.result()
                    # detect_board 返回非 None 时就是它
                    if isinstance(res, dict):
                        board_info = res

                # 如果 detect_board 成功找到棋盘，调用 check_board
                if board_info is not None:
                    # 拿到灰度图，执行格子内棋子检测
                    self.check_board(gray,
                                     board_info['square_centers'],
                                     board_info['square_radius'])
                    # 把当前整张画面显示一下
                cv2.imshow("board", frame)

                # 按 'q' 键退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            loop.close()



    # 计算给定的2点之间的欧几里得距离
    def compute_distance(self,pt1,pt2):
        return ((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**0.5
    


    # 识别盘外棋子
    # 在图像左右侧检测，如果外界矩形近似于正方形且边长在一定范围内,就认为检测到圆形
    # 然后根据左黑右白，分别把圆的中心点加入到列表中
    async def detect_outside_chess(self,edges:np.ndarray, left_edge:int, right_edge:int) -> None:
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
        print(f'outside has {len(black_list)} black chess, {len(white_list)} white chess')



    # 异步协程：检测棋盘（中间区域），找到最大的四边形轮廓，计算出棋盘参数并返回
    # 返回值是字典类型，分别是小方格中心坐标，小方格半径，棋盘4个角点
    async def detect_board(self, edges: np.ndarray, left_edge: int, right_edge: int, frame: np.ndarray):
        mask = np.zeros_like(edges)
        # 中间区域全为255,左右边缘为0
        mask[: , left_edge:right_edge] = 255

        # 面积最大的就是棋盘
        biggest_area : int = 0
        biggest = None
        board_cons = cv2.findContours(cv2.bitwise_and(edges,edges,mask),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        if board_cons is not None:
            for con in board_cons:
                area = cv2.contourArea(con)
                if area>=config.MIN_AREA and area>biggest_area:
                    biggest_area = area
                    biggest = con
        
        if biggest is not None:
            perimeter = cv2.arcLength(biggest, True)
            board = cv2.approxPolyDP(biggest,config.EPSILON_RATIO*perimeter,True)
            if len(board) == 4:
                # 提取四个顶点，并根据左上/右上/右下/左下排序
                box = np.intp([pt[0] for pt in board])  # shape (4,2)
                # 根据 x+y 最小=> 左上，x+y 最大=> 右下，x-y 最大=> 右上，y-x 最大=> 左下
                top_left     = min(box,    key=lambda p: p[0] + p[1]) + np.array([left_edge, 0])
                bottom_right = max(box,    key=lambda p: p[0] + p[1]) + np.array([left_edge, 0])
                top_right    = min(box,    key=lambda p: p[0] - p[1]) + np.array([left_edge, 0])
                bottom_left  = max(box,    key=lambda p: p[1] - p[0]) + np.array([left_edge, 0])

                # 画出棋盘边框以调试
                cv2.rectangle(frame, tuple(top_left), tuple(bottom_right), (200,0,0), 1)

                 # 计算棋盘宽度和高度
                w1 = self.compute_distance(top_left, top_right)
                w2 = self.compute_distance(bottom_left, bottom_right)
                h1 = self.compute_distance(top_left, bottom_left)
                h2 = self.compute_distance(top_right, bottom_right)
                self.board_width = float((w1 + w2) / 2.0)
                self.board_height = float((h1 + h2) / 2.0)
                # 棋盘中心：取左上与右下的中点
                self.board_center = (top_left.astype(np.float32) + bottom_right.astype(np.float32)) / 2.0

                # 计算每个小格子内接圆半径
                self.square_radius = float(np.mean([self.board_width, self.board_height]) / 6.0)
                print(f'square_radius = {self.square_radius}')

                # 计算 9 个小格子的中心 (在棋盘坐标系中先算，再做旋转)
                # 先求棋盘在图像中的旋转角度
                delta = bottom_right.astype(np.float32) - top_right.astype(np.float32)
                rotate_angle = np.arctan2(delta[1], delta[0])  # 逆时针为正
                rotation_matrix = np.array([
                    [np.cos(rotate_angle), -np.sin(rotate_angle)],
                    [np.sin(rotate_angle),  np.cos(rotate_angle)]
                ], dtype=np.float32)

                # 棋盘中心拿去算完后作为旋转中心
                # 9 个小格子在“未旋转”状态下，相对于棋盘中心的偏移：
                square_order = [
                    (0, 0), (0, 1), (0, 2),
                    (1, 0), (1, 1), (1, 2),
                    (2, 0), (2, 1), (2, 2)
                ]
                centers = []
                for row, col in square_order:
                    # 未旋转时：每个方格中心相对于棋盘中心的偏移向量
                    rel_x = (2 * col - 2) * self.square_radius
                    rel_y = (2 * row - 2) * self.square_radius
                    rel_vec = np.array([rel_x, rel_y], dtype=np.float32)

                    # 旋转
                    rotated = rotation_matrix.dot(rel_vec.reshape(2,1)).reshape(2,)
                    abs_center = self.board_center + rotated
                    centers.append((abs_center, row * 3 + col + 1))

                    # 可视化：画出小圆
                    cv2.circle(frame,
                            (int(abs_center[0]), int(abs_center[1])),
                            int(self.square_radius),
                            (255, 0, 0), 2)

                # 将结果打包返回
                result = {
                    'square_centers': centers,
                    'square_radius': self.square_radius,
                    'board_corners': (top_left, top_right, bottom_right, bottom_left)
                }
                return result



    # 识别每个方格内是否有棋子，状态分别是0，1，2
    # 暂且先试试灰度图像的处理效果
    def check_board(self, gray: np.ndarray, square_centers: List[Tuple[np.ndarray, int]], square_radius: float):
        self.board_status = []
        for idx, (center, seq) in enumerate(square_centers):
            # 创建掩膜：一个和 gray 一样大小的全零矩阵
            h, w = gray.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask,
                       (int(center[0]), int(center[1])),
                       int(square_radius),  # 半径
                       1, -1)

            # 计算该圆内的平均灰度值
            mean_val = cv2.mean(gray, mask=mask)[0]

            if mean_val >= config.WHITE_CHESS_THRESH:
                self.board_status.append(2)
                print(f'格子 {seq} 是 白棋 (2)，平均灰度={mean_val:.2f}')
            elif mean_val <= config.BLACK_CHESS_THRESH:
                self.board_status.append(1)
                print(f'格子 {seq} 是 黑棋 (1)，平均灰度={mean_val:.2f}')
            else:
                self.board_status.append(0)
                print(f'格子 {seq} 是 无棋子 (0)，平均灰度={mean_val:.2f}')




       


'''
class Board(QWidget):
    def __init__(self):
        super().__init__()
        self.grid_size = 3
        self.cell_size = 100
        self.setFixedSize(self.grid_size * self.cell_size, self.grid_size * self.cell_size)
        self.board = [['' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.current_tool = None  # 'black' or 'white'

    def set_tool(self, tool):
        self.current_tool = tool

    def reset_board(self):
        self.board = [['' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.update()

    def mousePressEvent(self, event):
        if self.current_tool not in ('black', 'white'):
            return

        x = event.x() // self.cell_size
        y = event.y() // self.cell_size

        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.board[y][x] = self.current_tool
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        # Draw grid
        for i in range(1, self.grid_size):
            painter.drawLine(0, i * self.cell_size, self.width(), i * self.cell_size)
            painter.drawLine(i * self.cell_size, 0, i * self.cell_size, self.height())

        # Draw symbols
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                cell = self.board[y][x]
                center_x = x * self.cell_size + self.cell_size // 2
                center_y = y * self.cell_size + self.cell_size // 2
                half_size = self.cell_size // 3

                if cell == 'black':
                    painter.setBrush(QBrush(Qt.black))
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(center_x - half_size, center_y - half_size,
                                        2 * half_size, 2 * half_size)
                elif cell == 'white':
                    painter.setPen(QPen(Qt.black, 3))
                    painter.drawLine(center_x - half_size, center_y - half_size,
                                     center_x + half_size, center_y + half_size)
                    painter.drawLine(center_x + half_size, center_y - half_size,
                                     center_x - half_size, center_y + half_size)


class DeployPage(QWidget):
    def __init__(self, go_back_callback):
        super().__init__()
        self.current_mode = None  # 'black' or 'white'
        self.buttons = {}

        layout = QHBoxLayout()

        # 左边是棋盘
        self.board = Board()
        layout.addWidget(self.board)

        # 右边是按钮列
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)

        # Back 按钮在右上角
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(go_back_callback)
        right_layout.addWidget(back_btn, alignment=Qt.AlignRight)

        # 工具按钮（black, white, deploy, reset）
        for label in ['black', 'white', 'deploy', 'reset']:
            btn = QPushButton(label)
            btn.setFixedHeight(50)
            btn.clicked.connect(lambda _, l=label: self.handle_button(l))
            btn.setCheckable(True)
            self.buttons[label] = btn
            right_layout.addWidget(btn)

        layout.addLayout(right_layout)
        self.setLayout(layout)

    def handle_button(self, label):
        # 清除所有按钮的“选中”状态（视觉变暗）
        for btn in self.buttons.values():
            btn.setChecked(False)
            btn.setStyleSheet("")

        if label in ['black', 'white']:
            self.current_mode = label
            self.board.set_tool(label)
            self.buttons[label].setChecked(True)
            self.buttons[label].setStyleSheet("background-color: lightgray;")
        elif label in ['deploy', 'reset']:
            self.current_mode = None
            self.board.set_tool(None)
            self.board.reset_board()



class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # self.setWindowTitle("多页面 GUI 示例 - 扩展 Deploy")
        self.resize(600, 400)

        self.stack = QStackedWidget(self)
        self.main_page = self.create_main_page()
        self.deploy_page = DeployPage(go_back_callback=lambda: self.stack.setCurrentWidget(self.main_page))

        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.deploy_page)

        layout = QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)

    def create_main_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.addStretch()

        for label, callback in [
            ("deploy", lambda: self.stack.setCurrentWidget(self.deploy_page)),
            ("first_mover", lambda: print("TODO: 添加 first_mover 页面")),
            ("second_mover", lambda: print("TODO: 添加 second_mover 页面"))
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(80)
            btn.clicked.connect(callback)
            layout.addWidget(btn, alignment=Qt.AlignCenter)

        layout.addStretch()
        page.setLayout(layout)
        return page


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_camera()

    def init_ui(self):
        self.setWindowTitle("Camera + GUI 异步示例")
        self.resize(800, 600)

        # 页面堆栈
        self.stack = QStackedWidget()
        self.main_page = self.create_main_page()
        self.camera_page = self.create_camera_page()

        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.camera_page)

        layout = QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)

    def create_main_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        btn = QPushButton("打开摄像头页面", self)
        btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.camera_page))
        layout.addWidget(btn)
        page.setLayout(layout)
        return page

    def create_camera_page(self):
        page = QWidget()
        layout = QVBoxLayout()

        # 摄像头显示区域
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera_label)

        # 返回按钮
        back_btn = QPushButton("返回主页面")
        back_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.main_page))
        layout.addWidget(back_btn)

        page.setLayout(layout)
        return page

    def init_camera(self):
        # 创建摄像头线程
        self.camera_thread = CameraThread()
        # 连接信号到更新UI的槽函数
        self.camera_thread.frame_ready.connect(self.update_camera_view)

    def update_camera_view(self, image):
        # 在主线程更新UI
        pixmap = QPixmap.fromImage(image)
        self.camera_label.setPixmap(pixmap)

    def showEvent(self, event):
        # 页面显示时启动摄像头线程
        if not self.camera_thread.isRunning():
            self.camera_thread.start()

    def closeEvent(self, event):
        # 关闭窗口时停止摄像头线程
        self.camera_thread.stop()
        event.accept()



def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
'''

if __name__ == "__main__":
    camThread = CameraThread()
    camThread.run()
