import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,QWidget, QLabel,QTextEdit,QSizePolicy
from PyQt5.QtCore import Qt,QThread, pyqtSignal, pyqtSlot, QMutex
from PyQt5.QtGui import QImage, QPixmap

import numpy as np
import cv2
import sys

# 导入 config 模块
import config

# 开启多线程

# from concurrent.futures import ThreadPoolExecutor
import asyncio


from typing import List, Tuple


class CameraThread(QThread):

    # 定义一个信号，用来把处理完的 numpy.ndarray 帧发给主线程
    frame_ready = pyqtSignal(QImage)

    def __init__(self,parent=None):
        super().__init__(parent)

        # 用于控制线程循环
        self._running = True
        
        self._lock = QMutex()

         # 9个方格的中心点坐标     
        self.square_centers: List[Tuple[np.ndarray, int]] = []
        self.square_center :List[Tuple] = []
        # 该元组用于存储每个方格的状态，0，1，2分别为无，黑，白
        self.board_status : List[int] = []

        # 场外黑白棋的位置
        self.outside_black_chess_location : List[List[float]]= []
        self.outside_white_chess_location : List[List[float]]= []


        # 初始化串口
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(config.CameraConfig.INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,config.CameraConfig.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,config.CameraConfig.HEIGHT)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G')) 
        if not self.cap.isOpened():
            print('Cannot open camera!')
        


    def stop(self):
        """外部调用，用来停止循环"""
        self._lock.lock()
        self._running = False
        self._lock.unlock()


    '''
    # 视频处理主函数
    # 主循环：不断读帧并处理
    def run(self):
        """
        将 detect_outside_chess 与 detect_board 并行运行，等 detect_board 返回后再做 check_board
        """
        # 创建一个 asyncio 事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

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

            # 3. 并行启动两个协程：盘外棋子检测 & 棋盘检测
            #    注意：detect_board 需要原始的 frame 用于可视化
            coroutines = [
                self.detect_outside_chess(...),
                self.detect_board(...)
            ]
            tasks = [loop.create_task(coro) for coro in coroutines]
            done = loop.run_until_complete(asyncio.wait(tasks))[0]

            board_info = None
            for task in done:
                res = task.result()

                if isinstance(res, dict):
                    board_info = res

            # 如果 detect_board 成功找到棋盘，调用 check_board
            if board_info is not None:
                # 拿到灰度图，执行格子内棋子检测
                self.check_board(gray,
                                 board_info['square_centers'],
                                 board_info['square_radius'])

            # 处理结束后，不再用 cv2.imshow，而是通过信号把 frame 发给主线程
            self.frame_ready.emit(frame.copy())

            # 这个 sleep 可以稍微降低 CPU 占用（可选）
            self.msleep(10)

        # 循环结束后释放资源
        self.cap.release()
        loop.close()

    '''
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
            self.detect_outside_chess(edges, left_edge, right_edge)
            board_info = self.detect_board(edges, left_edge, right_edge, frame)

            # 4. 如果成功检测到棋盘，则检测棋盘上的棋子
            if board_info is not None:
                self.check_board(
                    gray,
                    board_info['square_centers'],
                    board_info['square_radius']
                )

            # 转为 QImage 并发射
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        print(f'outside has {len(black_list)} black chess, {len(white_list)} white chess')



    # 异步协程：检测棋盘（中间区域），找到最大的四边形轮廓，计算出棋盘参数并返回
    # 返回值是字典类型，分别是小方格中心坐标，小方格半径，棋盘4个角点
    def detect_board(self, edges: np.ndarray, left_edge: int, right_edge: int, frame: np.ndarray):
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




       



# 3个子窗口的共性功能
class BaseFunctionWindow(QMainWindow):
    def __init__(self, main_window, title):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle(title)
        self.resize(600, 400)

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

        # 在主界面竖直居中排列3个botton

        # 分别创建3个按钮
        self.button1 = QPushButton("deploy", self)
        self.button1.clicked.connect(lambda: self.switch_to_window(self.deploy_window))
        self.button1.setFixedSize(100, 50)
        # self.button1.setStyleSheet("background-color: white; color: black; font-size: 20px;")

        self.button2 = QPushButton("first_gamer", self)
        self.button2.clicked.connect(lambda: self.switch_to_window(self.first_gamer_window))
        self.button2.setFixedSize(100, 50)

        self.button3 = QPushButton("secong_gamer", self)
        self.button3.clicked.connect(lambda: self.switch_to_window(self.second_gamer_window))
        self.button3.setFixedSize(100, 50)

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
        self.camera_thread.start()



    def switch_to_window(self, target_window):
        """切换到指定窗口"""
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

    def on_deploy_clicked(self):
        """
        Deploy：目前仅清空所有格子上的标记（后续可加入串口发送逻辑）
        """
        # TODO: 串口发送逻辑放在这里
        pass

    # BaseFunctionWindow 已经把 return_to_main_window() 连接到 self.back_button
    # 这里不需要 override closeEvent，除非你在 DeployPage 里有其他额外资源需要释放。




# 人机对弈界面
class Gamer(BaseFunctionWindow):
    def __init__(self, main_window, lecture_title):
        super().__init__(main_window, lecture_title)

        # 界面左侧放一个 QLabel，用来显示视频帧 
        self.video_label = QLabel()
        # 给一个初始大小，后面可以根据需要调整
        self.video_label.setFixedSize(240, 180)
        self.video_label.setStyleSheet("background-color: black;")

         # ———————— 2. 右侧：信息区（状态 + 作弊 + Save + Back） ———————— #

        # 2.1 状态区：用一个只读的 QTextEdit 或 QLabel 也可以
        self.status_area = QTextEdit()
        self.status_area.setReadOnly(True)
        self.status_area.setFixedHeight(80)
        self.status_area.setPlaceholderText("Status: your turn / my turn / you win / you lose")
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

        # 2.4 Back 按钮（继承自 BaseFunctionWindow，已经在 __init__ 里创建好）
        # 这里我们设置它的大小，并且放到最底部
        self.back_button.setFixedSize(100, 40)

        # 2.5 把 2.1、2.2、2.3、2.4 按纵向顺序放到一个 QVBoxLayout 里
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.status_area)
        right_layout.addWidget(self.cheat_area)
        right_layout.addStretch(1)               # 在按钮和状态区之间占个“弹性”空白，让按钮靠底下
        right_layout.addWidget(self.save_button, alignment=Qt.AlignRight)
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
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def save_board(self):
        """
        Save 按钮的槽函数，此处先留空，后续再实现保存棋盘状态的逻辑。
        """
        # TODO: 在这里把当前棋盘状态保存到文件或数据库
        pass



    def closeEvent(self, event):
    # 拦截关闭事件，隐藏窗口而不是关闭
    # self.hide()
    # event.ignore()
        super().closeEvent(event)

    
    


def main():
    app = QApplication(sys.argv)

    # 创建并显示主窗口
    main_win = MainWindow()
    main_win.show()

    # 进入 Qt 事件循环
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()




