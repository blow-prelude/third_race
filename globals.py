# 先手执黑棋
# 先手的标志位，1表示用户先手first_gamer ,反之2表示系统先手
# 在实际使用中，first_turn 表示用户，而 3- first_turn 表示自己
first_turn = 0

# 判断人机对弈有无结束
game_finished = False

# 判断用户是否下完
user_ready :bool = False


# 判断棋盘是否旋转
isrotate = False