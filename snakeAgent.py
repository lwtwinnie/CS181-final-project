from snakeClass import Snake
import random
from qlearningAgent import FeatureBasedQLearningAgent
from minimaxAgent import MinimaxAgent
from dqnAgent import DQNAgent
from MLPagent import MLPagent

def Agent(snakeID):
    if  snakeID == 1:
        return randomAgent
    elif snakeID == 2:
        return minimaxAgent
    elif snakeID == 3:
        return QlearningAgent
    else:
        return MLPAgent
    #为不同编号的蛇分配不同的agent


def Snake_num():
    return 3 #代表蛇的数量，可修改 最大13

def set_width():
    return 420 #设置地图宽度 必须是10的倍数（一步10格）

def set_height():
    return 420 #设置地图高度
def set_length():
    return 6

def set_speed():
    return 80 #调整游戏速度，如果训练的话可以把这个值调的很大
'''

参数解释
    cur 为当前需要移动的蛇 Snake类
    snakes 为所有蛇的列表(包括玩家) Snake类的列表
    cur, snakes均为Snake类
    以下是Snake的定义
class Snake:
    def __init__(self, color, pos, body, score, alive):
        self.color = color 颜色，不会用到
        self.pos = pos 蛇头部的位置,横坐标pos[0],纵坐标pos[1],向右就是pos[0]+=1,向下就是pos[0]+=1
        self.dir = dir 蛇前进的方向 (±1,0)/(0,±1) ,注意蛇不能反向,所以实际上蛇每步只有三个方向可以选择。
                如果Agent返回了错误的方向,则不会执行该方向
                dir不需要修改,main.py会根据return的方向坐标修改dir
        self.body = body 蛇所有身体的坐标,例如可以用some_pos in some_snake.body来判断某个位置是否会撞到别的蛇
        self.score = score 蛇的得分
        self.alive = alive 蛇是否还活着 True/False 所以访问蛇数组时要先判断是否alive

        self.id = id 蛇的编号, 玩家的编号默认为0
        self.agent = agent 蛇的agent, 可以达到不同的蛇有不同的agent的效果
        

    x1 x2 y1 y2为地图边界 横坐标[x1,x2] 纵坐标[y1,y2]

    foodpos食物坐标

    当写好Agent后,修改Agent()函数（在最上面！）的返回值为其他函数名以测试

'''
def keepRightAgent(cur, snakes: Snake, x1, x2, y1, y2, foodpos):

    return (1, 0)  # 向右走

def randomAgent(cur, snakes: Snake, x1, x2, y1, y2, foodpos):
    
    return random.choice([(1, 0),(-1,0),(0,1),(0,-1)])

# input:当前移动的蛇，玩家蛇，边界，食物位置，全部蛇的数组
def directionalAgent(current_snake, snakes, x1, x2, y1, y2, foodpos):
    moves = [(10, 0), (-10, 0), (0, 10), (0, -10)]  # 可能的移动方向
    actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    scores = []

    def score_move(snake, x1, x2, y1, y2, foodpos, snakes, move):
        def calculate_manhattan_distance(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        score = 0
        next_pos = (snake.pos[0] + move[0], snake.pos[1] + move[1])
        # 1. 检查是否撞墙
        if next_pos[0] <= x1 or next_pos[0] >= x2 or next_pos[1] <= y1 or next_pos[1] >= y2:
            score += -1e5

        # 2. 计算离食物曼哈顿距离的得分
        distance_to_food = calculate_manhattan_distance(next_pos, foodpos)
        score += -distance_to_food + 1000

        # 3. 检查是否撞上其他蛇的身体
        for other_snake in snakes:
            if other_snake.alive and other_snake != snake:
                if other_snake != snake and next_pos in other_snake.body:
                    score += -1e5  # 撞上其他蛇的身体

        # 4. 排除与当前方向相反的移动方向
        if move[0] == -10 * snake.dir[0] and move[1] == -10 * snake.dir[1]:
            score += -1e5  # 与当前方向相反，分数为负无穷
        return score
    for move in moves:
        # 对每个可能的移动方向计算得分
        score = score_move(current_snake, x1, x2, y1, y2, foodpos, snakes, move)
        scores.append(score)

    # 选择分数最高的移动方向
    best_move = actions[scores.index(max(scores))]
    return best_move


def directional_random_Agent(current_snake, snakes, x1, x2, y1, y2, foodpos):
    direction = directionalAgent(current_snake, snakes, x1, x2, y1, y2, foodpos)
    epsilon = 0.1
    actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # 随机选择动作的概率
    if random.random() < epsilon:
        non_opposite_directions = [action for action in actions if action != (-current_snake.dir[0], -current_snake.dir[1])]
        direction = random.choice(non_opposite_directions)

    return direction

def QlearningAgent(current_snake, snakes, x1, x2, y1, y2, foodpos):
    snakeAgent = FeatureBasedQLearningAgent(current_snake, snakes, x1, x2, y1, y2, foodpos)
    direction = snakeAgent.choose_action()
    return direction

def minimaxAgent(current_snake, snakes, x1, x2, y1, y2, foodpos):
    snakeAgent = MinimaxAgent(current_snake, snakes, x1, x2, y1, y2, foodpos)
    direction = snakeAgent.getAction()
    return direction

def MLPAgent(current_snake, snakes, x1, x2, y1, y2, foodpos):
    snakeAgent = MLPagent(current_snake, snakes, x1, x2, y1, y2, foodpos)
    direction = snakeAgent.select_action()
    return direction

def DeepQNAgent(current_snake, snakes, x1, x2, y1, y2, foodpos):
    action = DQNAgent(current_snake, snakes, x1, x2, y1, y2, foodpos)
    return action
    
    