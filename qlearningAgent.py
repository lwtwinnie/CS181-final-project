import numpy as np
import random
import pickle
import os
import copy

class FeatureBasedQLearningAgent:
    def __init__(self, current_snake, snakes, x1, x2, y1, y2, foodpos):
        self.moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.epsilon = 0.1  # 探索率
        self.alpha = 0.01  # 学习率
        self.alpha2 = 0.003  # 学习率
        self.gamma = 1  # 折扣因子
        self.graders = 4  #子评分器个数
        self.weights = np.ones(4)  #各评分器权重
        self.iteration = 0
        self.current_snake = current_snake   #当前的主角蛇
        self.snakes = snakes
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.foodpos = foodpos

        '''读取训练数据'''
        self.weightFile = 'weights.pkl'  #权重数据
        self.iterationFile = 'iterations.txt'   #迭代轮数
        self.historyFile = 'history.txt'  #全部历史数据
        self.load_weights()
    def load_weights(self):
        if self.weightFile and os.path.exists(self.weightFile):
            with open(self.weightFile, 'rb') as file:
                self.weights = pickle.load(file)
        if self.iterationFile and os.path.exists(self.iterationFile):
            with open(self.iterationFile, 'r') as file:
                self.iteration = int(file.read())
            self.iteration += 1
    def save_weights(self):
        with open(self.weightFile, 'wb') as file:
            pickle.dump(self.weights, file)
        with open(self.iterationFile, 'w') as file:
            file.write(str(self.iteration))
        with open(self.historyFile, 'a') as file:
            file.write(f'iterations: {self.iteration}, weights: {self.weights}\n')
    
    def get_features(self, move):
        features =  np.zeros(self.graders)
        #-------------helper function---------------
        def calculate_manhattan_distance(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        # -------------one more step---------------
        pos = self.current_snake.pos
        pos = (pos[0] + 10 * move[0], pos[1] + 10 * move[1])
        #--------------cal features-----------------
        # 特征1: 离食物的曼哈顿距离
        distance_to_food = calculate_manhattan_distance(pos, self.foodpos)
        features[0] = 1/(distance_to_food+ 1)
        # 特征2: 离最近的蛇头的曼哈顿距离，效果不好，删掉了
        features[1] =0
        # 特征3: 距离边界的距离
        dx = min(abs(pos[0]-self.x1), abs(pos[0]-self.x2))
        dy = min(abs(pos[1]-self.y1), abs(pos[1]-self.y2))
        if min(dx, dy) <= 0:
            features[2] = -5
        else:
            features[2] = -1 / (min(dx, dy))

        # 特征4: 距离最近的蛇的身体的距离
        min_distance = (self.x2-self.x1)+(self.y2-self.y1)
        for other_snake in self.snakes:
            if other_snake.alive and other_snake.id != self.current_snake.id:
                for body_segment in other_snake.body:
                    distance = calculate_manhattan_distance(pos, body_segment)
                    min_distance = min(min_distance, distance)
        if min_distance <= 0:
            features[3] = -5
        else:
            features[3] = -1 / (min_distance+1)


        return np.array(features)

    # 给(s,a)打分
    def grade_sa(self, move):
        features = self.get_features(move)
        grades = np.dot(self.weights, features)
        return grades

    def choose_action(self):
        scores = []
        for move in self.moves:
            if move[0] == -self.current_snake.dir[0] and move[1] == -self.current_snake.dir[1]:
                scores.append(-1e5)
            else:
                scores.append(self.grade_sa(move))
        #print(scores)
        choose_move = self.moves[scores.index(max(scores))]
        estimateQ = max(scores)
        estimateF = self.get_features(choose_move)
        self.update_weights(choose_move, estimateQ, estimateF)
        return choose_move


        # # epsilon-greedy策略选择动作
        # if random.random() < self.epsilon:
        #     return random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        # else:
        #     pass

    def update_weights(self, choosemove, estimateQ, estimateF):
        # -------------helper function---------------
        def calculate_manhattan_distance(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        # 假设执行了该动作，到下一个格子
        choosemove = list(choosemove)
        choosemove = [x * 10 for x in choosemove]
        copy_snake = copy.deepcopy(self.current_snake)
        copy_snake.pos = [pos + move for pos, move in zip(self.current_snake.pos, choosemove)]
        new_agent = FeatureBasedQLearningAgent(copy_snake, self.snakes, self.x1, self.x2, self.y1, self.y2, self.foodpos)
        scores = []


        # 已经死亡
        if (self.x1 == copy_snake.pos[0] or self.x2 == copy_snake.pos[0] or self.y1 == copy_snake.pos[1] or self.y2 == copy_snake.pos[1]):
            real_Q = -1
        else:
            for move in self.moves:
                    scores.append(new_agent.grade_sa(move))
            new_Q = max(scores)
            reward = 0
            #碰到了食物
            if (self.foodpos == copy_snake.pos):
                reward = 1
            real_Q = reward + new_Q * self.gamma
        for other_snake in self.snakes:
            if other_snake.alive and other_snake.id != self.current_snake.id:
                if copy_snake.pos in other_snake.body:
                    real_Q = -1
        # -----------------------------------------
        #自动保存权重到文件里
        # print('------------------')
        # print(real_Q)
        # print(estimateQ)

        difference = real_Q - estimateQ
        # with open('difference_values.txt', 'a') as file:
        #         file.write(f'{difference}\n')  # Write only the 'difference' value
        if self.iteration >= 5000:
            self.weights += self.alpha * difference * estimateF
        else:
            self.weights += self.alpha2 * difference * estimateF
        with open('weights_history.txt', 'a') as file:
            file.write(','.join(map(str, self.weights)) + '\n')
        self.save_weights()


