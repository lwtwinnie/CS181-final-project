import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import os
save_dir = 'saved_model'
os.makedirs(save_dir, exist_ok=True)
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(10, 64, bias=True)
        self.fc2 = nn.Linear(64, 32, bias=True)
        self.fc3 = nn.Linear(32, 1, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class MLPagent:
    def __init__(self, current_snake, snakes, x1, x2, y1, y2, foodpos):
        #----------------------游戏参数-----------------------
        self.moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]#右左上下
        self.current_snake = current_snake
        self.snakes = snakes
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.foodpos = foodpos
        self.iteration = 0
        self.load_history()
        self.save_dir = 'saved_model'
        self.iterationFile = 'MLPiteration.txt'
        self.gamma = 0.8
        #-------------------预测网络-----------------------------
        #如果有，就读入历史数据
        self.net=CustomNet()
        if os.path.exists(os.path.join(save_dir, 'snake_model.pth')):
            self.net.load_state_dict(torch.load(os.path.join(save_dir, 'snake_model.pth')))
        else:
            self.net = CustomNet()
        self.learning_rate = 0.0001 * self.gamma ** (self.iteration % 10)
        if self.iteration <= 4000:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    def load_history(self):
        with open('MLPiteration.txt', 'r') as file:
            self.iteration = int(file.read())
        self.iteration += 1

    def stateof(self,move):
        # 计算相对坐标差值
        pos = self.current_snake.pos
        pos = (pos[0] + 10 * move[0], pos[1] + 10 * move[1])
        relative_x = self.foodpos[0] - pos[0]
        relative_y = self.foodpos[1] - pos[1]
        # 各方向距离边界的距离
        obstacles = [0, 0, 0, 0]  # 上、下、左、右
        obstacles[0] = self.y2 - pos[1]
        obstacles[1] = pos[1]-self.y1
        obstacles[2] = pos[0]-self.x1
        obstacles[3] = self.x2 - pos[0]
        normalized_obstacles = [obstacle / (self.x2 - self.x1) for obstacle in obstacles]

        concatenated_body = []
        # 遍历每个snake对象，将其body拼接到concatenated_body中
        for snake in self.snakes:
            if snake.id != self.current_snake.id and snake.alive:
                concatenated_body.extend(snake.body)
        min_distance_up = 1600
        min_distance_down = 1600
        min_distance_left = 1600
        min_distance_right = 1600
        # 遍历concatenated_body中的每个坐标
        for coord in concatenated_body:
            # 上方向
            if pos[1] < coord[1]:  # 如果pos在coord的上方
                min_distance_up = min(min_distance_up, abs(pos[0] - coord[0]) + abs(pos[1] - coord[1]))
            # 下方向
            if pos[1] > coord[1]:  # 如果pos在coord的下方
                min_distance_down = min(min_distance_down, abs(pos[0] - coord[0]) + abs(pos[1] - coord[1]))
            # 左方向
            if pos[0] < coord[0]:  # 如果pos在coord的左方
                min_distance_left = min(min_distance_left, abs(pos[0] - coord[0]) + abs(pos[1] - coord[1]))
            # 右方向
            if pos[0] > coord[0]:  # 如果pos在coord的右方
                min_distance_right = min(min_distance_right, abs(pos[0] - coord[0]) + abs(pos[1] - coord[1]))
        min_body = [min_distance_up, min_distance_left, min_distance_right, min_distance_down]
        min_body = [d / (self.x2 - self.x1) for d in min_body]
        # 归一化相对坐标差值
        normalized_relative_x = relative_x / (self.x2 - self.x1)
        normalized_relative_y = relative_y / (self.y2 - self.y1)

        # 构建状态向量
        state = [normalized_relative_x, normalized_relative_y] + normalized_obstacles + min_body
        # state = [relative_x, relative_y] + normalized_obstacles + min_body

        # return np.array(state)
        # print(state)
        return state
    def get_score(self, move):
        # 1 找到state
        state = self.stateof(move)
        # 2 对state进行打分,使用神经网络，输出分数
        #YOUR CODE HERE
        state_tensor = torch.tensor(state)
        state_tensor = torch.unsqueeze(state_tensor, 0).float()
        output = self.net(state_tensor)
        # print("output:",output)
        # normalized_output = (output - torch.min(output)) / (torch.max(output) - torch.min(output))
        return output.item()


    #利用get_score选择4个move中最好的一个(用神经网络打分，选择一个得分最高的)
    def select_action(self):
        scores = []
        for move in self.moves:
            if move[0] == -self.current_snake.dir[0] and move[1] == -self.current_snake.dir[1]:
                scores.append(-1e5)
            else:
                score = self.get_score(move)
                scores.append(score)
        scores_list = scores
        #print("scores_list:",scores_list)
        # 选择评分最高的行为
        chosen_index = scores_list.index(max(scores_list))
        chosen_score = scores_list[chosen_index]
        choose_move = self.moves[chosen_index]

        with open(self.iterationFile, 'w') as file:
            file.write(str(self.iteration))

        self.backward_propagation(chosen_score, choose_move)
        return choose_move


    # 模拟采取select_action之后的reward，算真实reward
    def get_reward(self, move):#choose_move
        features = np.zeros(4)
        # -------------helper function---------------
        def calculate_manhattan_distance(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        # -------------one more step---------------
        pos = self.current_snake.pos
        pos = (pos[0] + 10 * move[0], pos[1] + 10 * move[1])
        # --------------cal features-----------------
        # 特征1: 离食物的曼哈顿距离
        distance_to_food = calculate_manhattan_distance(pos, self.foodpos)
        features[0] = 1 / (distance_to_food + 1)
        # 特征2: 离最近的蛇头的曼哈顿距离
        if len(self.current_snake.body) >= 3:
            body = copy.copy(self.current_snake.body)
            body.insert(0, pos)
            body.pop()
            tail = body[3:]
            min_kill = 1600
            for other_snake in self.snakes:
                if other_snake.alive and other_snake.id != self.current_snake.id:
                    for tail_segment in tail:
                        d = calculate_manhattan_distance(other_snake.pos, tail_segment)
                        min_kill = min(min_kill, d)
            features[1] = 5 / (distance_to_food + 1)
        else:
            features[1] = 0

        # 特征3: 距离边界的距离
        dx = min(abs(pos[0] - self.x1), abs(pos[0] - self.x2))
        dy = min(abs(pos[1] - self.y1), abs(pos[1] - self.y2))
        if min(dx, dy) <= 0:
            features[2] = -5
        else:
            features[2] = -1 / (min(dx, dy))

        # 特征4: 距离最近的蛇的身体的距离
        min_distance = (self.x2 - self.x1) + (self.y2 - self.y1)
        for other_snake in self.snakes:
            if other_snake.alive and other_snake.id != self.current_snake.id:
                for body_segment in other_snake.body:
                    distance = calculate_manhattan_distance(pos, body_segment)
                    min_distance = min(min_distance, distance)
        if min_distance <= 0:
            features[3] = -5
        else:
            features[3] = -1 / (min_distance + 1)
        # 点乘权重
        weights = np.array([0.8196472306066146, 0.09823615303306298, 0.9974289027902798, 0.9974541434333334])
        reward = features * weights*0.1
        return reward



    def backward_propagation(self, max_score, choose_move):

        def append_loss_to_file_reward_output_loss(file_path, loss1, loss2, loss3):
            with open(file_path, 'a') as file:
                file.write(f'{loss1} {loss2} {loss3}\n')

        #1 计算损失,利用reward更新网络,乘以学习率self.learning_rate

        max_score_tensor = torch.tensor(max_score, dtype=torch.float32, requires_grad=True)
        reward_value = self.get_reward(choose_move)
        reward_tensor = torch.tensor(reward_value, dtype=torch.float32, requires_grad=True)
        #print(max_score_tensor)
        loss = self.criterion(max_score_tensor, reward_tensor)

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # # 2 保存模型
        # torch.save(self.net.state_dict(), os.path.join(save_dir, 'snake_model.pth'))
        # print(f'Model parameters saved to {os.path.join(save_dir, "snake_model.pth")}')

