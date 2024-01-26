import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from snakeClass import Snake
from snakeAgent import Snake_num, Agent, DQNAgent

#------获取特征-------
#目标：将特征[己方蛇头所在坐标、对手蛇头所在坐标、对手身体所在坐标、食物所在坐标]输入3层的神经网络卷积，输出相应的动作，以执行该动作的得分作为reward。
#获取特征实际方法1；蛇头和食物的相对x坐标和y坐标，蛇头上、下、左、右是否有自身身体或者游戏边界作为state，效果很好，训练后AI超过普通玩家水平
#state=[xfood−xhead,yfood−yhead,k1,k2,k3,k4]
def get_state(snake_head, food_position, snake_body, grid_size):
    # 计算相对坐标差值
    relative_x = food_position[0] - snake_head[0]
    relative_y = food_position[1] - snake_head[1]

    # 初始化方向上是否有障碍的标志
    obstacles = [0, 0, 0, 0]  # 上、下、左、右

    # 检查上方是否有障碍
    if snake_head[1] == 0 or (snake_head[0], snake_head[1] - 1) in snake_body:
        obstacles[0] = 1

    # 检查下方是否有障碍
    if snake_head[1] == grid_size[1] - 1 or (snake_head[0], snake_head[1] + 1) in snake_body:
        obstacles[1] = 1

    # 检查左方是否有障碍
    if snake_head[0] == 0 or (snake_head[0] - 1, snake_head[1]) in snake_body:
        obstacles[2] = 1

    # 检查右方是否有障碍
    if snake_head[0] == grid_size[0] - 1 or (snake_head[0] + 1, snake_head[1]) in snake_body:
        obstacles[3] = 1

    # 归一化相对坐标差值
    normalized_relative_x = relative_x / grid_size[0]
    normalized_relative_y = relative_y / grid_size[1]

    # 构建状态向量
    state = [normalized_relative_x, normalized_relative_y] + obstacles

    return np.array(state)

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # 输入层，6个特征
        self.fc1 = nn.Linear(6, 64)
        # 隐藏层
        self.fc2 = nn.Linear(64, 32)
        # 输出层，4个动作
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        # 输入层
        x = F.relu(self.fc1(x))
        # 隐藏层
        x = F.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    # (s, a, r, n_s)= sample

# 生成食物
def generate_food(dis_width, dis_height, border_size, snakes):
    food_pos = [random.randrange(border_size, (dis_width - border_size) // 10) * 10,
                random.randrange(border_size, (dis_height - border_size) // 10) * 10]

    # 确保食物的位置不与玩家蛇和其他蛇的位置重合
    while any(food_pos in snake.body and snake.alive for snake in snakes):
        food_pos = [random.randrange(border_size, (dis_width - border_size) // 10) * 10,
                    random.randrange(border_size, (dis_height - border_size) // 10) * 10]

    return food_pos


def generate_snakes(dis_width, dis_height, border_size, snake_count):
    snakes = []

    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 128, 0),
              (128, 0, 255), (255, 0, 255), (128, 128, 0), (0, 128, 128),
              (128, 128, 128), (0, 0, 128), (128, 0, 0), (0, 128, 0)]  # 更多颜色

    for snakeID in range(1, snake_count + 1):
        snake_color = random.choice(colors)
        colors.remove(snake_color)  # 避免重复使用颜色

        snake_pos = [random.randrange(border_size, ((dis_width - border_size) // 10)) * 10 - 20,
                     random.randrange(border_size, ((dis_height - border_size) // 10)) * 10 - 20]

        # 确保新蛇的位置不和其他蛇的初始位置重叠
        while abs(snake_pos[0] - dis_width // 2) < 50 and abs(snake_pos[1] - dis_height // 2) < 50 and (any(abs(snake_pos[0] - other_snake.pos[0]) < 30 and abs(snake_pos[1] - other_snake.pos[1]) < 30 for other_snake in snakes)):
            snake_pos = [random.randrange(border_size, ((dis_width - border_size) // 10)) * 10,
                         random.randrange(border_size, ((dis_height - border_size) // 10)) * 10]

        snake_body = [snake_pos.copy(), [snake_pos[0] - 10, snake_pos[1]],
                      [snake_pos[0] - 20, snake_pos[1]]]

        snake = Snake(color=snake_color, id=snakeID, pos=snake_pos, dir=(1,0), 
                      body=snake_body, score=0, alive=True, agent=Agent(snakeID))

        snakes.append(snake)

    return snakes

# 检测蛇是否吃到食物
def check_food(snakes, food_pos):
    isFoodEat = False
    for snake in snakes:
        if not snake.alive:
            continue        
        if snake.pos[0] == food_pos[0] and snake.pos[1] == food_pos[1]:
            # 吃到食物，增加积分
            snake.score += 1
            isFoodEat = True
        else:
            snake.body.pop()

    if isFoodEat:
        food_pos = generate_food(
            dis_width, dis_height, border_size, snakes)
    return (snakes,food_pos)

# 检测蛇是否碰到其他蛇
def check_collision(snakes):
    score = 0
    die_list = []

    # 撞到其他蛇
    for i in range(len(snakes)):
        if not snakes[i].alive:
            continue
        for j in range(i + 1, len(snakes)):
            if not snakes[j].alive:
                continue
            # 蛇i碰到了蛇j，蛇i死亡
            for body in snakes[i].body:
                if body in snakes[j].body:
                    snakes[j].score += snakes[i].score
                    die_list.append(snakes[i])
                    break
            #if snakes[i].pos == snakes[j].pos or snakes[i].pos in snakes[j].body:

    for snake in die_list:
        snake.alive = False
        snake.score = 0

    return (snakes, score)

# 检测蛇是否碰到边界
def check_boundary_collision(snakes):
    for snake in snakes:
        if not snake.alive:
            continue
        if snake.pos[0] <= border_size or snake.pos[0] >= dis_width - border_size or \
           snake.pos[1] <= border_size or snake.pos[1] >= dis_height - border_size:
            # 蛇碰到边界，蛇死亡
            snake.alive = False
            snake.score = 0
    return snakes

def get_ori_reward(snakes, food_pos,  dis_width, dis_height, border_size):
    def calculate_reward(snakes, food_pos, dis_width, dis_height, border_size):
        reward = 0
        for snake in snakes:
            if snake.agent == DQNAgent:
                player = snake
        player_head = player.pos

        # Reward for being closer to food (using reciprocal of distance)
        food_distance = manhattan_distance(player_head, food_pos)
        if food_distance != 0:
            reward += 2 / food_distance  

        # Penalty for being too close to enemies
        for snake in snakes:
            if snake != player:
                enemy_distance = min(manhattan_distance(player_head, snake.pos),
                                     min(manhattan_distance(player_head, segment) for segment in snake.body))
                if enemy_distance != 0:
                    reward -= 0.5 / enemy_distance 

        # Small penalty for being close to boundaries
        boundary_distance = min(player_head[0] - border_size, dis_width - border_size - player_head[0],
                                player_head[1] - border_size, dis_height - border_size - player_head[1])
        if boundary_distance != 0:
            reward -= 4 / (1 * boundary_distance)

        return reward

    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    ans_reward = calculate_reward(snakes, food_pos, dis_width, dis_height, border_size)
    return torch.tensor(ans_reward).float()


# 模拟采取select_action之后的reward
def get_reward(choose_move, snakes, foodpos, x1, x2, y1, y2):
    features = np.zeros(5)

    # -------------helper function---------------
    def calculate_manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    for snake in snakes:
        if snake.agent == DQNAgent:
            current_snake = snake
    # -------------模拟向前移动一步---------------
    pos = current_snake.pos
    next_pos = (pos[0] + 10 * choose_move[0], pos[1] + 10 * choose_move[1])

    # --------------cal features-----------------
    # 特征1: 离食物的曼哈顿距离
    distance_to_food = calculate_manhattan_distance(next_pos, foodpos)
    if distance_to_food == 0:
        features[0] = 10 / (distance_to_food + 1.1)
    else:
        features[0] = 10 / (distance_to_food + 0.1)
    # 特征2: 离最近的蛇头的曼哈顿距离
    features[1] = 0
    # 特征3: 距离边界的距离
    dx = min(abs(pos[0] - x1), abs(pos[0] - x2))
    dy = min(abs(pos[1] - y1), abs(pos[1] - y2))
    if min(dx, dy) <= 0:
        features[2] = -100
    else:
        features[2] = -10 / (min(dx, dy) - 0.5)
    # 特征4: 距离最近的蛇的身体的距离
    min_distance = 1000 # 1e9
    for other_snake in snakes:
        if other_snake.alive and other_snake.id != current_snake.id:
            for body_segment in other_snake.body:
                distance = calculate_manhattan_distance(next_pos, body_segment)
                min_distance = min(min_distance, distance)
    if min_distance == 0:
        features[3] = -100
    else:
        features[3] = -10 / (min_distance - 0.5)
    # 特征5：与当前方向不相同
    if choose_move[0] == -current_snake.dir[0] and choose_move[1] == -current_snake.dir[1]:
        features[4] = -100  # 与当前方向相反，分数为负无穷
    reward = sum(features)
    return reward
    
def move_snakes(snakes, x1, x2, y1, y2, foodpos, epsilon):
    all_snake = snakes.copy()

    for snake in snakes:
        if snake.agent == DQNAgent:
            state = get_state(snake.pos, foodpos, snake.body, (dis_width, dis_height))
            state_tensor = torch.from_numpy(state)
            input_data = torch.unsqueeze(state_tensor, 0).float()
            
            if random.random() > epsilon:
                output = model(input_data)
                sorted_indices = torch.argsort(output, dim=1, descending=True)[0]
                action_index = sorted_indices[0].item()
                action = directions[action_index]
                if not is_valid_action(snake.dir, action):
                    action_index = sorted_indices[1].item()
                    action = directions[action_index]
            else:
                # Choose a random action from the start, ensuring it's valid
                valid_directions = [d for d in directions if is_valid_action(snake.dir, d)]
                action = random.choice(valid_directions)
                action_index = directions.index(action)  
            snake_pos = snake.pos
            movepos = action
        else:
            movepos = snake.agent(snake, all_snake, x1, x2, y1, y2, foodpos)
        if movepos[0] + snake.dir[0] == 0 and movepos[1] + snake.dir[1] == 0:
            movepos = snake.dir
        else:
            snake.dir = movepos
        snake.pos[0] += movepos[0] * 10
        snake.pos[1] += movepos[1] * 10
        snake.body.insert(0, list(snake.pos))
    
    return state, snakes, action, action_index, snake_pos, snake.body

def is_valid_action(snake_dir, action):
    # Check if the action is not a 180-degree turn
    return not (action[0] + snake_dir[0] == 0 and action[1] + snake_dir[1] == 0)

directions = [(0, 1), (0, -1), (-1, 0), (1, 0)] # up, down, left, right
max_steps = 100
direction_to_index = {direction: idx for idx, direction in enumerate(directions)}
def train(epochs, model, target_model, optimizer, criterion, replay_buffer, dis_width, dis_height, border_size, batch_size=32, gamma=0.99, target_update_frequency=25, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.994, lr=0.0005):
    losses, eps = [], []
    x1, x2, y1, y2 = 20, 400, 20, 400
    for epoch in range(epochs):
        total_loss = 0
        # reset game
        score, steps = 0, 0
        game_over = False

        snakes = generate_snakes(dis_width, dis_height, border_size, Snake_num())

        food_pos = generate_food(dis_width, dis_height, border_size, snakes)
        
        while steps < max_steps:
            for snake in snakes:
                if snake.agent == DQNAgent:
                    if not snake.alive:
                        break
            alive_snake=0
            for snake in snakes:
                if snake.alive:
                    alive_snake+=1
            if(alive_snake==1):
                game_over = True
                break

            state, snakes, action, action_index, snake_pos, snake_body = move_snakes(snakes, border_size, dis_width -
                        border_size, border_size, dis_height - border_size, food_pos, epsilon)
            # 检测食物
            snakes, food_pos = check_food(snakes, food_pos)

            # 检测碰撞
            snakes, addscore = check_collision(snakes)
            snakes = check_boundary_collision(snakes)

            score += addscore
            
            next_state = get_state(snake_pos, food_pos, snake_body, (dis_width, dis_height))
            reward = get_ori_reward(snakes, food_pos,  dis_width, dis_height, border_size)
            # reward = get_reward(action, snakes, food_pos, x1, x2, y1, y2)

            # Store experience in replay buffer
            # print(reward)
            replay_buffer.push(state, action_index, reward, next_state, float(0.0))
            

            # Check if replay buffer is large enough for sampling
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                for state, action_index, reward, next_state, done in batch:
                    state = torch.FloatTensor(state).unsqueeze(0) 
                    next_state = torch.FloatTensor(next_state).unsqueeze(0)
                    action = torch.LongTensor([action_index])  
                    reward = torch.FloatTensor([reward])
                    done = torch.FloatTensor([done])

                    current_q_values = model(state)
                    current_q = current_q_values[0, action] # .unsqueeze(0) 

                    next_state_q_values = target_model(next_state)
                    max_next_state_q_values = torch.max(next_state_q_values)  
                    target_q = reward + gamma * max_next_state_q_values * (1 - done)
                    # print(current_q, target_q)
                    
                    # Compute loss
                    loss = criterion(current_q, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    
                    if total_loss > 1e6:
                        break
            if total_loss > 1e6:
                break
            steps += 1
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        losses.append(total_loss)
        eps.append(epsilon)
        # Update target network
        if epoch % target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())
            
        if epoch % 100 == 0:
            torch.save(model.state_dict(), f'dqn_re_ep{epoch+3000}_{lr}.pth')
            
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}, Epsilon: {epsilon}")

    return model, losses, eps

dis_width = 420
dis_height = 420
border_size = 20

# Initialize models, optimizer, loss function, and replay buffer
learning_rate = 0.0001 # 0.0005
model = CustomNet()
target_model = CustomNet()
saved_model_path = "./dqn_re_ep3000_0.0001.pth"  # Update path if different
model.load_state_dict(torch.load(saved_model_path))
target_model.load_state_dict(model.state_dict())  # Ensure target model is updated
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
replay_buffer = ReplayBuffer(capacity=10000)

# Call the train function with necessary parameters
trained_model, losses, eps = train(epochs=1001, model=model, target_model=target_model, optimizer=optimizer, 
                      criterion=criterion, replay_buffer=replay_buffer, dis_width=500, dis_height=500, 
                      border_size=10, lr=learning_rate)

# Writing losses to a file
with open('./loss0125.txt', 'a') as file:
    for i in range(len(losses)):
        file.write(f"{losses[i], eps[i]}\n")
    file.write('\n\n')


