import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



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

model_path = "./dqn_models/dqn_re_ep4000_0.0001.pth"
model = CustomNet()  # Initialize the model
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

def DQNAgent(current_snake, snakes, x1, x2, y1, y2, foodpos):
    # Convert the current state into the format expected by the model
    state = get_state((current_snake.pos[0],current_snake.pos[1]), foodpos, snakes, (x2-x1, y2-y1))
    state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        q_values = model(state_tensor)
        # print(q_values)
        action_index = torch.argmax(q_values, dim=1).item()

    actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    selected_action = actions[action_index]
    return selected_action