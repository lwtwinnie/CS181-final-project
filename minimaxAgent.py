import copy

class State:
    def __init__(self, current_snake, snakes, x1, x2, y1, y2, foodpos):
        self.current_snake = current_snake
        self.snakes = snakes
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.foodpos = foodpos
        self.moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    #求id蛇当前可以执行的移动
    def getLegalActions(self, id):
        found_snake = next((snake for snake in self.snakes if snake.id == id), None)
        #不能倒着走
        opposite_dir = (-found_snake.dir[0], -found_snake.dir[1])
        LegalActions = [action for action in self.moves if action != opposite_dir]
        return LegalActions
    
    def getNumAgents(self):
        return len(self.snakes)

    #模拟id蛇的移动move
    def getNextState(self, id, move):
        # Deep copy of the state is required
        new_state = copy.deepcopy(self)
        found_snake = next((snake for snake in new_state.snakes if snake.id == id), None)

        # Update the snake's position
        new_head = (found_snake.pos[0] + 10 * move[0], found_snake.pos[1] + 10 * move[1])
        found_snake.body.insert(0, new_head)
        found_snake.pos = new_head

        # Check for food consumption
        if new_head != new_state.foodpos:
            found_snake.body.pop()  # Remove tail segment if not eating food

        return new_state

    #给当前节点状态打分
    def evaluationFunction(self):
        scores = [0, 0, 0, 0]
        pos = self.current_snake.pos
        def calculate_manhattan_distance(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        # --------------cal scores-----------------
        # 特征1: 离食物的曼哈顿距离
        distance_to_food = calculate_manhattan_distance(pos, self.foodpos)
        scores[0] = 10000 / (distance_to_food + 0.1)
        # 特征2: 离最近的蛇头的曼哈顿距离
        # min_distance = 1e9
        # for other_snake in self.snakes:
        #     if other_snake.alive and other_snake.id != self.current_snake.id:
        #         distance = calculate_manhattan_distance(pos, other_snake.pos)
        #         min_distance = min(min_distance, distance)
        scores[1] = 0
        # 特征3: 距离边界的距离
        dx = min(abs(pos[0] - self.x1), abs(pos[0] - self.x2))
        dy = min(abs(pos[1] - self.y1), abs(pos[1] - self.y2))
        if min(dx, dy) <= 0:
            scores[2] = -1e9
        else:
            scores[2] = -10 / (min(dx, dy) - 0.1)
        # 特征4: 距离最近的蛇的身体的距离
        min_distance = 1e9
        for other_snake in self.snakes:
            if other_snake.alive and other_snake.id != self.current_snake.id:
                for body_segment in other_snake.body:
                    distance = calculate_manhattan_distance(pos, body_segment)
                    min_distance = min(min_distance, distance)
        if min_distance <= 0:
            scores[3] = -1e9
        else:
            scores[3] = -10 / (min_distance - 0.5)
        return sum(scores)

class MinimaxAgent:
    def __init__(self, current_snake, snakes, x1, x2, y1, y2, foodpos):
        #根据读入信息创建一个gamestate
        self.state = State(current_snake, snakes, x1, x2, y1, y2, foodpos)
        self.depth = 1  #最大递归深度
        self.current_snake=current_snake

    def getAction(self):
        #读入初始状态
        gameState = copy.deepcopy(self.state)
        def minimax(id, currentDepth, gameState,alpha,beta):
            # terminal check-->  如果撞墙/撞到其他蛇/吃到食物/达到最大递归深度，就认为是叶节点
            if gameState.evaluationFunction() >= 1e3 or gameState.evaluationFunction() <= -1e3:
                return gameState.evaluationFunction()
            if currentDepth == (self.depth):
                return gameState.evaluationFunction()

            # 如果不是叶节点，就要进一步递归
            if id == self.state.current_snake.id-1:
                nextOpponentIndex = id % gameState.getNumAgents()+2
            else:
                nextOpponentIndex = id % gameState.getNumAgents()+1
            #合法操作是，去除和当前方向相反的操作
            legalActions = gameState.getLegalActions(id)

            # -->MAX
            if id == self.state.current_snake.id:
                maxVal = float('-inf')
                for action in legalActions:
                    nextState = gameState.getNextState(id, action)
                    if self.state.current_snake.id==1:
                        evalScore = minimax(2, currentDepth + 1, nextState,alpha,beta)
                    else:
                        evalScore = minimax(1, currentDepth + 1, nextState,alpha,beta)
                    maxVal = max(maxVal, evalScore)
                    if maxVal>beta: return maxVal
                    alpha=max(maxVal,alpha)
                return maxVal

            # -->MIN
            else:
                minVal = float('inf')
                for action in legalActions:
                    nextState = gameState.getNextState(id, action)
                    if id == self.current_snake.id - 1:
                        evalScore = minimax(self.current_snake.id, currentDepth + 1, nextState,alpha,beta)#####计算自己的得分（max层）
                    else:
                        evalScore = minimax(nextOpponentIndex, currentDepth, nextState,alpha,beta)#####计算下一条对手蛇的得分（min层）
                    minVal = min(minVal, evalScore)
                    if minVal<alpha: return minVal
                    beta=min(minVal,beta)
                return minVal

        # Next legal action
        nextActions = gameState.getLegalActions(self.state.current_snake.id)  # ['West', 'Stop', 'East']
        # find the best one among next legal action
        bestValue = -1e9
        bestAction = (1, 0)
        alpha=float('-inf')
        beta=float('inf')
        # Correctly use the new getNextState
        for action in nextActions:
            nextState = gameState.getNextState(self.state.current_snake.id, action)
            value = minimax(self.state.current_snake.id, 0, nextState,alpha,beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha=max(alpha,value)

        return bestAction