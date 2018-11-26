import numpy as np
import pyglet
import random
import math
import collections

# write a txt file
file = open('WithTL.txt','w')

# window size
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# grid size （i.e 50 * 50）
HORIZONTAL_GRID_NUM = int(WINDOW_WIDTH/50)
VERTICAL_GRID_NUM = int(WINDOW_HEIGHT/50)
GRID_WIDTH = WINDOW_WIDTH / HORIZONTAL_GRID_NUM
GRID_HEIGHT = WINDOW_HEIGHT / VERTICAL_GRID_NUM

# block color & amount & positions
BLOCK_COLOR = (0, 0, 0)
BLOCK_NUM = 10
BLOCK_POSITION = []
for i in range(BLOCK_NUM):
    LEFT_BOT_X = random.randint(0, HORIZONTAL_GRID_NUM-1) * 50
    LEFT_BOT_Y = random.randint(0, VERTICAL_GRID_NUM-1) * 50
    # avoid duplication
    if (LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1) in BLOCK_POSITION:
        while ((LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1) in BLOCK_POSITION):
            LEFT_BOT_X = random.randint(0, HORIZONTAL_GRID_NUM-1) * 50
            LEFT_BOT_Y = random.randint(0, VERTICAL_GRID_NUM-1) * 50
    else:
        LEFT_BOT_X = LEFT_BOT_X
        LEFT_BOT_Y = LEFT_BOT_Y
    BLOCK_POSITION.append((LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1))

# rubbish color & amount & dynamic positions
RUBBISH_COLOR = (222, 227, 255)
RUBBISH_NUM = 50
RUBBISH_POSITION = []
CLEAN_COLOR = (255,255,255)
CLEAN_POSITION = []
for i in range(RUBBISH_NUM):
    LEFT_BOT_X = random.randint(0, HORIZONTAL_GRID_NUM-1) * 50
    LEFT_BOT_Y = random.randint(0, VERTICAL_GRID_NUM-1) * 50
    # rubbish has a unique position and cannot be duplicated with the block
    if ((LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1) in BLOCK_POSITION) or ((LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1) in RUBBISH_POSITION):
        while (((LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1) in BLOCK_POSITION) or ((LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1) in RUBBISH_POSITION)):
            LEFT_BOT_X = random.randint(0, HORIZONTAL_GRID_NUM-1) * 50
            LEFT_BOT_Y = random.randint(0, VERTICAL_GRID_NUM-1) * 50
    else:
        LEFT_BOT_X = LEFT_BOT_X
        LEFT_BOT_Y = LEFT_BOT_Y
    RUBBISH_POSITION.append((LEFT_BOT_X/50+1,LEFT_BOT_Y/50+1))

# bot number & color & initial positions
BOT_NUM = 3
BOT_COLOR = (255, 0, 0)
BOT_POSITION = [None] * BOT_NUM
TMP_BOT_POSITION = [None] * BOT_NUM
BOT1_LEFT_BOT_X = random.randint(0, HORIZONTAL_GRID_NUM-1) * 50
BOT1_LEFT_BOT_Y = random.randint(0, VERTICAL_GRID_NUM-1) * 50
# bot position cannot be duplicated with other bots and blocks
if (BOT1_LEFT_BOT_X/50+1,BOT1_LEFT_BOT_Y/50+1) in BLOCK_POSITION:
    while ((BOT1_LEFT_BOT_X/50+1,BOT1_LEFT_BOT_Y/50+1) in BLOCK_POSITION):
        BOT1_LEFT_BOT_X = random.randint(0, HORIZONTAL_GRID_NUM-1) * 50
        BOT1_LEFT_BOT_Y = random.randint(0, VERTICAL_GRID_NUM-1) * 50
else:
    BOT1_LEFT_BOT_X = BOT1_LEFT_BOT_X
    BOT1_LEFT_BOT_Y = BOT1_LEFT_BOT_Y
BOT_POSITION[0] = (BOT1_LEFT_BOT_X/50+1,BOT1_LEFT_BOT_Y/50+1)

BOT2_LEFT_BOT_X = random.randint(0, HORIZONTAL_GRID_NUM/2-1) * 50
BOT2_LEFT_BOT_Y = random.randint(0, VERTICAL_GRID_NUM/2-1) * 50
# bot position cannot be duplicated with other bots and blocks
if ((BOT2_LEFT_BOT_X/50+1,BOT2_LEFT_BOT_Y/50+1) in BLOCK_POSITION) or ((BOT2_LEFT_BOT_X/50+1,BOT2_LEFT_BOT_Y/50+1) in BOT_POSITION):
    while ((BOT2_LEFT_BOT_X/50+1,BOT2_LEFT_BOT_Y/50+1) in BLOCK_POSITION) or ((BOT2_LEFT_BOT_X/50+1,BOT2_LEFT_BOT_Y/50+1) in BOT_POSITION):
        BOT2_LEFT_BOT_X = random.randint(0, HORIZONTAL_GRID_NUM-1) * 50
        BOT2_LEFT_BOT_Y = random.randint(0, VERTICAL_GRID_NUM-1) * 50
else:
    BOT2_LEFT_BOT_X = BOT2_LEFT_BOT_X
    BOT2_LEFT_BOT_Y = BOT2_LEFT_BOT_Y
BOT_POSITION[1] = (BOT2_LEFT_BOT_X/50+1,BOT2_LEFT_BOT_Y/50+1)

BOT3_LEFT_BOT_X = random.randint(0, HORIZONTAL_GRID_NUM-1) * 50
BOT3_LEFT_BOT_Y = random.randint(0, VERTICAL_GRID_NUM-1) * 50
# bot position cannot be duplicated with other bots and blocks
if ((BOT3_LEFT_BOT_X/50+1,BOT3_LEFT_BOT_Y/50+1) in BLOCK_POSITION) or ((BOT3_LEFT_BOT_X/50+1,BOT3_LEFT_BOT_Y/50+1) in BOT_POSITION):
    while ((BOT3_LEFT_BOT_X/50+1,BOT3_LEFT_BOT_Y/50+1) in BLOCK_POSITION) or ((BOT3_LEFT_BOT_X/50+1,BOT3_LEFT_BOT_Y/50+1) in BOT_POSITION):
        BOT3_LEFT_BOT_X = random.randint(0, HORIZONTAL_GRID_NUM-1) * 50
        BOT3_LEFT_BOT_Y = random.randint(0, VERTICAL_GRID_NUM-1) * 50
else:
    BOT3_LEFT_BOT_X = BOT3_LEFT_BOT_X
    BOT3_LEFT_BOT_Y = BOT3_LEFT_BOT_Y
BOT_POSITION[2] = (BOT3_LEFT_BOT_X/50+1,BOT3_LEFT_BOT_Y/50+1)

# initialise bots' observation，distribution of actions and utility
observation = []    # [{key: observation (bot:-1,block:-1,boundary:-1,vacant:0,rubbish:1), value: happened frequency}]
distribution = []   # [{key: observation, value: {key: action, value: probability}}]
utility = []        # [{key: observation, value: {key: action, value: utility}}]
tmp_observation = [None] * BOT_NUM
reward_matrix = [0] * BOT_NUM

# parameters
alpha = 0.1
gamma = 0.9
zeta = 0.1
epsilon = 1
sensitivity = 3
ln_t = 1 # ln_t = 1 to 10

# lines color and start position
LINE_COLOR = (120, 120, 120, 120)
START_X = 0
START_Y = 0

hit_num = 0

class BotEnv(object):
    viewer = None
    actions = ['up', 'down', 'left', 'right']
    goal = [] # goals' positions (i.e. rubbish position)
    for j in range(RUBBISH_NUM):
        goal.append({'x': RUBBISH_POSITION[j][0], 'y':RUBBISH_POSITION[j][1]})

    def __init__(self):
        self.bot_info = np.zeros(3, dtype=[('x', np.float32), ('y', np.float32)])
        for i in range(BOT_NUM):
            self.bot_info[i]['x'] = BOT_POSITION[i][0]
            self.bot_info[i]['y'] = BOT_POSITION[i][1]

    # take an action from bot
    def step(self, action):
        done = False
        reward = 0
        global BOT_POSITION
        global RUBBISH_POSITION
        global CLEAN_POSITION
        global distribution
        global utility

        for i in range(BOT_NUM):
            self.bot_info[i]['x'] = BOT_POSITION[i][0]
            self.bot_info[i]['y'] = BOT_POSITION[i][1]
            if action[i] == 'up':
                self.bot_info[i]['x'] += 0
                self.bot_info[i]['y'] += 1
            elif action[i] == 'down':
                self.bot_info[i]['x'] += 0
                self.bot_info[i]['y'] += -1
            elif action[i] == 'left':
                self.bot_info[i]['x'] += -1
                self.bot_info[i]['y'] += 0
            elif action[i] == 'right':
                self.bot_info[i]['x'] += 1
                self.bot_info[i]['y'] += 0

        (BOT1_X, BOT2_X, BOT3_X) = self.bot_info['x'] # bots current x coordination
        (BOT1_Y, BOT2_Y, BOT3_Y) = self.bot_info['y'] # bots current y coordination
        TMP_BOT_POSITION[0] = (BOT1_X, BOT1_Y)
        TMP_BOT_POSITION[1] = (BOT2_X, BOT2_Y)
        TMP_BOT_POSITION[2] = (BOT3_X, BOT3_Y)
        
        # done and reward
        for i in range(BOT_NUM):
            global hit_num
            utility.append([]) 
            REST_BOT_POSITION = []
            tmp_list = TMP_BOT_POSITION[:]
            tmp_list.pop(i)
            REST_BOT_POSITION.append(tmp_list)
            # reward if move to a rubbish position and move
            if (self.bot_info[i]['x'], self.bot_info[i]['y']) in RUBBISH_POSITION:
                print('1')
                reward = 10
                BOT_POSITION[i] = TMP_BOT_POSITION[i]
                RUBBISH_POSITION.remove(TMP_BOT_POSITION[i])
                CLEAN_POSITION.append(TMP_BOT_POSITION[i])
                done = True
            # punish if hit a block and do not move
            elif (self.bot_info[i]['x'], self.bot_info[i]['y']) in BLOCK_POSITION:
                reward = -5
                hit_num += 1
                done = True
                print('2')
            # punish if hit the boundary and do not move
            elif self.bot_info[i]['x'] < 1:
                reward = -5
                hit_num += 1
                done = True
                print('3')
            elif self.bot_info[i]['x'] > HORIZONTAL_GRID_NUM :
                reward = -5
                hit_num += 1
                done = True
                print('4')
            elif self.bot_info[i]['y'] < 1:
                reward = -5
                hit_num += 1
                done = True
                print('5')
            elif self.bot_info[i]['y'] > VERTICAL_GRID_NUM:
                reward = -5
                hit_num += 1
                done = True
                print('6')
            # punish if hit other bots and do not move
            elif (self.bot_info[i]['x'], self.bot_info[i]['y']) in REST_BOT_POSITION:
                reward = -10
                hit_num += 1
                done = True
                print('7')
            # neither reward or punish if move to a vacant grid
            else:
                print('8')
                reward = 0
                BOT_POSITION[i] = TMP_BOT_POSITION[i]
                done = True
            # calculate reward matrix
            reward_matrix[i] += reward
            # utility of t+1
            new_observation = BotEnv().get_observation(i)
            if new_observation in observation:
                max_utility = max(utility[i][new_observation].values())
            else:
                max_utility = 1
            utility[i][tmp_observation[i]][action[i]] = (1 - alpha) * utility[i][tmp_observation[i]][action[i]] + alpha * (reward + gamma * max_utility)
            # total reward
            total_reward = 0
            for j in range(len(self.actions)):
                total_reward += distribution[i][tmp_observation[i]][self.actions[j]] * utility[i][tmp_observation[i]][self.actions[j]]
            for j in range(len(self.actions)):
                distribution[i][tmp_observation[i]][self.actions[j]] = distribution[i][tmp_observation[i]][self.actions[j]] + zeta * (utility[i][tmp_observation[i]][self.actions[j]] - total_reward)
            BotEnv().normalise(distribution[i][tmp_observation[i]])
        print(BOT_POSITION)
        return reward, done

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.bot_info, self.goal)
        self.viewer.render()

    # random move
    def sample_action(self):
        action = []
        for i in range(BOT_NUM):
            action.append(random.sample(self.actions, 1)[0])    # generate an action for each bot
        return action

    # get the observation of a bot
    def get_observation(self, i):
        x = BOT_POSITION[i][0]
        y = BOT_POSITION[i][1]
        around = [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1), (x, y + 1), (x + 1, y + 1), (x + 1, y), (x + 1, y - 1), (x, y - 1)]
        observation_ = ""
        for j in range(len(around)):
            if around[j] in RUBBISH_POSITION:
                observation_ += "1,"
            elif around[j] in BLOCK_POSITION:
                observation_ += "-1,"
            elif around[j] in BOT_POSITION:
                observation_ += "-2,"
            elif around[j][0] < 1:
                observation_ += "-1,"
            elif around[j][0] > HORIZONTAL_GRID_NUM:
                observation_ += "-1,"
            elif around[j][1] < 1:
                observation_ += "-1,"
            elif around[j][0] > VERTICAL_GRID_NUM:
                observation_ += "-1,"
            else:
                observation_ += "0,"
        observation_ = observation_[:-1]
        return observation_

    # normalise the distribution of observations
    def normalise(self, distribution_):
        c0 = 0.5
        delta = 0.001
        d = min(distribution_.values())
        if d < delta:
            rho = (c0 - delta)/(c0-d)
            for k in range(len(self.actions)):
                distribution_[self.actions[k]] = c0 - rho * (c0 - distribution_[self.actions[k]])
        pi = sum(distribution_.values())
        for k in range(len(self.actions)):
            distribution_[self.actions[k]] = distribution_[self.actions[k]]/pi
        return distribution_

    # find the most similar observation
    def similar_observation(self, key_, dict1):
        keys = list(dict1.keys())
        key_ = key_.split(',')
        similar_observation_ = {}
        for i in range(len(keys)):
            keys[i] = keys[i].split(',')
            # make a comparison for each surroundings
            difference = 0
            similar = True
            for j in range(len(keys[i])):
                if keys[i][j] != key_[j]:
                    difference += 1
                # set the tolerated diffenence
                if difference > 1:
                    similar = False
                    break
            if similar:
                similar_observation_[list(dict1.keys())[i]] = list(dict1.values())[i]
        # return the observation with highest happened frequency
        if similar_observation_ != {}:
            return max(similar_observation_, key = similar_observation_.get)
        else:
            return {}

    # add laplace noise
    def laplace(self, m, n, mu, b):
        u = random.uniform(0, 1)
        u = u - 0.5
        sigma = 1
        b = sigma/math.sqrt(2)
        if u >= 0:
            y = mu - b * u * math.log(1 - 2 * abs(u))
        else:
            y = mu + b * u * math.log(1 - 2 * abs(u))
        return y

    # algorithm 1: Adviser player selection
    def algorithm_one(self, NEIGHBOR_NUM):
        b = sensitivity * ln_t/epsilon
        for i in range(len(reward_matrix)):
            reward_matrix[i] += BotEnv().laplace(1, 1, 0, b)
        p = []
        e = 0.1 #0.1,0.2...
        selected_bot = 0
        for j in range(len(NEIGHBOR_NUM)):
            if NEIGHBOR_NUM[j] == reward_matrix.index(max(reward_matrix)):
                p.append((1 - e) + (e/(BOT_NUM - 1)))
            else:
                p.append(e/(BOT_NUM - 1))
        # select a player
        random_ = np.random.rand()
        for m in range(len(NEIGHBOR_NUM)):
            if random_ <= sum(p[:(m + 1)]):
                selected_bot = m
                break
        return selected_bot

    # algorithm 2: knowledge transfer
    def algorithm_two(self, ob, i):
        distribution_ = {}
        for j in range(len(self.actions)):
            distribution_[self.actions[j]] =  math.exp((epsilon * utility[i][ob][self.actions[j]]) / ((2 * sensitivity * ln_t)))
        distribution_ = BotEnv().normalise(distribution_)
        return distribution_

    # algorithm 3: the transfer learning algorithm
    def algorithm_three(self):
        action = []
        global BOT_POSITION
        global observation 
        global distribution
        global utility
        global tmp_observation

        for i in range(BOT_NUM):
            observation.append({})
            utility.append({})
            distribution.append({})
            key = BotEnv().get_observation(i)
            tmp_observation[i] = key
            # decide whether to request knowledge
            if '-2' in key:
                x = BOT_POSITION[i][0]
                y = BOT_POSITION[i][1]
                # find neighbor bots
                NEIGHBOR_POS = 0
                NEIGHBOR_NUM = []
                for j in range(len(key.split(','))):
                    if key.split(',')[j] == '-2':
                        if j == 0:
                            NEIGHBOR_POS = (x - 1, y - 1)
                        elif j == 1:
                            NEIGHBOR_POS = (x - 1, y)
                        elif j == 2:
                            NEIGHBOR_POS = (x - 1, y + 1)
                        elif j == 3:
                            NEIGHBOR_POS = (x, y + 1)
                        elif j == 4:
                            NEIGHBOR_POS = (x + 1, y + 1)
                        elif j == 5:
                            NEIGHBOR_POS = (x + 1, y)
                        elif j == 6:
                            NEIGHBOR_POS = (x + 1, y - 1)
                        elif j == 7:
                            NEIGHBOR_POS = (x, y - 1)
                        NEIGHBOR_NUM.append(BOT_POSITION.index(NEIGHBOR_POS))

                if key in observation[i]:
                    confidence = 1/observation[i][key]
                    rand = np.random.rand()
                    observation[i][key] += 1
                    if rand < confidence:
                        tmp_dict = observation[i].copy()
                        tmp_dict.pop(key)
                        ob3 = BotEnv().similar_observation(key, tmp_dict)
                        if ob3 == {}:
                            # generate an action for each bot according to the distribution
                            print("distribution = ",distribution[i][key])
                            random_ = np.random.rand()
                            for m in range(len(self.actions)):
                                if random_ <= sum(list(distribution[i][key].values())[:(m + 1)]):
                                    action.append(self.actions[m])
                                    break
                        else:
                            distribution[i][key] = BotEnv().algorithm_two(ob3, i)
                            # generate an action for each bot according to the distribution
                            print("distribution = ",distribution[i][key])
                            random_ = np.random.rand()
                            for m in range(len(self.actions)):
                                if random_ <= sum(list(distribution[i][key].values())[:(m + 1)]):
                                    action.append(self.actions[m])
                                    break
                    else:
                        # generate an action for each bot according to the distribution
                        print("distribution = ",distribution[i][key])
                        random_ = np.random.rand()
                        for m in range(len(self.actions)):
                            if random_ <= sum(list(distribution[i][key].values())[:(m + 1)]):
                                action.append(self.actions[m])
                                break
                else:
                    print("observation[i]:",observation[i])
                    ob3 = BotEnv().similar_observation(key, observation[i])
                    observation[i][key] = 1
                    utility[i][key] = {}
                    utility[i][key]['up'] = 1
                    utility[i][key]['down'] = 1
                    utility[i][key]['left'] = 1
                    utility[i][key]['right'] = 1
                    distribution[i][key] = {}
                    distribution[i][key]['up'] = 0.25
                    distribution[i][key]['down'] = 0.25
                    distribution[i][key]['left'] = 0.25
                    distribution[i][key]['right'] = 0.25
                    if ob3 == {}:
                        selected_bot = BotEnv().algorithm_one(NEIGHBOR_NUM)
                        ob2 = BotEnv().similar_observation(key, observation[selected_bot])
                        if ob2 != {}:
                            distribution[i][key] = BotEnv().algorithm_two(ob2, selected_bot)
                            # generate an action for each bot according to the distribution
                            print("distribution = ",distribution[i][key])
                            random_ = np.random.rand()
                            for m in range(len(self.actions)):
                                if random_ <= sum(list(distribution[i][key].values())[:(m + 1)]):
                                    action.append(self.actions[m])
                                    break
                        else:
                            # generate an action for each bot according to the distribution
                            print("distribution = ",distribution[i][key])
                            random_ = np.random.rand()
                            for m in range(len(self.actions)):
                                if random_ <= sum(list(distribution[i][key].values())[:(m + 1)]):
                                    action.append(self.actions[m])
                                    break
                    else:
                        distribution[i][key] = BotEnv().algorithm_two(ob3, i)
                        # generate an action for each bot according to the distribution
                        print("distribution = ",distribution[i][key])
                        random_ = np.random.rand()
                        for m in range(len(self.actions)):
                            if random_ <= sum(list(distribution[i][key].values())[:(m + 1)]):
                                action.append(self.actions[m])
                                break
            else:
                # if the observation exists in the knowledge, happened time accumulates
                if key in observation[i].keys():
                    observation[i][key] += 1
                # if the observation is new, a new key is created and the probability of each action is indifferent
                else:
                    observation[i][key] = 1
                    utility[i][key] = {}
                    utility[i][key]['up'] = 1
                    utility[i][key]['down'] = 1
                    utility[i][key]['left'] = 1
                    utility[i][key]['right'] = 1
                    distribution[i][key] = {}
                    distribution[i][key]['up'] = 0.25
                    distribution[i][key]['down'] = 0.25
                    distribution[i][key]['left'] = 0.25
                    distribution[i][key]['right'] = 0.25
                # generate an action for each bot according to the distribution
                print("distribution = ", distribution[i][key])
                random_ = np.random.rand()
                for m in range(len(self.actions)):
                    if random_ <= sum(list(distribution[i][key].values())[:(m + 1)]):
                        action.append(self.actions[m])
                        break
        print('action: ', action)
        return action

class Viewer(pyglet.window.Window):
    def __init__(self, bot_info, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(WINDOW_WIDTH, WINDOW_HEIGHT, resizable=False, caption='Room', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.bot_info = bot_info

        self.batch = pyglet.graphics.Batch()                       # display whole batch at once
        # draw blocks
        for i in range(BLOCK_NUM):
            BLOCK_LEFT_BOT_X = BLOCK_POSITION[i][0] * 50 - 50
            BLOCK_LEFT_BOT_Y = BLOCK_POSITION[i][1] * 50 - 50
            self.point = self.batch.add(
                4, pyglet.gl.GL_QUADS, None,                       # 4 corners
                ('v2f', [BLOCK_LEFT_BOT_X, BLOCK_LEFT_BOT_Y,       # location
                         BLOCK_LEFT_BOT_X, BLOCK_LEFT_BOT_Y + 50,
                         BLOCK_LEFT_BOT_X + 50, BLOCK_LEFT_BOT_Y + 50,
                         BLOCK_LEFT_BOT_X + 50, BLOCK_LEFT_BOT_Y]),
                ('c3B', (BLOCK_COLOR) * 4))                        # color
        print('Block Position: ', BLOCK_POSITION)

        # draw rubbish
        for i in range(RUBBISH_NUM):
            RUBBISH_LEFT_BOT_X = goal[i]['x'] * 50 - 50
            RUBBISH_LEFT_BOT_Y = goal[i]['y'] * 50 - 50
            self.goal = self.batch.add(
                4, pyglet.gl.GL_QUADS, None,                       # 4 corners
                ('v2f', [RUBBISH_LEFT_BOT_X, RUBBISH_LEFT_BOT_Y,   # location
                         RUBBISH_LEFT_BOT_X, RUBBISH_LEFT_BOT_Y + 50,
                         RUBBISH_LEFT_BOT_X + 50, RUBBISH_LEFT_BOT_Y + 50,
                         RUBBISH_LEFT_BOT_X + 50, RUBBISH_LEFT_BOT_Y]),
                ('c3B', (RUBBISH_COLOR) * 4))                      # color
        print('Rubbshi Position: ', RUBBISH_POSITION)

        # draw bots at the initial positions 
        self.bot1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [BOT1_LEFT_BOT_X, BOT1_LEFT_BOT_Y,             # location
                     BOT1_LEFT_BOT_X, BOT1_LEFT_BOT_Y + 50,
                     BOT1_LEFT_BOT_X + 50, BOT1_LEFT_BOT_Y + 50,
                     BOT1_LEFT_BOT_X + 50, BOT1_LEFT_BOT_Y]),
            ('c3B', (BOT_COLOR) * 4))                              # color

        self.bot2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [BOT2_LEFT_BOT_X, BOT2_LEFT_BOT_Y,             # location
                     BOT2_LEFT_BOT_X, BOT2_LEFT_BOT_Y + 50,
                     BOT2_LEFT_BOT_X + 50, BOT2_LEFT_BOT_Y + 50,
                     BOT2_LEFT_BOT_X + 50, BOT2_LEFT_BOT_Y]), 
            ('c3B', (BOT_COLOR) * 4))                              # color

        self.bot3 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [BOT3_LEFT_BOT_X, BOT3_LEFT_BOT_Y,             # location
                     BOT3_LEFT_BOT_X, BOT3_LEFT_BOT_Y + 50,
                     BOT3_LEFT_BOT_X + 50, BOT3_LEFT_BOT_Y + 50,
                     BOT3_LEFT_BOT_X + 50, BOT3_LEFT_BOT_Y]), 
            ('c3B', (BOT_COLOR) * 4))                              # color
        print('Bot Position: ', BOT_POSITION)

    # draw grid
    def draw_grid(self,start_x,start_y):
        rows = VERTICAL_GRID_NUM+ 1
        columns = HORIZONTAL_GRID_NUM + 1
        # draw rows
        for row in range(rows):
            pyglet.graphics.draw(
                2, pyglet.gl.GL_LINES,
                ('v2f',
                    (
                        start_x, row * GRID_HEIGHT + start_y,
                        GRID_HEIGHT * HORIZONTAL_GRID_NUM + start_x, row * GRID_HEIGHT + start_y
                    )
                ),
                ('c4B', LINE_COLOR * 2)
            )

        # draw columns
        for column in range(columns):
            pyglet.graphics.draw(
                2, pyglet.gl.GL_LINES,
                ('v2f',
                    (
                        column * GRID_WIDTH + start_x, start_y,
                        column * GRID_WIDTH + start_x, GRID_WIDTH * VERTICAL_GRID_NUM + start_y
                    )
                ),
                ('c4B', LINE_COLOR * 2)
            )

    def render(self):
        self._update()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        self.draw_grid(START_X,START_Y)

    def _update(self):
        (BOT1_X, BOT2_X, BOT3_X) = self.bot_info['x'] # bots x current coordination
        (BOT1_Y, BOT2_Y, BOT3_Y) = self.bot_info['y'] # bots y current coordination

        # re-draw the bots
        self.bot1.vertices = np.concatenate(([BOT1_X * 50 -50, BOT1_Y * 50 - 50], [BOT1_X * 50, BOT1_Y * 50 - 50], [BOT1_X * 50, BOT1_Y * 50], [BOT1_X * 50 -50, BOT1_Y * 50]))
        self.bot2.vertices = np.concatenate(([BOT2_X * 50 -50, BOT2_Y * 50 - 50], [BOT2_X * 50, BOT2_Y * 50 - 50], [BOT2_X * 50, BOT2_Y * 50], [BOT2_X * 50 -50, BOT2_Y * 50]))
        self.bot3.vertices = np.concatenate(([BOT3_X * 50 -50, BOT3_Y * 50 - 50], [BOT3_X * 50, BOT3_Y * 50 - 50], [BOT3_X * 50, BOT3_Y * 50], [BOT3_X * 50 -50, BOT3_Y * 50]))

        # re-draw the rubbish
        for i in range(len(CLEAN_POSITION)):
            RUBBISH_LEFT_BOT_X = CLEAN_POSITION[i][0] * 50 - 50
            RUBBISH_LEFT_BOT_Y = CLEAN_POSITION[i][1] * 50 - 50
            self.goal = self.batch.add(
                4, pyglet.gl.GL_QUADS, None,                       # 4 corners
                ('v2f', [RUBBISH_LEFT_BOT_X, RUBBISH_LEFT_BOT_Y,   # location
                         RUBBISH_LEFT_BOT_X, RUBBISH_LEFT_BOT_Y + 50,
                         RUBBISH_LEFT_BOT_X + 50, RUBBISH_LEFT_BOT_Y + 50,
                         RUBBISH_LEFT_BOT_X + 50, RUBBISH_LEFT_BOT_Y]),
                ('c3B', (CLEAN_COLOR) * 4))                      # color
        print('Clean Position: ', CLEAN_POSITION)

        for i in range(len(RUBBISH_POSITION)):
            RUBBISH_LEFT_BOT_X = RUBBISH_POSITION[i][0] * 50 - 50
            RUBBISH_LEFT_BOT_Y = RUBBISH_POSITION[i][1] * 50 - 50
            self.goal = self.batch.add(
                4, pyglet.gl.GL_QUADS, None,                       # 4 corners
                ('v2f', [RUBBISH_LEFT_BOT_X, RUBBISH_LEFT_BOT_Y,   # location
                         RUBBISH_LEFT_BOT_X, RUBBISH_LEFT_BOT_Y + 50,
                         RUBBISH_LEFT_BOT_X + 50, RUBBISH_LEFT_BOT_Y + 50,
                         RUBBISH_LEFT_BOT_X + 50, RUBBISH_LEFT_BOT_Y]),
                ('c3B', (RUBBISH_COLOR) * 4))                      # color
        print('Rubbshi Position: ', RUBBISH_POSITION)

if __name__ == '__main__':
    env = BotEnv()
    TotalStep = 1
    TurnStep = 0
    turn = 1
    file.write("Turn     " + "Block     " + "Rubbish     " + "Hit     " + "TurnStep     " + "TotalStep     " + "\n")
    while turn <= 1000:
        while len(RUBBISH_POSITION) > 5:
            env.render()
            #env.step(env.sample_action())
            env.step(env.algorithm_three())
            #alpha = (TotalStep/(TotalStep + 1)) * alpha
            print("turn = ", turn, "TotalStep = ", TotalStep, "TurnStep = ", TurnStep)
            TotalStep += 1
            TurnStep += 1
        file.write(str(turn) +"        "+ str(BLOCK_NUM) +"        "+ str(RUBBISH_NUM) +"          "+ str(hit_num) + "        "+ str(TurnStep) + "           " + str(TotalStep) + '\n')
        turn += 1
        TurnStep = 0
        RUBBISH_POSITION += CLEAN_POSITION
        CLEAN_POSITION = []
    file.close() 