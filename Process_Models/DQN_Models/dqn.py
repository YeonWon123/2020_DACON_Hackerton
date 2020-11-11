import random
import datetime
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


class Simulator:
    def __init__(self):

        self.sample_submission = pd.read_csv('module/sample_submission.csv')
        self.max_count = pd.read_csv('module/max_count.csv')
        self.stock = pd.read_csv('module/stock.csv')
        self.order = pd.read_csv('module/order.csv')

        cut = {f'BLK_{i}': 506 if i <= 2 else 400 for i in range(1,5) }

        ratio = {}

        ratio['BLK_1'] = {}
        ratio['BLK_1'][4] = 0.851
        ratio['BLK_1'][5] = 0.851
        ratio['BLK_1'][6] = 0.851

        ratio['BLK_2'] = {}
        ratio['BLK_2'][4] = 0.901
        ratio['BLK_2'][5] = 0.901
        ratio['BLK_2'][6] = 0.901

        ratio['BLK_3'] = {}
        ratio['BLK_3'][4] = 0.710
        ratio['BLK_3'][5] = 0.742
        ratio['BLK_3'][6] = 0.759

        ratio['BLK_4'] = {}
        ratio['BLK_4'][4] = 0.700
        ratio['BLK_4'][5] = 0.732
        ratio['BLK_4'][6] = 0.749

        self.cut = cut
        self.ratio = ratio

        order_dic = { }
        order = self.order

        for time, BLK_1, BLK_2, BLK_3, BLK_4 in zip(order['time'],order['BLK_1'],order['BLK_2'],order['BLK_3'],order['BLK_4']):

            order_count_time = str(pd.to_datetime(time) + pd.Timedelta(hours=18))
            order_dic[order_count_time] = {}

            order_dic[order_count_time][1] = BLK_1
            order_dic[order_count_time][2] = BLK_2
            order_dic[order_count_time][3] = BLK_3
            order_dic[order_count_time][4] = BLK_4

        self.order_dic = order_dic

    def make_init(self):

        PRT_dic = {time : {i : 0 for i in range(1,5)} for time in self.sample_submission['time']}
        MOL_dic = {time : {i : 0 for i in range(1,5)} for time in self.sample_submission['time']}
        BLK_dic = {time : {i : 0 for i in range(1,5)} for time in self.sample_submission['time']}

        ## 4/1일 00:00:00에 기초재고를 추가 
        for i in range(1,5):
            PRT_dic['2020-04-01 00:00:00'][i] = int(self.stock[f'PRT_{i}'])
            MOL_dic['2020-04-01 00:00:00'][i] = int(self.stock[f'MOL_{i}'])
            BLK_dic['2020-04-01 00:00:00'][i] = int(self.stock[f'BLK_{i}'])

        self.PRT_dic = PRT_dic
        self.MOL_dic = MOL_dic
        self.BLK_dic = BLK_dic


    def cal_prt_mol(self,machine_name):

        df = self.df

        # PRT 개수와 MOL 개수 구하기
        process_list = []
        for time, event, mol in zip(self.sample_submission['time'],df[f'Event_{machine_name}'],df[f'MOL_{machine_name}']):

            # check한 몰의 개수만큼 PRT로
            try:
                val = int(event[-1])
            except:
                pass

            if event == 'PROCESS':
                process_list.append((time,mol,val))

            self.PRT_dic[time][val] += -mol

        for p_start, p_end in zip(process_list[:-48],process_list[48:]):

            p_start_time, p_start_mol, p_start_number = p_start
            p_end_time, p_end_mol, p_end_number = p_end

            self.MOL_dic[p_end_time][p_start_number] += p_start_mol * 0.975


    def cal_blk(self):

        PRT_dic = self.PRT_dic
        MOL_dic = self.MOL_dic
        BLK_dic = self.BLK_dic
        order_dic = self.order_dic
        ratio = self.ratio
        cut = self.cut

        PRT_stock_dic = {time : {i : 0 for i in range(1,5)} for time in self.sample_submission['time']}
        MOL_stock_dic = {time : {i : 0 for i in range(1,5)} for time in self.sample_submission['time']}
        BLK_stock_dic = {time : {i : 0 for i in range(1,5)} for time in self.sample_submission['time']}

        blk_diffs = []
        previous_time = [self.sample_submission['time'][0]] + list(self.sample_submission['time'])

        for time, previous in zip(self.sample_submission['time'], previous_time[:-1]):

            month = int(time[6])

            for i in range(1,5):

                if str(time) == '2020-04-01 00:00:00':
                    PRT_stock_dic[time][i] = PRT_dic[time][i]
                    MOL_stock_dic[time][i] = MOL_dic[time][i]
                    BLK_stock_dic[time][i] = BLK_dic[time][i]

                else:
                    PRT_stock_dic[time][i] = PRT_stock_dic[previous][i] + PRT_dic[time][i]
                    MOL_stock_dic[time][i] = MOL_stock_dic[previous][i] + MOL_dic[time][i]
                    BLK_stock_dic[time][i] = BLK_stock_dic[previous][i] + BLK_dic[time][i]

                    if int(time[11:13]) == 18:

                        val = order_dic[time][i]

                        if val > 0:
                            mol_number = i
                            mol = MOL_stock_dic[time][i]
                            MOL_stock_dic[time][i] = 0

                            blk_gen = int(mol*ratio[f'BLK_{i}'][month]*cut[f'BLK_{i}'])
                            blk_stock = BLK_stock_dic[time][i] + blk_gen
                            blk_diff = blk_stock - val

                            BLK_stock_dic[time][i] = blk_diff
                            blk_diffs.append(blk_diff)

        self.PRT_stock_dic = PRT_stock_dic
        self.MOL_stock_dic = MOL_stock_dic
        self.BLK_stock_dic = BLK_stock_dic
        self.blk_diffs = blk_diffs

    def F(self, x, a): return 1 - x/a if x < a else 0

    def cal_change_stop_time(self):

        df = self.df

        change_type = {'A':'', 'B':''}
        change_num = 0
        change_time = 0
        stop_num = 0
        stop_time = 0
        previous_event = {'A':'', 'B':''}
        for row in df.iterrows():
            for machine in ['A', 'B']:
                if 'CHANGE' in row[1][f'Event_{machine}']:
                    change_time += 1
                    if change_type[machine] != row[1][f'Event_{machine}'][-2:]:
                        change_num += 1
                        change_type[machine] = row[1][f'Event_{machine}'][-2:]

                if 'STOP' == row[1][f'Event_{machine}']:
                    stop_time += 1
                    if previous_event[machine] != 'STOP':
                        stop_num += 1

                previous_event[machine] = row[1][f'Event_{machine}']
        return change_time, change_num, stop_time, stop_num

    def cal_score(self):

        p = 0
        q = 0
        for item in self.blk_diffs:
            if item < 0:
                p = p + abs(item)
            if item > 0:
                q = q + abs(item)

        N = sum([sum(self.order[f'BLK_{i}']) for i in range(1,5)])
        M = len(self.df) * 2

        c, c_n, s, s_n = self.cal_change_stop_time()

        self.score = 50*self.F(p, 10*N)+20*self.F(q, 10*N)+\
                20*self.F(c, M)/(1+0.1*c_n) + 10*self.F(s, M)/(1 + 0.1*s_n)

        self.p = p
        self.q = q
        self.N = N
        self.M = M
        self.c = c
        self.c_n = c_n
        self.s = s
        self.s_n = s_n

        return self.score

    def make_stock_df(self):

        PRT_l = {i : [] for i in range(1,5)}
        MOL_l = {i : [] for i in range(1,5)}
        BLK_l = {i : [] for i in range(1,5)}

        for time in self.sample_submission['time']:
            for i in range(1,5):
                PRT_l[i].append(self.PRT_stock_dic[time][i])
                MOL_l[i].append(self.MOL_stock_dic[time][i])
                BLK_l[i].append(self.BLK_stock_dic[time][i])

        df_stock = pd.DataFrame(index = self.sample_submission['time'])

        for i in range(1,5):
            df_stock[f'PRT_{i}'] = PRT_l[i]
        for i in range(1,5):
            df_stock[f'MOL_{i}'] = MOL_l[i]
        for i in range(1,5):
            df_stock[f'BLK_{i}'] = BLK_l[i]

        self.df_stock = df_stock

    def get_score(self,df):

        self.df = df
        self.make_init()
        self.cal_prt_mol('A')
        self.cal_prt_mol('B')
        self.cal_blk()
        self.cal_score()
        self.make_stock_df()

        return self.score, self.df_stock

submission_ini = pd.read_csv('module/sample_submission.csv')
order_ini = pd.read_csv('module/order.csv')

order = order_ini.copy()
for i in range(35):
    order.loc[91+i,:] = ['0000-00-00', 0, 0, 0, 0]


# BLOCK생산 환경
class Env:
    def __init__(self):
        # Event 종류
        self.mask = np.zeros([5], np.bool) # 가능한 이벤트 검사용 마스크
        self.event_map = {0:'CHECK_1', 1:'CHECK_2', 2:'CHECK_3', 3:'CHECK_4', 4:'PROCESS'}
        self.simulator = Simulator()

        self.sample_submission = pd.read_csv('module/sample_submission.csv')
        self.max_count = pd.read_csv('module/max_count.csv')
        self.stock = pd.read_csv('module/stock.csv')
        self.order = pd.read_csv('module/order.csv')

        cut = np.zeros(4) # blk_type

        cut[0] = 506
        cut[1] = 506
        cut[2] = 400
        cut[3] = 400

        ratio = np.zeros((3, 4)) # month, blk_type

        ratio[0][0] = 0.851
        ratio[1][0] = 0.851
        ratio[2][0] = 0.851

        ratio[0][1] = 0.901
        ratio[1][1] = 0.901
        ratio[2][1] = 0.901

        ratio[0][2] = 0.710
        ratio[1][2] = 0.742
        ratio[2][2] = 0.759

        ratio[0][3] = 0.700
        ratio[1][3] = 0.732
        ratio[2][3] = 0.749

        self.cut = cut
        self.ratio = ratio

        self.order = self.order.values[:, 1:]

        self.reset()

    def done(self):
        return self.s >= self.s_max

    def state(self):
        inputs = np.array(order.loc[self.s//24:(self.s//24+30), 'BLK_1':'BLK_4']).reshape(-1)
        inputs = np.append(inputs, self.s%24)
        return inputs

    def states(self):
        all_states = np.zeros((self.submission.shape[0] + 1, 125))

        for i in range(self.submission.shape[0] + 1):
            all_states[i][0:124] = np.array(order.loc[i//24:(i//24+30), 'BLK_1':'BLK_4']).reshape(-1)
            all_states[i][124] = i % 24

        return all_states

    def reset(self):
        self.mask = np.zeros([5], np.bool)
        self.simulator = Simulator()

        self.submission = submission_ini.copy()
        self.submission.loc[:, 'PRT_1':'PRT_4'] = 0
        self.s = 0
        self.s_max = self.submission.shape[0]

        self.check_time = 28
        self.process = 0
        self.process_mode = 0
        self.process_time = 0
        self.last_score = 30

        self.global_process_time = 0

        self.PRT_stock = np.zeros(4)
        self.MOL_stock = np.zeros(4)
        self.BLK_stock = np.zeros(4)

        for i in range(4):
            self.PRT_stock[i] = int(self.stock[f'PRT_{i + 1}'])
            self.MOL_stock[i] = int(self.stock[f'MOL_{i + 1}'])
            self.BLK_stock[i] = int(self.stock[f'BLK_{i + 1}'])

        self.process_A = np.zeros((self.submission.shape[0], 4))
        self.process_B = np.zeros((self.submission.shape[0], 4))

        self.update_mask()

    def step(self, actions):
        day = self.s // 24

        if day <= 30:
            month = 0
        elif day <= 61:
            month = 1
        else:
            month = 2

        action1, action2 = actions

        out1 = self.event_map[action1]
        out2 = action2 / 2

        if out1 == 'CHECK_1':
            if self.process == 1:
                self.process = 0
                self.check_time = 28
            self.check_time -= 1
            self.process_mode = 0
            if self.check_time == 0:
                self.process = 1
                self.process_time = 0
        elif out1 == 'CHECK_2':
            if self.process == 1:
                self.process = 0
                self.check_time = 28
            self.check_time -= 1
            self.process_mode = 1
            if self.check_time == 0:
                self.process = 1
                self.process_time = 0
        elif out1 == 'CHECK_3':
            if self.process == 1:
                self.process = 0
                self.check_time = 28
            self.check_time -= 1
            self.process_mode = 2
            if self.check_time == 0:
                self.process = 1
                self.process_time = 0
        elif out1 == 'CHECK_4':
            if self.process == 1:
                self.process = 0
                self.check_time = 28
            self.check_time -= 1
            self.process_mode = 3
            if self.check_time == 0:
                self.process = 1
                self.process_time = 0
        elif out1 == 'PROCESS':
            self.process_time += 1
            self.global_process_time += 1
            if self.global_process_time >= 48:
                for i in range(4):
                    self.MOL_stock[i] += self.process_A[self.global_process_time - 48][i] * 0.975
                    self.MOL_stock[i] += self.process_B[self.global_process_time - 48][i] * 0.975
            if self.process_time == 140:
                self.process = 0
                self.check_time = 28

        self.submission.loc[self.s, 'Event_A'] = out1
        if self.submission.loc[self.s, 'Event_A'] == 'PROCESS' and self.s >= 24*23:
            self.submission.loc[self.s, 'MOL_A'] = out2
            self.process_A[self.global_process_time][self.process_mode] += out2
            self.process_B[self.global_process_time][self.process_mode] += out2
        else:
            self.submission.loc[self.s, 'MOL_A'] = 0

        self.submission.loc[self.s, 'Event_B'] = self.submission.loc[self.s, 'Event_A']
        self.submission.loc[self.s, 'MOL_B'] = self.submission.loc[self.s, 'MOL_A']

        self.update_mask()
        self.s += 1

        # blk, reward계산
        if self.s % 24 == 18:
            blk_diff = 0
            for i in range(4):
                if self.order[day][i] != 0:

                    self.BLK_stock[i] += int(self.MOL_stock[i] * self.cut[i] * self.ratio[month][i])
                    self.MOL_stock[i] = 0
                    self.BLK_stock[i] -= self.order[day][i]

                    if self.BLK_stock[i] < 0:
                        blk_diff += abs(self.BLK_stock[i]) * 5
                    elif self.BLK_stock[i] > 0:
                        blk_diff += abs(self.BLK_stock[i]) * 2

            if blk_diff != 0:
                blk_diff -= 500000
            reward = -(blk_diff / 50000000)
        else:
            reward = 0

        return reward

    def get_score(self):
        score, stock = self.simulator.get_score(self.submission)
        return score

    def update_mask(self):
        self.mask[:] = False
        if self.process == 0:
            if self.check_time == 28:
                self.mask[:4] = True
            if self.check_time < 28:
                self.mask[self.process_mode] = True
        if self.process == 1:
            self.mask[4] = True
            if self.process_time > 98:
                self.mask[:4] = True


# 모델 변수 설정

input_length = 125                     # 입력 데이터 길이
output_length_1 = 5                    # Event (CHECK_1~4, PROCESS)
output_length_2 = 12                   # MOL(0~5.5, step:0.5)

alpha = 0.01
alpha_decay = 0.01
dropout = 0.3

# 모델 생성
model1 = Sequential()
model1.add(Dense(50, input_shape=(input_length,), activation='linear', use_bias=False))
model1.add(Dense(50, activation='linear', use_bias=False))
model1.add(Dense(50, activation='sigmoid', use_bias=False))
model1.add(Dense(output_length_1, activation='softmax', use_bias=False))

model1.compile(loss='mse', optimizer=Adam(lr=alpha, decay=alpha_decay))

model2 = Sequential()
model2.add(Dense(50, input_shape=(input_length,), activation='linear', use_bias=False))
model2.add(Dense(50, activation='linear', use_bias=False))
model2.add(Dense(50, activation='sigmoid', use_bias=False))
model2.add(Dense(output_length_2, activation='softmax', use_bias=False))

model2.compile(loss='mse', optimizer=Adam(lr=alpha, decay=alpha_decay))


# 학습 변수 설정
EPOCHS = 500
CHECK_EPOCH = 1
THRESHOLD = 90

batch_size = 2000
memory_size = 50000
gamma = 0.98

epsilon = 1

epsilon_min = 0.001
epsilon_decay = 0.95

memory = deque(maxlen=memory_size)
score_history = []

scores = deque(maxlen=CHECK_EPOCH)
avg_scores = []


#%% start train
for e in tf.range(EPOCHS):
    env = Env()
    if e % CHECK_EPOCH == 0:
        print()
        print(f'epoch {e} start, epsilon: ', epsilon, ' time: ', datetime.datetime.now())

    done = False
    i = 0
    # before_time = datetime.datetime.now()

    states = env.states()
    predicts1 = model1.predict(states)
    predicts2 = model2.predict(states)


    reward_sum = 0
    action_dict = defaultdict(int)
    out1_dict = defaultdict(int)
    mask_dict = defaultdict(int)

    while not done:

        # action 선택
        if np.random.random() <= epsilon:
            out1 = np.random.rand(output_length_1)
        else:
            out1 = predicts1[i]

        if np.random.random() <= epsilon:
            out2 = np.random.rand(output_length_2)
        else:
            out2 = predicts2[i]

        env.update_mask()

        out1 += 1
        out2 += 1

        action1 = np.argmax(out1 * env.mask)
        action2 = np.argmax(out2)

        # print(action1)
        action_dict[action1] += 1
        out1_dict[np.argmax(out1)] += 1
        mask_dict[tuple(env.mask)] += 1

        reward = env.step((action1, action2))
        reward_sum += reward

        done = env.done()

        # memory에 저장
        memory.append((states[i][np.newaxis, :], (action1, action2), reward, states[i + 1][np.newaxis, :], done))

        i += 1

    epsilon = max(epsilon_min, epsilon_decay * epsilon)


    score = env.get_score()
    s = env.simulator
    print('부족분 점수: ', 50 * s.F(s.p, 10*s.N))
    print('추가분 점수: ', 20 * s.F(s.q, 10*s.N))
    print('기타 점수: ', 20*s.F(s.c, s.M)/(1+0.1*s.c_n) + 10*s.F(s.s, s.M)/(1 + 0.1*s.s_n))


    scores.append(score)

    mean_score = np.average(scores)
    avg_scores.append(mean_score)

    score_history.append([e, score])

    # early-stop 확인 / 현재 상태 
    if mean_score >= THRESHOLD and e >= CHECK_EPOCH:
        print(f'Ran {e + 1} times. Solved afer {e-CHECK_EPOCH} trials!')
        break
    if e % CHECK_EPOCH == 0:
        print(f'[Episode {e + 1}] - Mean score over last {CHECK_EPOCH} times:{mean_score}')

    # replay    
    x_batch1, y_batch1 = [], []
    x_batch2, y_batch2 = [], []

    minibatch = random.sample(memory, min(len(memory), batch_size))

    state = minibatch[0][0]
    states = np.empty((batch_size, state.shape[1]))
    next_states = np.empty((batch_size, state.shape[1]))

    for i, (state, actions, reward, next_state, done) in enumerate(minibatch):
        states[i] = np.squeeze(state)
        next_states[i] = np.squeeze(next_state)

    y_targets1 = model1.predict(states)
    y_targets2 = model2.predict(states)

    predicts1 = model1.predict(next_states)
    predicts2 = model2.predict(next_states)

    for i, (state, actions, reward, next_state, done) in enumerate(minibatch):
        action1, action2 = actions

        y_targets1[i][action1] = reward if done else reward + gamma * np.max(predicts1[i])
        x_batch1.append(state[0])
        y_batch1.append(y_targets1[i])

        y_targets2[i][action2] = reward if done else reward + gamma * np.max(predicts2[i])
        x_batch2.append(state[0])
        y_batch2.append(y_targets2[i])

    model1.fit(np.array(x_batch1), np.array(y_batch1), batch_size=len(minibatch), verbose=0)
    model2.fit(np.array(x_batch2), np.array(y_batch2), batch_size=len(minibatch), verbose=0)

    env.reset()


#%% show graph
score_history = np.array(score_history)

plt.plot(score_history[:,0], score_history[:,1], '-o', label='score')

plt.legend()
plt.xlim(0, EPOCHS)
plt.ylim(bottom=0)
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.show()


env = Env()
done = False

# submission 파일 만들기
while not env.done():
    state = env.state()[np.newaxis, :]
    # action 선택
    out1 = model1.predict(state)
    out2 = model2.predict(state)

    action1 = np.argmax(out1 * env.mask)
    action2 = np.argmax(out2)

    reward = env.step((action1, action2))


# 재고 계산
simulator = Simulator()
_, df_stock = simulator.get_score(env.submission)

# PRT 개수 계산
PRTs = df_stock[['PRT_1', 'PRT_2', 'PRT_3', 'PRT_4']].values
PRTs = (PRTs[:-1] - PRTs[1:])[24*23:]
PRTs = np.ceil(PRTs * 1.1)
PAD = np.zeros((24*23+1, 4))
PRTs = np.append(PRTs, PAD, axis=0).astype(int)

# Submission 파일에 PRT 입력
env.submission.loc[:, 'PRT_1':'PRT_4'] = PRTs
env.submission.to_csv('submission_dqn.csv', index=False)
