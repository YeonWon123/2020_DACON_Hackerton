import os
import pandas as pd
import numpy as np
from pathlib import Path
from module.simulator import Simulator
import time
from tensorflow import keras
import datetime

simulator = Simulator()
submission_ini = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'sample_submission.csv'))
order_ini = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'order.csv'))
max_count_ini = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'max_count.csv'))
stock_ini = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'stock.csv'))

order_values = order_ini.values[:, 1:]

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


day_len = submission_ini.shape[0]

order = order_ini.copy()
for i in range(35):
    order.loc[91+i, :] = ['0000-00-00', 0, 0, 0, 0]

all_states = np.zeros((day_len + 1, 125))

for i in range(day_len + 1):
    all_states[i][0:124] = np.array(order.loc[i//24:(i//24+30), 'BLK_1':'BLK_4']).reshape(-1)
    all_states[i][124] = i % 24

mol_stock = np.zeros(4)
blk_stock = np.zeros(4)

for i in range(4):
    mol_stock[i] = int(stock_ini[f'MOL_{i + 1}'])
    blk_stock[i] = int(stock_ini[f'BLK_{i + 1}'])


def F(x, a): return 1 - x/a if x < a else 0

N = sum([sum(order[f'BLK_{i}']) for i in range(1,5)])
M = day_len * 2

event_map = {0:'CHECK_1', 1:'CHECK_2', 2:'CHECK_3', 3:'CHECK_4', 4:'PROCESS'}

max_count = {0:140.59, 1:140.80, 2:141.01}

prt_b_stock = 258
        
class Genome():
    def __init__(self, score_ini, input_len, output_len_1, output_len_2, h1=50, h2=50, h3=50):
        
        # 평가 점수 초기화
        self.score = score_ini

        self.model1 = keras.models.Sequential()
        self.model1.add(keras.layers.Dense(h1, input_shape=(input_len,), activation='linear', use_bias=False))
        self.model1.add(keras.layers.Dense(h2, activation='linear', use_bias=False))
        self.model1.add(keras.layers.Dense(h3, activation='sigmoid', use_bias=False))
        self.model1.add(keras.layers.Dense(output_len_1, activation='softmax', use_bias=False))
        self.model1.compile()

        self.model2 = keras.models.Sequential()
        self.model2.add(keras.layers.Dense(h1, input_shape=(input_len,), activation='linear', use_bias=False))
        self.model2.add(keras.layers.Dense(h2, activation='linear', use_bias=False))
        self.model2.add(keras.layers.Dense(h3, activation='sigmoid', use_bias=False))
        self.model2.add(keras.layers.Dense(output_len_2, activation='softmax', use_bias=False))
        self.model2.compile()

        # Event 종류
        self.mask_1 = np.zeros([5], np.bool)
        self.mask_2 = np.zeros([5], np.bool)

    def update_mask(self):
        self.mask_1[:] = False
        if self.process_1 == 0:
            if self.check_time_1 == 28:
                self.mask_1[:4] = True
            if self.check_time_1 < 28:
                self.mask_1[self.process_mode_1] = True
        if self.process_1 == 1:
            self.mask_1[4] = True
            if self.process_time_1 > 98:
                self.mask_1[:4] = True

        self.mask_2[:] = False
        if self.process_2 == 0:
            if self.check_time_2 == 28:
                self.mask_2[:4] = True
            if self.check_time_2 < 28:
                self.mask_2[self.process_mode_2] = True
        if self.process_2 == 1:
            self.mask_2[4] = True
            if self.process_time_2 > 98:
                self.mask_2[:4] = True
   
    def predict(self):
        predicts_1 = self.model1.predict(all_states)
        predicts_2 = self.model2.predict(all_states)
        predicts_1 += 1
        predicts_2 += 1

        p = 0
        q = 0

        # 변수 초기화
        self.check_time_1 = 28    # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        self.process_1 = 0        # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_mode_1 = 0   # 생산 물품 번호 1~4, stop시 0
        self.process_time_1 = 0   # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140
        self.global_process_time_1 = 0

        self.check_time_2 = 28    # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        self.process_2 = 0        # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_mode_2 = 0   # 생산 물품 번호 1~4, stop시 0
        self.process_time_2 = 0   # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140
        self.global_process_time_2 = 0

        self.mol_stock = mol_stock.copy()
        self.blk_stock = blk_stock.copy()

        self.process_A = np.zeros((day_len, 4))
        self.process_B = np.zeros((day_len, 4))
        self.process_num = 0
        self.process_today_A = 0
        self.process_today_B = 0

        for s in range(day_len):
            self.update_mask()
            
            day = s // 24
            time = s % 24

            if day <= 30:
                month = 0
            elif day <= 61:
                month = 1
            else:
                month = 2

            if time == 0:
                self.process_today_A = 0
                self.process_today_B = 0

            # out 1 process
            out1_1 = predicts_1[s][:5] * self.mask_1
            out1_2 = predicts_1[s][5:] * self.mask_2

            out1_1[4] = out1_1[4] * 1.2
            out1_2[4] = out1_2[4] * 1.2

            out1_1 = int(np.argmax(out1_1))
            out1_2 = int(np.argmax(out1_2))

            # out 2 process
            out2_1 = predicts_2[s][:12]
            out2_2 = predicts_2[s][12:]

            out2_1 = float(np.argmax(out2_1)) / 2
            out2_2 = float(np.argmax(out2_2)) / 2

            # 최댓값일 경우 5.85로 최적화
            if out2_1 == 5.5:
                out2_1 = min(6.66, max_count[month] - self.process_today_A)
            if out2_2 == 5.5:
                out2_2 = min(6.66, max_count[month] - self.process_today_B)
            #     out2_1 = 5.85
            # if out2_2 == 5.5:
            #     out2_2 = 5.85
            
            if out1_1 == 0:
                if self.process_1 == 1:
                    self.process_1 = 0
                    self.check_time_1 = 28
                self.check_time_1 -= 1
                self.process_mode_1 = 0
                if self.check_time_1 == 0:
                    self.process_1 = 1
                    self.process_time_1 = 0
            elif out1_1 == 1:
                if self.process_1 == 1:
                    self.process_1 = 0
                    self.check_time_1 = 28
                self.check_time_1 -= 1
                self.process_mode_1 = 1
                if self.check_time_1 == 0:
                    self.process_1 =1
                    self.process_time_1 = 0
            elif out1_1 == 2:
                if self.process_1 == 1:
                    self.process_1 = 0
                    self.check_time_1 = 28
                self.check_time_1 -= 1
                self.process_mode_1 = 2
                if self.check_time_1 == 0:
                    self.process_1 = 1
                    self.process_time_1 = 0
            elif out1_1 == 3:
                if self.process_1 == 1:
                    self.process_1 = 0
                    self.check_time_1 = 28
                self.check_time_1 -= 1
                self.process_mode_1 = 3
                if self.check_time_1 == 0:
                    self.process_1 = 1
                    self.process_time_1 = 0
            elif out1_1 == 4:
                self.process_time_1 += 1
                self.global_process_time_1 += 1
                if self.process_time_1 == 140:
                    self.process_1 = 0
                    self.check_time_1 = 28
                if self.global_process_time_1 >= 48:
                    for i in range(4):
                        self.mol_stock[i] += self.process_A[self.global_process_time_1 - 48][i] * 0.975

            if out1_1 == 4:
                # 16~23일간은 mol2만 생산 가능
                if (self.process_mode_1 == 1 and s >= 24*16 and self.process_num + out2_1 < prt_b_stock) or s >= 24*23:
                    self.process_A[self.global_process_time_1][self.process_mode_1] += out2_1
                    self.process_today_A += out2_1
                    self.process_num += out2_1


            if out1_2 == 0:
                if self.process_2 == 1:
                    self.process_2 = 0
                    self.check_time_2 = 28
                self.check_time_2 -= 1
                self.process_mode_2 = 0
                if self.check_time_2 == 0:
                    self.process_2 = 1
                    self.process_time_2 = 0
            elif out1_2 == 1:
                if self.process_2 == 1:
                    self.process_2 = 0
                    self.check_time_2 = 28
                self.check_time_2 -= 1
                self.process_mode_2 = 1
                if self.check_time_2 == 0:
                    self.process_2 =1
                    self.process_time_2 = 0
            elif out1_2 == 2:
                if self.process_2 == 1:
                    self.process_2 = 0
                    self.check_time_2 = 28
                self.check_time_2 -= 1
                self.process_mode_2 = 2
                if self.check_time_2 == 0:
                    self.process_2 = 1
                    self.process_time_2 = 0
            elif out1_2 == 3:
                if self.process_2 == 1:
                    self.process_2 = 0
                    self.check_time_2 = 28
                self.check_time_2 -= 1
                self.process_mode_2 = 3
                if self.check_time_2 == 0:
                    self.process_2 = 1
                    self.process_time_2 = 0
            elif out1_2 == 4:
                self.process_time_2 += 1
                self.global_process_time_2 += 1  
                if self.process_time_2 == 140:
                    self.process_2 = 0
                    self.check_time_2 = 28
                if self.global_process_time_2 >= 48:
                    for i in range(4):
                        self.mol_stock[i] += self.process_B[self.global_process_time_2 - 48][i] * 0.975

            if out1_2 == 4:
                # 23일간은 mol2만 생산 가능
                if (self.process_mode_2 == 1 and s >= 24*16 and self.process_num + out2_2 < prt_b_stock) or s >= 24*23:
                    self.process_B[self.global_process_time_2][self.process_mode_2] += out2_2
                    self.process_today_B += out2_2
                    self.process_num += out2_2
            
            if s % 24 == 18:
                for i in range(4):
                    if order_values[day][i] != 0:
                        self.blk_stock[i] += int(self.mol_stock[i] * cut[i] * ratio[month][i])
                        self.mol_stock[i] = 0
                        self.blk_stock[i] -= order_values[day][i]

                        if self.blk_stock[i] < 0:
                            p += abs(self.blk_stock[i])
                        elif self.blk_stock[i] > 0:
                            q += abs(self.blk_stock[i])

        self.score = 50*F(p, 10*N)+20*F(q, 10*N)+30
        return self.score

    def make_submission_file(self, file_name):
        self.submission = submission_ini.copy()
        self.submission.loc[:, 'PRT_1':'PRT_4'] = 0

        predicts_1 = self.model1.predict(all_states)
        predicts_2 = self.model2.predict(all_states)
        predicts_1 += 1
        predicts_2 += 1

        p = 0
        q = 0

        # 변수 초기화
        self.check_time_1 = 28    # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        self.process_1 = 0        # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_mode_1 = 0   # 생산 물품 번호 1~4, stop시 0
        self.process_time_1 = 0   # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140
        self.global_process_time_1 = 0

        self.check_time_2 = 28    # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        self.process_2 = 0        # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_mode_2 = 0   # 생산 물품 번호 1~4, stop시 0
        self.process_time_2 = 0   # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140
        self.global_process_time_2 = 0

        self.mol_stock = mol_stock.copy()
        self.blk_stock = blk_stock.copy()

        self.process_A = np.zeros((day_len, 4))
        self.process_B = np.zeros((day_len, 4))
        self.process_num = 0
        self.process_today_A = 0
        self.process_today_B = 0

        for s in range(day_len):
            self.update_mask()

            day = s // 24
            time = s % 24

            if day <= 30:
                month = 0
            elif day <= 61:
                month = 1
            else:
                month = 2

            if time == 0:
                self.process_today_A = 0
                self.process_today_B = 0

            # out 1 process
            out1_1 = predicts_1[s][:5] * self.mask_1
            out1_2 = predicts_1[s][5:] * self.mask_2

            out1_1[4] = out1_1[4] * 1.1
            out1_2[4] = out1_2[4] * 1.1

            out1_1 = int(np.argmax(out1_1))
            out1_2 = int(np.argmax(out1_2))

            # out 2 process
            out2_1 = predicts_2[s][:12]
            out2_2 = predicts_2[s][12:]

            out2_1 = float(np.argmax(out2_1)) / 2
            out2_2 = float(np.argmax(out2_2)) / 2

            # 최댓값일 경우 5.85로 최적화
            if out2_1 == 5.5:
                out2_1 = min(6.66, max_count[month] - self.process_today_A)
            if out2_2 == 5.5:
                out2_2 = min(6.66, max_count[month] - self.process_today_B)
            #     out2_1 = 5.85
            # if out2_2 == 5.5:
            #     out2_2 = 5.85

            if out1_1 == 0:
                if self.process_1 == 1:
                    self.process_1 = 0
                    self.check_time_1 = 28
                self.check_time_1 -= 1
                self.process_mode_1 = 0
                if self.check_time_1 == 0:
                    self.process_1 = 1
                    self.process_time_1 = 0
            elif out1_1 == 1:
                if self.process_1 == 1:
                    self.process_1 = 0
                    self.check_time_1 = 28
                self.check_time_1 -= 1
                self.process_mode_1 = 1
                if self.check_time_1 == 0:
                    self.process_1 =1
                    self.process_time_1 = 0
            elif out1_1 == 2:
                if self.process_1 == 1:
                    self.process_1 = 0
                    self.check_time_1 = 28
                self.check_time_1 -= 1
                self.process_mode_1 = 2
                if self.check_time_1 == 0:
                    self.process_1 = 1
                    self.process_time_1 = 0
            elif out1_1 == 3:
                if self.process_1 == 1:
                    self.process_1 = 0
                    self.check_time_1 = 28
                self.check_time_1 -= 1
                self.process_mode_1 = 3
                if self.check_time_1 == 0:
                    self.process_1 = 1
                    self.process_time_1 = 0
            elif out1_1 == 4:
                self.process_time_1 += 1
                self.global_process_time_1 += 1
                if self.process_time_1 == 140:
                    self.process_1 = 0
                    self.check_time_1 = 28
                if self.global_process_time_1 >= 48:
                    for i in range(4):
                        self.mol_stock[i] += self.process_A[self.global_process_time_1 - 48][i] * 0.975

            if out1_1 == 4:
                # 16~23일간은 mol2만 생산 가능
                if (self.process_mode_1 == 1 and s >= 24*16 and self.process_num + out2_1 < prt_b_stock) or s >= 24*23:
                    self.process_A[self.global_process_time_1][self.process_mode_1] += out2_1
                    self.process_today_A += out2_1
                    self.process_num += out2_1
                else:
                    out2_1 = 0
            else:
                out2_1 = 0

            if out1_2 == 0:
                if self.process_2 == 1:
                    self.process_2 = 0
                    self.check_time_2 = 28
                self.check_time_2 -= 1
                self.process_mode_2 = 0
                if self.check_time_2 == 0:
                    self.process_2 = 1
                    self.process_time_2 = 0
            elif out1_2 == 1:
                if self.process_2 == 1:
                    self.process_2 = 0
                    self.check_time_2 = 28
                self.check_time_2 -= 1
                self.process_mode_2 = 1
                if self.check_time_2 == 0:
                    self.process_2 =1
                    self.process_time_2 = 0
            elif out1_2 == 2:
                if self.process_2 == 1:
                    self.process_2 = 0
                    self.check_time_2 = 28
                self.check_time_2 -= 1
                self.process_mode_2 = 2
                if self.check_time_2 == 0:
                    self.process_2 = 1
                    self.process_time_2 = 0
            elif out1_2 == 3:
                if self.process_2 == 1:
                    self.process_2 = 0
                    self.check_time_2 = 28
                self.check_time_2 -= 1
                self.process_mode_2 = 3
                if self.check_time_2 == 0:
                    self.process_2 = 1
                    self.process_time_2 = 0
            elif out1_2 == 4:
                self.process_time_2 += 1
                self.global_process_time_2 += 1
                if self.process_time_2 == 140:
                    self.process_2 = 0
                    self.check_time_2 = 28
                if self.global_process_time_2 >= 48:
                    for i in range(4):
                        self.mol_stock[i] += self.process_B[self.global_process_time_2 - 48][i] * 0.975

            if out1_2 == 4:
                # 23일간은 mol2만 생산 가능
                if (self.process_mode_2 == 1 and s >= 24*16 and self.process_num + out2_2 < prt_b_stock) or s >= 24*23:
                    self.process_B[self.global_process_time_2][self.process_mode_2] += out2_2
                    self.process_today_B += out2_2
                    self.process_num += out2_2
                else:
                    out2_2 = 0
            else:
                out2_2 = 0

            if s % 24 == 18:
                for i in range(4):
                    if order_values[day][i] != 0:
                        self.blk_stock[i] += int(self.mol_stock[i] * cut[i] * ratio[month][i])
                        self.mol_stock[i] = 0
                        self.blk_stock[i] -= order_values[day][i]

                        if self.blk_stock[i] < 0:
                            p += abs(self.blk_stock[i])
                        elif self.blk_stock[i] > 0:
                            q += abs(self.blk_stock[i])

            self.submission.loc[s, 'Event_A'] = event_map[out1_1]
            self.submission.loc[s, 'Event_B'] = event_map[out1_2]

            self.submission.loc[s, 'MOL_A'] = out2_1
            self.submission.loc[s, 'MOL_B'] = out2_2

        _, df_stock = simulator.get_score(self.submission)

        # PRT 개수 계산
        PRTs = df_stock[['PRT_1', 'PRT_2', 'PRT_3', 'PRT_4']].values
        PRTs = (PRTs[:-1] - PRTs[1:])[24*23:]
        PRTs = np.ceil(PRTs * 1.1)
        PAD = np.zeros((24*23+1, 4))
        PRTs = np.append(PRTs, PAD, axis=0).astype(int)

        # Submission 파일에 PRT 입력
        self.submission.loc[:, 'PRT_1':'PRT_4'] = PRTs
        self.submission.to_csv(file_name, index=False)

        return self.submission

    def copy_weights(self, genome):
        self.model1.set_weights(genome.model1.get_weights())
        self.model2.set_weights(genome.model2.get_weights())

    def get_weights(self):
        return self.model1.get_weights(), self.model2.get_weights()

    def set_weights(self, weights):
        self.model1.set_weights(weights[0])
        self.model2.set_weights(weights[1])





