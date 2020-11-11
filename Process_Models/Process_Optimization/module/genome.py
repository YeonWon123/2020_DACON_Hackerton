import os
import pandas as pd
import numpy as np
from pathlib import Path
from module.simulator import Simulator
import time

simulator = Simulator()
submission_ini = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'sample_submission.csv'))
order_ini = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'order.csv'))

class Genome():
    def __init__(self, score_ini, input_len, output_len_1, output_len_2, h1=50, h2=50, h3=50):
        # 평가 점수 초기화
        self.score = score_ini
        
        # 히든레이어 노드 개수
        self.hidden_layer1 = h1
        self.hidden_layer2 = h2
        self.hidden_layer3 = h3
        
        # Event 신경망 가중치 생성
        self.w1 = np.random.randn(input_len, self.hidden_layer1)
        self.w2 = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w3 = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w4 = np.random.randn(self.hidden_layer3, output_len_1)
        
        # MOL 수량 신경망 가중치 생성
        self.w5 = np.random.randn(input_len, self.hidden_layer1)
        self.w6 = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w7 = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w8 = np.random.randn(self.hidden_layer3, output_len_2)
        
        # Event 종류
        self.mask_1 = np.zeros([5], np.bool) # 가능한 이벤트 검사용 마스크
        self.mask_2 = np.zeros([5], np.bool) # 가능한 이벤트 검사용 마스크
        self.event_map = {0:'CHECK_1', 1:'CHECK_2', 2:'CHECK_3', 3:'CHECK_4', 4:'PROCESS'}
        
        self.check_time_1 = 28    # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        self.process_1 = 0        # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_mode_1 = 0   # 생산 물품 번호 1~4, stop시 0
        self.process_time_1 = 0   # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140

        self.check_time_2 = 28    # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        self.process_2 = 0        # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_mode_2 = 0   # 생산 물품 번호 1~4, stop시 0
        self.process_time_2 = 0   # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140
    
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
    
    def forward(self, inputs):
        # Event 신경망
        net = np.matmul(inputs, self.w1)
        net = self.linear(net)
        net = np.matmul(net, self.w2)
        net = self.linear(net)
        net = np.matmul(net, self.w3)
        net = self.sigmoid(net)
        net = np.matmul(net, self.w4)
        net = self.softmax(net)
        net += 1
        net = net.reshape(1, 10)

        net_1 = net[:, :5] * self.mask_1
        net_2 = net[:, 5:] * self.mask_2
        
        # PROCESS에 약간의 가중치를 더 줌
        net_1[:, 4] = net_1[:, 4] * 1.3
        net_2[:, 4] = net_2[:, 4] * 1.3

        out1_1 = self.event_map[np.argmax(net_1)]
        out1_2 = self.event_map[np.argmax(net_2)]
        
        
        # MOL 수량 신경망
        net = np.matmul(inputs, self.w5)
        net = self.linear(net)
        net = np.matmul(net, self.w6)
        net = self.linear(net)
        net = np.matmul(net, self.w7)
        net = self.sigmoid(net)
        net = np.matmul(net, self.w8)
        net = self.softmax(net)
        net = net.reshape(1, 24)
        out2_1 = np.argmax(net[:, :12]) / 2
        out2_2 = np.argmax(net[:, 12:]) / 2
        
        if out2_1 == 5.5:
            out2_1 = 5.85
        if out2_2 == 5.5:
            out2_2 = 5.85
        
        return out1_1, out1_2, out2_1, out2_2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def linear(self, x):
        return x
    
    def create_order(self, order):
        for i in range(30):
            order.loc[91+i,:] = ['0000-00-00', 0, 0, 0, 0]        
        return order
   
    def predict(self, order):
        order = self.create_order(order)
        self.submission = submission_ini
        self.submission.loc[:, 'PRT_1':'PRT_4'] = 0
        for s in range(self.submission.shape[0]):
            self.update_mask()
            inputs = np.array(order.loc[s//24:(s//24+30), 'BLK_1':'BLK_4']).reshape(-1)
            inputs = np.append(inputs, s%24)
            out1_1, out1_2, out2_1, out2_2 = self.forward(inputs)
            
            if out1_1 == 'CHECK_1':
                if self.process_1 == 1:
                    self.process_1 = 0
                    self.check_time_1 = 28
                self.check_time_1 -= 1
                self.process_mode_1 = 0
                if self.check_time_1 == 0:
                    self.process_1 = 1
                    self.process_time_1 = 0
            elif out1_1 == 'CHECK_2':
                if self.process_1 == 1:
                    self.process_1 = 0
                    self.check_time_1 = 28
                self.check_time_1 -= 1
                self.process_mode_1 = 1
                if self.check_time_1 == 0:
                    self.process_1 =1
                    self.process_time_1 = 0
            elif out1_1 == 'CHECK_3':
                if self.process_1 == 1:
                    self.process_1 = 0
                    self.check_time_1 = 28
                self.check_time_1 -= 1
                self.process_mode_1 = 2
                if self.check_time_1 == 0:
                    self.process_1 = 1
                    self.process_time_1 = 0
            elif out1_1 == 'CHECK_4':
                if self.process_1 == 1:
                    self.process_1 = 0
                    self.check_time_1 = 28
                self.check_time_1 -= 1
                self.process_mode_1 = 3
                if self.check_time_1 == 0:
                    self.process_1 = 1
                    self.process_time_1 = 0
            elif out1_1 == 'PROCESS':
                self.process_time_1 += 1
                if self.process_time_1 == 140:
                    self.process_1 = 0
                    self.check_time_1 = 28

            self.submission.loc[s, 'Event_A'] = out1_1
            if self.submission.loc[s, 'Event_A'] == 'PROCESS':
                self.submission.loc[s, 'MOL_A'] = out2_1
            else:
                self.submission.loc[s, 'MOL_A'] = 0
                
            if out1_2 == 'CHECK_1':
                if self.process_2 == 1:
                    self.process_2 = 0
                    self.check_time_2 = 28
                self.check_time_2 -= 1
                self.process_mode_2 = 0
                if self.check_time_2 == 0:
                    self.process_2 = 1
                    self.process_time_2 = 0
            elif out1_2 == 'CHECK_2':
                if self.process_2 == 1:
                    self.process_2 = 0
                    self.check_time_2 = 28
                self.check_time_2 -= 1
                self.process_mode_2 = 1
                if self.check_time_2 == 0:
                    self.process_2 =1
                    self.process_time_2 = 0
            elif out1_2 == 'CHECK_3':
                if self.process_2 == 1:
                    self.process_2 = 0
                    self.check_time_2 = 28
                self.check_time_2 -= 1
                self.process_mode_2 = 2
                if self.check_time_2 == 0:
                    self.process_2 = 1
                    self.process_time_2 = 0
            elif out1_2 == 'CHECK_4':
                if self.process_2 == 1:
                    self.process_2 = 0
                    self.check_time_2 = 28
                self.check_time_2 -= 1
                self.process_mode_2 = 3
                if self.check_time_2 == 0:
                    self.process_2 = 1
                    self.process_time_2 = 0
            elif out1_2 == 'PROCESS':
                self.process_time_2 += 1
                if self.process_time_2 == 140:
                    self.process_2 = 0
                    self.check_time_2 = 28

            self.submission.loc[s, 'Event_B'] = out1_2
            if self.submission.loc[s, 'Event_B'] == 'PROCESS':
                self.submission.loc[s, 'MOL_B'] = out2_2
            else:
                self.submission.loc[s, 'MOL_B'] = 0
                
        # 23일간 MOL = 0
        self.submission.loc[:24*23, 'MOL_A'] = 0
        self.submission.loc[:24*23, 'MOL_B'] = 0
        
        # 변수 초기화
        self.check_time_1 = 28    # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        self.process_1 = 0        # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_mode_1 = 0   # 생산 물품 번호 1~4, stop시 0
        self.process_time_1 = 0   # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140

        self.check_time_2 = 28    # 28시간 검사를 완료했는지 검사, CHECK Event시 -1, processtime_time >=98 이면 28
        self.process_2 = 0        # 생산 가능 여부, 0 이면 28 시간 검사 필요
        self.process_mode_2 = 0   # 생산 물품 번호 1~4, stop시 0
        self.process_time_2 = 0   # 생산시간이 얼마나 지속되었는지 검사, PROCESS +1, CHANGE +1, 최대 140
        
        return self.submission    
    
def genome_score(genome):
    submission = genome.predict(order_ini)    
    genome.submission = submission    
    genome.score, _ = simulator.get_score(submission)    
    return genome