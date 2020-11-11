# 2020_DACON_Hackerton
2020 DACON 블록 공정 해커톤 프로젝트입니다! (with 정다운, 심효민)

대회 사이트 : https://dacon.io/competitions/official/235612/overview/

## 목적
각 데이터는 주어져 있으며, 언제 공장을 가동하고 멈출지, 그리고 검사를 할지를 판단하는 모델을 학습하는 것이 주목적입니다.
- 정해진 수요에 맞춰 최적 블럭 장난감 생산 공정 설계
- AI 기반 알고리즘으로 공정 계획을 만들어 csv 파일 제출
따라서, 유전 알고리즘과 DQN을 기반으로 한 모델을 각각 만들고, 두 모델에 대한 학습을 진행했습니다.

## 파일
1. Input Data : 입력 데이터입니다.
2. Process_Models : 실질적인 코드가 들어 있습니다.
3, 4. 테스트 케이스 모음, 테스트 케이스 백업 : 직접 테스트하여 본 여러가지 이력들이 남아 있습니다.

## 코드 설명 (Process_Models 내부 코드)
1. Process_Optimization : 유전 알고리즘을 기반으로, 기본 Baseline 파일을 응용한 모델
2. Process_Optimization_with_Keras : 위 모델을 Keras로 새롭게 구현한 모델
3. DQN_Models : DQN 알고리즘을 기반으로 새롭게 설계한 모델

### < Process_Optimization >
#### Genome.py
 유전 알고리즘의 게놈과 관련된 py 파일로 Genome 클래스가 정의되어 있습니다.
 유전 알고리즘의 신경망이 설계되어 있습니다.
 - def __init__(self)
 - def update_mask(self)
 - def forward(self, inputs)
 - def sigmoid(self, x), softmax(self, x), linear(self, x)
 - def create_order(self, order)
 - def predict(self, order)
 - def genome_score(genome)

#### Simulator.py
 공정 과정과 관련된 py 파일로 Simulator 클래스가 정의되어 있습니다.
 
 공정 과정에서 score 계산, change와 stop과 같은 이벤트와 관련된 함수가 존재합니다.
 - def __init__(self)
 - def make_init(self)
 - def cal_prt_mol(self, machine_name)
 - def cal_blk(self)
 - def cal_change_stop_time(self)
 - def make_stock_df(self)
 - def get_score(self, df)

#### main.ipynb 
 앞에서 정의한 두 파일에 있는 genome과 simulator 객체를 불러와서 유전 알고리즘 기반의 신경망 모델을 학습시키고 결과를 반환합니다.
 이 때 한 epoch에서 best값이 전체 epoch에서 best값보다 클 경우, 이를 갱신시키고, 작을 경우 갱신시키지 않습니다.
 - import library
 - Feature Engineering & Initial Modeling
 - Model Tuning & Evaluation
 - Conclusion & Discussion

### < DQN_Models >
 - Dqn.py : DQN을 이용하여 단순 mlp모델을 학습시키는 방법으로 submission을 계산합니다.
 - Class Env : 각 step마다 predict한 값을 받아서 reward를 반환합니다.
 - Reward: blk_diff에 해당합니다.
 - Model: 3단 mlp 모델로, loss = mae, activation 함수는 linear, sigmoid, softmax를 사용합니다.
