import pandas as pd
import numpy as np
from copy import deepcopy
from module.genome import Genome
import datetime
import random
import matplotlib.pyplot as plt

#%%
N_POPULATION = 250                      # 세대당 생성수
N_BEST = 15                             # 베스트 수
N_CHILDREN = 10                         # 자손 유전자 수
REVERSE = True                          # 배열 순서 (False: ascending order, True: descending order)

score_ini = 10                          # 초기 점수
input_length = 125                      # 입력 데이터 길이
output_length_1 = 5 * 2                 # A, B의 Event (CHECK_1~4, PROCESS)
output_length_2 = 12 * 2                # A, B의 MOL(0~5.5, step:0.5)
h1 = 50                                 # 히든레이어1 노드 수
h2 = 50                                 # 히든레이어2 노드 수
h3 = 50                                 # 히든레이어3 노드 수
EPOCHS = 300                            # 반복 횟수

PROB_MUTATION = 0.4                     # 돌연변이 확률
mut_mean = 0                            # 돌연변이 평균값
mut_stddev = 0.2                        # 돌연변이 표준편차 - learning rate 처럼 작용

genomes = []                            # 각 genome을 저장하는 리스트
best_genome_weights = []                # epoch의 상위 n_best개의 genome의 weights를 저장하는 리스트

for _ in range(N_POPULATION):           # Genome생성
    genome = Genome(score_ini, input_length, output_length_1, output_length_2, h1, h2, h3)
    genomes.append(genome)

#%%
n_gen = 1                               # 현재 epoch 번호 (세대 수)
high_score_history = []                 # epoch마다 최고 score 저장 리스트
mean_score_history = []                 # epoch의 모든 genome의 평균 score저장 리스트

best_gen = None                         # 현재까지의 genome중 가장 좋은 결과인 genome
best_score_ever = 0                     # 현재까지의 score 최댓값
#%%
while n_gen <= EPOCHS:
    print('EPOCH', n_gen, datetime.datetime.now())  # EPOCH번호, 시간 출력
    for idx, genome in enumerate(genomes):          # 각 genome마다 score계산
        genome.predict()

    genomes.sort(key=lambda x: x.score, reverse=REVERSE)  # score 기준으로 sort

    mean_score = 0                                  # 평균값 계산/저장
    for i in range(N_POPULATION):
        mean_score += genomes[i].score
    mean_score /= N_POPULATION
    mean_score_history.append([n_gen, mean_score])

    high_score_history.append([n_gen, genomes[0].score])  # 최댓값 계산/저장

    if genomes[0].score > best_score_ever:                # 최고 모델 저장
        best_score_ever = genomes[0].score
        best_gen = Genome(score_ini, input_length, output_length_1, output_length_2, h1, h2, h3)
        best_gen.copy_weights(genomes[0])

    print('EPOCH #%s\tHistory Best Score: %s\tBest Score: %s\tMean Score: %s' % (n_gen, best_score_ever, genomes[0].score, mean_score))

    best_genome_weights = []                            # best model weight 저장
    for i in range(N_BEST):
        best_genome_weights.append(genomes[0].get_weights())

    # best genome중 부모 a, b를 선택해서 자식 생성
    for i in range(N_CHILDREN):
        new_weights = deepcopy(best_genome_weights[0])
        a_weights = random.choice(best_genome_weights)
        b_weights = random.choice(best_genome_weights)

        # model_num, weight_num
        for m in range(len(new_weights)):  # genome의 model마다
            for w in range(len(new_weights[m])):  # model의 각 layer의 weight마다
                for j in range(new_weights[m][w].shape[0]):
                    if len(new_weights[m][w].shape) > 1:  # 일반적인 weight의 경우
                        cut = np.random.randint(new_weights[m][w].shape[1])
                        new_weights[m][w][j, :cut] = a_weights[m][w][j, :cut]
                        new_weights[m][w][j, cut:] = b_weights[m][w][j, cut:]
                    else:                                  # bias인 경우
                        new_weights[m][w][j] = random.choice((a_weights[m][w][j], b_weights[m][w][j]))

        best_genome_weights.append(new_weights)

    # 다음 generation 생성 시 새로운 genome을 생성하는 것은 오버헤드가 크기 때문에
    # 동일한 genomes의 weights만 업데이트 해서 사용한다

    # 다음 generation의 처음 genome들은 부모/자식을 그대로 사용
    for i in range(len(best_genome_weights)):
        genomes[i].set_weights(best_genome_weights[i])

    # 나머지 자식들은 돌연변이를 적용시켜서 생성한다
    for i in range(len(best_genome_weights), N_POPULATION):
        bgw = best_genome_weights[i % len(best_genome_weights)]
        new_weights = deepcopy(bgw)

        for m in range(len(new_weights)):
            for w in range(len(new_weights[m])):
                if np.random.uniform(0, 1) < PROB_MUTATION:
                    new_weights[m][w] += new_weights[m][w]\
                                         * np.random.normal(mut_mean, mut_stddev, size=new_weights[m][w].shape)\
                                         * np.random.randint(0, 2, new_weights[m][w].shape)

        genomes[i].set_weights(new_weights)

    n_gen += 1

#%%
# Score Graph
high_score_history = np.array(high_score_history)
mean_score_history = np.array(mean_score_history)
plt.plot(high_score_history[:,0], high_score_history[:,1], '-o', label='High')
plt.plot(mean_score_history[:,0], mean_score_history[:,1], '-o', label='Mean')
plt.legend()
plt.xlim(0, EPOCHS)
plt.ylim(bottom=0)
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.show()

#%%
# submission 파일 생성
best_gen.make_submission_file("submit.csv")