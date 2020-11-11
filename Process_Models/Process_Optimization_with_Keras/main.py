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
PROB_MUTATION = 0.4                    # 돌연변이
REVERSE = True                        # 배열 순서 (False: ascending order, True: descending order)

score_ini = 10                         # 초기 점수
input_length = 125                     # 입력 데이터 길이
output_length_1 = 5 * 2                # Event (CHECK_1~4, PROCESS)
output_length_2 = 12 * 2                   # MOL(0~5.5, step:0.5)
h1 = 50                                # 히든레이어1 노드 수
h2 = 50                                # 히든레이어2 노드 수
h3 = 50                                # 히든레이어3 노드 수
EPOCHS = 300                            # 반복 횟수

mut_mean = 0
mut_stddev = 0.2

genomes = []
best_genome_weights = []

for _ in range(N_POPULATION):
    genome = Genome(score_ini, input_length, output_length_1, output_length_2, h1, h2, h3)
    genomes.append(genome)

#%%
n_gen = 1
high_score_history = []
mean_score_history = []

best_gen = None
best_score_ever = 0
#%%
while n_gen <= EPOCHS:
    print('EPOCH', n_gen, datetime.datetime.now())
    for idx, genome in enumerate(genomes):
        genome.predict()

    genomes.sort(key=lambda x: x.score, reverse=REVERSE)

    # 평균
    mean_score = 0
    for i in range(N_POPULATION):
        mean_score += genomes[i].score
    mean_score /= N_POPULATION
    mean_score_history.append([n_gen, mean_score])

    # 최고
    high_score_history.append([n_gen, genomes[0].score])

    # 최고 모델 저장
    if genomes[0].score > best_score_ever:
        best_score_ever = genomes[0].score
        best_gen = Genome(score_ini, input_length, output_length_1, output_length_2, h1, h2, h3)
        best_gen.copy_weights(genomes[0])

    print('EPOCH #%s\tHistory Best Score: %s\tBest Score: %s\tMean Score: %s' % (n_gen, best_score_ever, genomes[0].score, mean_score))

    # best model weight 저장
    best_genome_weights = []
    for i in range(N_BEST):
        best_genome_weights.append(genomes[0].get_weights())

    # CHILDREN 생성
    for i in range(N_CHILDREN):
        new_weights = deepcopy(best_genome_weights[0])
        a_weights = random.choice(best_genome_weights)
        b_weights = random.choice(best_genome_weights)

        # model_num, weight_num
        for m in range(len(new_weights)):
            for w in range(len(new_weights[m])):
                for j in range(new_weights[m][w].shape[0]):
                    if len(new_weights[m][w].shape) > 1:
                        cut = np.random.randint(new_weights[m][w].shape[1])
                        new_weights[m][w][j, :cut] = a_weights[m][w][j, :cut]
                        new_weights[m][w][j, cut:] = b_weights[m][w][j, cut:]
                    else:
                        new_weights[m][w][j] = random.choice((a_weights[m][w][j], b_weights[m][w][j]))

        best_genome_weights.append(new_weights)

    # 모델 초기화 and 돌연변이

    for i in range(len(best_genome_weights)):
        genomes[i].set_weights(best_genome_weights[i])

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
best_gen.make_submission_file("submit.csv")