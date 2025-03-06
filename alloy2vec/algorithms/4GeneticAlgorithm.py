import time
import warnings
# 需要额外引入参数：同质化
import numpy as np
import pandas as pd
import copy
import pickle
import random
from torch import nn
import torch
import joblib
from utils import feature19
import xgboost as xgb
mean = np.array([0,0,0,0,0,0,0,0,0])
std = np.array([100,100,100,100,100,100,100,100,100])

mean_19= np.array([ 7.19553194e-02, 2.11543009e-01, 7.55467102e+00, -2.35888432e+01,
  1.43134286e+01,  1.50926926e+03,  2.79587517e+03,  1.0814744005999999,
  2.16999634e-01,  5.10667337e+00, -6.89672575e-02,  8.08409556e-03,
  1.12829129e+01,  2.38861923e+02,  1.14445694e+04,  8.13674012e+01,
  3.70283544e-01,  3.56536932e+01,  7.58330745e+00])
std_19 = np.array([3.41608005e-03, 2.40631639e-02, 4.02798976e-01, 6.75503497e+00,
 1.74465170e+00, 4.04824817e+02, 4.92891225e+02, 5.28857547e-04,
 3.24019172e-02, 2.47665980e-01, 5.75368651e-02, 1.01359870e-03,
 6.82565731e-01, 1.65190981e+01, 9.01506998e+02, 4.25281715e+00,
 4.74993416e-02, 5.88496127e+00, 4.94751632e-01])

mean_28= np.array([0,0,0,0,0,0,0,0,0,7.19553194e-02, 2.11543009e-01, 7.55467102e+00, -2.35888432e+01,
  1.43134286e+01,  1.50926926e+03,  2.79587517e+03,  1.0814744005999999,
  2.16999634e-01,  5.10667337e+00, -6.89672575e-02,  8.08409556e-03,
  1.12829129e+01,  2.38861923e+02,  1.14445694e+04,  8.13674012e+01,
  3.70283544e-01,  3.56536932e+01,  7.58330745e+00])
std_28 =  np.array([100,100,100,100,100,100,100,100,100,3.41608005e-03, 2.40631639e-02, 4.02798976e-01, 6.75503497e+00,
 1.74465170e+00, 4.04824817e+02, 4.92891225e+02, 5.28857547e-04,
 3.24019172e-02, 2.47665980e-01, 5.75368651e-02, 1.01359870e-03,
 6.82565731e-01, 1.65190981e+01, 9.01506998e+02, 4.25281715e+00,
 4.74993416e-02, 5.88496127e+00, 4.94751632e-01])
unique_ranges = [(20,84), (0, 8), (0, 24), (10, 18), (0, 10), (0, 12), (0, 12), (6, 12), (0, 10)]
LABEL_MEAN = 1300
LABEL_STD = 100
import json
warnings.filterwarnings("ignore", message="X does not have valid feature names")
with open("data/hash.json", "r") as json_file:
    data_dict = json.load(json_file)
# np.random.seed(42)
# random.seed(42)
def manhattan_distance(row1, row2):
    return np.sum(np.abs(row1 - row2))

class MLP(nn.Module):
    def __init__(self,num_classes,dropout=0.3):
        super().__init__()
        self.layer1 = nn.Linear(num_classes,64)
        self.layer2 = nn.Linear(64,256)
        self.layer3 = nn.Linear(256,64)
        self.layer4 = nn.Linear(64,1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.act(self.layer1(x)))
        x = self.dropout(self.act(self.layer2(x)))
        x = self.dropout(self.act(self.layer3(x)))
        x = self.layer4(x)
        return x

def generate_array():
    array = []
    for start, end in unique_ranges:
        array.append(random.randint(start, end))
    total_sum = sum(array)
    tmp = {0,1,2,3,4,5,6,7,8}
    while total_sum != 100:
        if total_sum < 100:
            index = random.choice(list(tmp))
            if array[index] < unique_ranges[index][1]:  # 确保增加后仍在取值范围内
                array[index] += 1
            else:
                tmp.remove(index)
        else:
            index = random.choice(list(tmp))
            if array[index] > unique_ranges[index][0]:  # 确保减少后仍在取值范围内
                array[index] -= 1
            else:
                tmp.remove(index)
        total_sum = sum(array)
    return array


class GeneticAlgorithm:
    def __init__(self,fitness_model_info,num_generations ,population_size=200, mutation_rate=0.1,  parents_num = 20,envolve_per_generation=100,better_parents_ratio=0.9,eliminate_candidates_num=150,diversity_score=9):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = np.array([generate_array() for i in range(population_size)])
        self.fitness_model_info = fitness_model_info
        self.parents_num = parents_num
        self.envolve_per_generation = envolve_per_generation
        self.better_parents_ratio = better_parents_ratio
        self.eliminate_candidates_num = eliminate_candidates_num
        self.num_generations = num_generations
        self.diversity_score = diversity_score
        if fitness_model_info['type'] == "rf":
            with open(fitness_model_info['pth'], 'rb') as f:
                self.fitness_model = joblib.load(f)
            # self.score = self.fitness_model.feature_importances_[:9]
            self.score = [0.1 for _ in range(9)]
        elif fitness_model_info['type'] == "mlp":
            self.fitness_model = MLP(fitness_model_info['num'])
            self.fitness_model.load_state_dict(torch.load(fitness_model_info['pth']))
            self.fitness_model.eval()
            self.score = [0.1 for _ in range(9)]
            # To do
        elif fitness_model_info['type'] == "xgboost":
            self.fitness_model = xgb.Booster()
            self.fitness_model.load_model(fitness_model_info['pth'])
            # self.score = list(self.fitness_model.get_score(importance_type='gain').values())[:9]
            self.score = [0.1 for _ in range(9)]
        total_sum = sum(self.score)
        self.score = [x / total_sum for x in self.score]
        self.fitness_values = self.fitness(self.population)
        self.original_population = copy.deepcopy(self.population)
        self.original_fitness_values = self.fitness_values
        # self.diversity = np.zeros(population_size)
        # for i in range(len(self.diversity)):
        #     for j in range(i + 1, len(self.diversity)):
        #         tmp = manhattan_distance(self.population[i], self.population[j])
        #         self.diversity[i] += tmp
        #         self.diversity[j] += tmp

    def fitness(self, chromosome):
        if self.fitness_model_info['num'] == 9:
            data = chromosome
        elif self.fitness_model_info['num'] == 19:
            data = np.apply_along_axis(feature19, axis=1, arr=chromosome)
        elif self.fitness_model_info['num'] == 28:
            data = np.hstack([chromosome, np.apply_along_axis(feature19, axis=1, arr=chromosome)])

        if self.fitness_model_info['type'] == "rf":
            prediction = self.fitness_model.predict(data)
        elif self.fitness_model_info['type'] == "mlp":
            if self.fitness_model_info['num'] == 9:
                data = (data - mean) / std
            elif self.fitness_model_info['num'] == 19:
                data = (data - mean_19) / std_19
            elif self.fitness_model_info['num'] == 28:
                data = (data - mean_28) / std_28

            prediction = (self.fitness_model(torch.tensor(data,dtype=torch.float)).squeeze().detach().cpu().numpy()* LABEL_MEAN + LABEL_MEAN)
        elif self.fitness_model_info['type'] == "xgboost":
            prediction = self.fitness_model.predict(xgb.DMatrix(data))
        return prediction

    def select_parents(self):
        sort_values = np.argsort(self.fitness_values)
        # 优秀父母
        better_parents_num = int(self.parents_num * self.better_parents_ratio)
        better_parents = sort_values[:better_parents_num]
        # 差父母
        worse_parents = np.random.choice(sort_values[better_parents_num:],self.parents_num-better_parents_num,False)

        return np.random.choice(np.hstack([better_parents,worse_parents]),2,replace=False)

    def crossover(self, parents):
        # 交叉策略1：根据1-importance的大小视为交叉的概率，交叉为父母二者的平均值
        tmp = []
        for i in range(9):
            if random.random() <= 1-self.score[i]:
                tmp.append(int(
                    (
                        self.population[parents[0]][i] + self.population[parents[1]][i]
                    )/2
                ))
            else:
                tmp.append(self.population[parents[0]][i])
        return np.array(tmp)

    def mutate(self, offspring):
        for i in range(9):
            if np.random.random() < self.mutation_rate:
                offspring[i] = np.random.randint(unique_ranges[i][0], unique_ranges[i][1]+1)
        return offspring

    def evolve(self):
        parents = self.select_parents()
        offspring = self.crossover(parents)
        offspring = self.mutate(offspring)
        for i in range(len(self.population)):
            if manhattan_distance(self.population[i],offspring)<= self.diversity_score:
                return

        # if np.any(np.all(self.population[parents] == offspring, axis=1)):
        #     return
        self.population = np.vstack([self.population, offspring])
        self.fitness_values = np.hstack([self.fitness_values, self.fitness(offspring.reshape(1, -1))])

    def optimize(self):
        start_time = time.time()
        for idx in range(self.num_generations):
            for _ in range(self.envolve_per_generation):
                self.evolve()

            indices_to_remove = np.argsort(self.fitness_values)[-self.eliminate_candidates_num:]
            indices_to_remove = np.random.choice(indices_to_remove,len(self.population)-self.population_size,replace=False)
            # print(len(self.population))
            self.population = np.delete(self.population, indices_to_remove, axis=0)
            self.fitness_values = np.delete(self.fitness_values, indices_to_remove, axis=0)
            # if (idx+1) % 100 == 0:
            #     print(f"Elapsed time after {idx+1} generations: {time.time() - start_time:.2f} seconds")
        self.elapsed_time = f"{time.time() - start_time:.2f}s"

        # best_chromosome = self.population[np.argmax(self.fitness_values)]
        # str_array1 = np.array([','.join(map(str, row)) for row in self.population])
        # str_array2 = np.array([','.join(map(str, row)) for row in self.original_population])
        # common_rows = np.intersect1d(str_array1, str_array2)
        # common_rows_ratio = len(common_rows) / len(self.original_population)
        # print(f"种群进化前后相同比率为：{common_rows_ratio}")
        # return best_chromosome, np.max(self.fitness_values)

    def stactistic(self,path):
        hash_values_population = [hash(tuple(row)) for row in self.population]
        hash_values_original_population = [hash(tuple(row)) for row in self.original_population]
        common_hash_values = set(hash_values_population) & set(hash_values_original_population)
        common_rows_ratio = len(common_hash_values) / len(self.original_population)
        common_hash_values_all = set(hash_values_population) & set(list(data_dict.keys()))
        common_values_all = self.fitness_values[np.where(np.array(hash_values_population) == np.array(common_hash_values_all))]
        mae = np.abs(np.array([data_dict[i] for i in common_hash_values_all]) - common_values_all)
        diversity = np.zeros(self.population_size)
        for i in range(len(diversity)):
            for j in range(i + 1, len(diversity)):
                tmp = manhattan_distance(self.population[i], self.population[j])
                diversity[i] += tmp
                diversity[j] += tmp


        log = {
            "代理模型信息": self.fitness_model_info,
            "种群": self.population[np.argsort(self.fitness_values)].tolist(),
            "种群熔点": np.sort(self.fitness_values).tolist(),
            "种群更新率": 1-common_rows_ratio,
            "熔点降低": f"{(np.mean(self.original_fitness_values) - np.mean(self.fitness_values)):.2f} 摄氏度",
            "L1距离": sum(diversity) / (self.population_size**2 *9),
            "与原数据集交集MAE": np.nan if mae.size > 0 else np.mean(mae),
            "与原数据集交集hash值": str(common_hash_values_all),
            "多样性最低分":diversity_score,
            "消耗时间": self.elapsed_time,
            "超参数": {
                "population_size": self.population_size,
                "mutation_rate": self.mutation_rate,
                "num_generations": self.num_generations,
                "envolve_per_generation": self.envolve_per_generation,
                "eliminate_candidates_num": self.eliminate_candidates_num,
                "parents_num": self.parents_num,
                "better_parents_ratio": self.better_parents_ratio
            }
        }
        with open(path, 'w') as json_file:
            json.dump(log, json_file, indent=4)
        # print(log)



diversity_score = 9
population_size=500
mutation_rate=0.2
num_generations = 1000            # 有多少代
envolve_per_generation = 50    # 在每一代有多少个个体得到进化
eliminate_candidates_num = 200
parents_num = 200                   # 选取父母的个数
better_parents_ratio = 0.7          # 在选取父母配对时，有多少几率选取到优秀父母



for idx in range(5):
    for _model in ["rf","xgboost"]:
        for _feature in [9,19,28]:
            if _model == "mlp":
                for _epoch in [30,50,100,150,200]:
                    model_name = f"{_model}_{_feature}_{_epoch}epoch.pth"
                    fitness_model_info = {
                        "type": f"{_model}",
                        "pth": f"checkpoints/{model_name}",
                        "num": _feature
                    }
                    ga = GeneticAlgorithm(fitness_model_info, num_generations, population_size, mutation_rate,
                                          parents_num,
                                          envolve_per_generation, better_parents_ratio, eliminate_candidates_num,
                                          diversity_score)
                    ga.optimize()
                    ga.stactistic(path=f"logs/{model_name}_{idx}.json")
                    print(f"logs/{model_name}_{idx}.json Saved")
            else:
                if _model == "rf":
                    model_name = f"{_model}_{_feature}.pkl"
                if _model == "xgboost":
                    model_name = f"{_model}_{_feature}.model"

                fitness_model_info = {
                    "type": f"{_model}",
                    "pth": f"checkpoints/{model_name}",
                    "num": _feature
                }
                ga = GeneticAlgorithm(fitness_model_info, num_generations, population_size, mutation_rate, parents_num,
                                      envolve_per_generation, better_parents_ratio, eliminate_candidates_num, diversity_score)
                ga.optimize()
                ga.stactistic(path=f"logs/normal/{model_name}_{idx}.json")
                print(f"logs/{model_name}_{idx}.json Saved")