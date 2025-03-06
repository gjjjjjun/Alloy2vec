import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

MEAN = 1300
STD = 100


# test_sets = []
# test_sets_19 = []
# test_sets_28 = []
# test_set_properties = []
# for i in range(4):
#     test_sets.append(pd.read_csv(rf"data/{i+1}/test.csv"))
#     test_sets_19.append(pd.read_csv(rf"data/{i+1}/test_19.csv"))
#     test_sets_28.append(pd.concat([test_sets[i], test_sets_19[i]], axis=1))
#     test_set_properties.append(pd.read_csv(rf"data/{i+1}/test_property.csv"))
#
# test_set = pd.read_csv(r"C:\Users\Administrator\Desktop\code\Ni\data\test.csv")
# test_set_19 = pd.read_csv(r"C:\Users\Administrator\Desktop\code\Ni\data\test_19.csv")
# test_set_property = pd.read_csv(r"C:\Users\Administrator\Desktop\code\Ni\data\test_property.csv")
# test_set_28 = pd.concat([test_set, test_set_19], axis=1)


class NiDataset(Dataset):
    def __init__(self, data, label,scalar_mean,scalar_std, mean=MEAN, std=STD):
        self.data = (data - scalar_mean) / scalar_std
        self.labels = (label - mean) / std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = torch.tensor(self.data.iloc[index],dtype=torch.float)
        label = torch.tensor(self.labels.iloc[index],dtype=torch.float)
        return sample, label, torch.tensor(index)

class NiPretrainDataset(Dataset):
    def __init__(self, data):
        self.input = torch.tensor(data.iloc[:,:9].values,dtype=torch.int64)
        self.input[:,0] -= 20
        self.input[:,3] -= 10
        self.input[:,7] -= 6
        self.label = torch.tensor(data.iloc[:,9:].values,dtype=torch.float)
    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.label[index]

# def predict_and_save(model,model_name,model_type,off_fix=""):
#     if off_fix != "":
#         off_fix = "_"+off_fix
#     if model_type == "sklearn":
#         joblib.dump(model, f'checkpoints/{model_name}.pkl')
#         for i in range(4):
#             os.makedirs(rf"C:\Users\Administrator\Desktop\code\Ni\predictions\{model_name}",exist_ok=True)
#             pd.DataFrame(model.predict(eval(f"test_sets{off_fix}")[i])).to_csv(rf"C:\Users\Administrator\Desktop\code\Ni\predictions\{model_name}\{i+1}.csv",index=None)
#     elif model_type =="pytorch":
#         torch.save(model.state_dict(), f'checkpoints/{model_name}.pth')
#         for i in range(4):
#             os.makedirs(rf"C:\Users\Administrator\Desktop\code\Ni\predictions\{model_name}",exist_ok=True)
#             ni_dataset = NiDataset(eval(f"test_sets{off_fix}")[i],eval(f"test_set_properties")[i],eval(f"mean{off_fix}"),eval(f"std{off_fix}"))
#             tmp=[]
#             for j in range(len(ni_dataset)):
#                 item = ni_dataset[j]
#                 tmp.append(item[0])
#             tmp = torch.stack(tmp)
#             tmp = tmp.to("cuda")
#             pd.DataFrame((model(tmp)* STD + MEAN).detach().cpu().numpy()).to_csv(rf"C:\Users\Administrator\Desktop\code\Ni\predictions\{model_name}\{i+1}.csv",index=None)
#     elif model_type == "xgboost":
#         model.save_model(f'checkpoints/{model_name}.model')
#         for i in range(4):
#             os.makedirs(rf"C:\Users\Administrator\Desktop\code\Ni\predictions\{model_name}",exist_ok=True)
#             pd.DataFrame(model.predict(eval(f"test_sets{off_fix}")[i])).to_csv(rf"C:\Users\Administrator\Desktop\code\Ni\predictions\{model_name}\{i+1}.csv",index=None)
#
#
# def get_performance(model_name,off_fix=""):
#
#     performance = []
#     predictions = []
#     for i in range(4):
#         prediction = pd.read_csv(rf"C:\Users\Administrator\Desktop\code\Ni\predictions\{model_name}\{i+1}.csv")
#         predictions.append(prediction)
#         df1 = prediction
#         df2 = test_set_properties[i]
#         performance.append(
#             {"rmse" : np.sqrt(mean_squared_error(df1, df2)),
#             "mae" : mean_absolute_error(df1, df2),
#             "r2" : r2_score(df1, df2)}
#         )
#     df1 = pd.concat([predictions[i] for i in range(4)])
#     df2 = test_set_property
#     performance.append(
#             {"rmse" : np.sqrt(mean_squared_error(df1, df2)),
#             "mae" : mean_absolute_error(df1, df2),
#             "r2" : r2_score(df1, df2)}
#         )
#     return performance