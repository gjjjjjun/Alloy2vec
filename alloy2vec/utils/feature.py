# feature19.py

import numpy as np
import pandas as pd

p=pd.read_excel('../assets/property.xlsx').T
p.columns=p.iloc[1,:]
p=p.iloc[2:,:]
properity= pd.DataFrame(columns=['Ni', 'Al', 'Co', 'Cr', 'Mo', 'Nb', 'Ta', 'Ti', 'W'])
for i in properity.columns:
    properity[i]=p[i]
properity = properity.T

H=pd.read_excel('../assets/Hmix.xlsx')
H=H.set_index('ele')

def feature19(point):
    mask_index = point != 0  # 取出含量不为0的元素的掩码
    C = point[mask_index] / 100  # 元素对应的含量
    en, radius, vec, mp, sm, ym, wf, ce, p_ratio = properity[mask_index].values.T  # 从property.xlsx中索引出每个元素对应的值
    r_min = radius.min()
    r_max = radius.max()
    r_mean = radius.mean()
    en_mean = en.mean()
    C_outer = np.outer(C, C)  # 对应浓度乘积
    mask_2d = np.outer(mask_index, mask_index)  # 求掩码的外积，用来索引enthalpy
    np.fill_diagonal(mask_2d, False)  # 对角线补False
    sm_mean = sm.mean()
    p_ratio_mean = p_ratio.mean()
    ym_mean = ym.mean()

    # 计算各种物理量
    difference_of_atomic_radii = (C * (1 - radius / r_mean) ** 2).sum() ** 0.5  # 原子半径差
    difference_of_electronegativity = (C * (en_mean - en) ** 2).sum() ** 0.5  # 电负性差
    valance_electron_concentration = (C * vec).sum()  # 价电子浓度
    mixing_enthalpy = (H.where(mask_2d).dropna(how='all').dropna(axis=1, how='all') * C_outer).sum().sum() * 2  # 混合熵
    configuration_entropy = sum(-8.314 * C * np.log(C))  # 构型熵
    parameter_omiga = mp.mean() * configuration_entropy / abs(mixing_enthalpy)
    parameter_lambda = configuration_entropy / difference_of_atomic_radii ** 2
    parameter_gamma = (1 - (((r_mean + r_min) ** 2 - r_mean ** 2) ** 0.5 / (r_mean + r_min) ** 2)) / (
            1 - (((r_mean + r_max) ** 2 - r_mean ** 2) ** 0.5 / (r_mean + r_max) ** 2))
    mismatch_of_local_electronegativity = (C_outer * pd.DataFrame(abs(en[:, None] - en))).sum().sum()  # 局部不匹配电负性
    cohesive_energy = (C * ce).sum()  # 内聚能
    modulus_mismatch_in_strengthening_model = (C * ((2 * (sm - sm_mean)) / (sm + sm_mean)) / (
            1 + 0.5 * abs(C * ((2 * (sm - sm_mean)) / (sm + sm_mean))))).sum()  # 强化模型中的不匹配模量
    local_size_mismatch = (C_outer * abs(radius[:, None] - radius)).sum()  # 不匹配规模
    energy_term_in_strengthening_model = sm_mean * difference_of_atomic_radii * (1 + p_ratio_mean) / (
            1 - p_ratio_mean)  # 强化模型中的能量限制
    peierls_nabarro_factor_pai = (2 * sm_mean) / (1 - p_ratio_mean)
    six_square_of_work_function = (C * wf).sum() ** 6
    share_model = (C * sm).sum()
    difference_of_shear_model = (C * (1 - (sm / sm_mean)) ** 2).sum() ** 0.5
    local_model_mismatc = (C_outer * (abs(sm[:, None] - sm))).sum()
    lattice_distortion_energy = 0.5 * ym_mean * difference_of_atomic_radii

    # 返回计算的各项特征
    return pd.Series([
        difference_of_atomic_radii,
        difference_of_electronegativity,
        valance_electron_concentration,
        mixing_enthalpy,
        configuration_entropy,
        parameter_omiga,
        parameter_lambda,
        parameter_gamma,
        mismatch_of_local_electronegativity,
        cohesive_energy,
        modulus_mismatch_in_strengthening_model,
        local_size_mismatch,
        energy_term_in_strengthening_model,
        peierls_nabarro_factor_pai,
        six_square_of_work_function,
        share_model,
        difference_of_shear_model,
        local_model_mismatc,
        lattice_distortion_energy
    ])

