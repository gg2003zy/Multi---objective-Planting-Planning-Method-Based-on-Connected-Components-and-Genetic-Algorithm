import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import multiprocessing

import rasterio
import numpy as np

from scipy.ndimage import label
from functools import partial
from scipy.stats import mode

def find_connected_components(binary_data, j):
    # Create a binary mask for the current crop type
    mask = (binary_data == j)

    # Label connected components
    labeled_array, num_features = label(mask)

    # Compute the sizes of connected components
    sizes = np.bincount(labeled_array.ravel())

    # Exclude the background (assuming background is labeled as 0)
    sizes = sizes[1:]

    sorted_sizes = np.sort(sizes)[::-1]

    return np.sum(np.exp(sizes / 1000)), np.sum(sorted_sizes[:5])



# 计算每亩的收益
# # 2023
# profits_per_mu = np.array([-33.16, -356.59, -122.39])  # 黑龙江每亩的利润，其中水稻使用的梗稻
# # 2022
# profits_per_mu = np.array([-75.65, -171.21, 82])  # 黑龙江每亩的利润，其中水稻使用的梗稻
# # 2019
# profits_per_mu = np.array([-109.21, -263.55, -143.23])  # 黑龙江每亩的利润，其中水稻使用的梗稻
# 2020
profits_per_mu = np.array([-66.27, -137.92, 41.17])  # 黑龙江每亩的利润，其中水稻使用的梗稻
# # 2021
# profits_per_mu = np.array([-57.60, -34.96, 109.01])  # 黑龙江每亩的利润，其中水稻使用的梗稻
num_plots = 95435
bl = 6731.6/25790
WEIGHT_ALPHA = 10

# 例：假设您知道5个目标在算法初始阶段的大致上下界
obj_mins = np.array([min(profits_per_mu)*num_plots,  0,  0.0e0, 0.0e0,  0,0])
obj_maxs = np.array([num_plots*(max(profits_per_mu)*(1-bl)+min(profits_per_mu)*bl), num_plots, 1.1**(num_plots/1000),1.0e0,num_plots,num_plots])

def objective_transform(objs):
    """将原始 objs=(o1, o2, o3, o4, o5) 线性映射到 [0,1]^5"""
    normed = (objs - obj_mins) / (obj_maxs - obj_mins)
    # 注意：要确保分母非零，并保证负数、极大值不会导致爆炸
    # 若第三目标 1e50量级差异太大，可考虑对数缩放
    return normed

def evalCrops(individual, initial_planting_scheme, coords, PROFITS, AREAS, CROPS, SOYBEAN_THRESHOLD, WEIGHT_ALPHA,yes=False):
    # Compute crop indices
    crop_indices = individual - 1  # Assuming crop numbers start from 1

    # Get corresponding profits and compute total profit
    profits = PROFITS[np.arange(len(individual)), crop_indices]
    total_profit = np.dot(profits, AREAS)

    # Identify soybean plots
    is_soybean = (CROPS[crop_indices] == 'soybean')
    all_soybean_area = np.dot(AREAS, is_soybean)
    # soybean_area = np.dot(AREAS, is_soybean)
    # & (initial_planting_scheme != 2)

    # Condition masks for rotation conflicts
    mask_1 = (initial_planting_scheme == 2)
    mask_2 = (initial_planting_scheme == 1)
    mask_3 = (initial_planting_scheme == 3)

    rotation = np.sum((individual[mask_1] == 1)) - np.sum((individual[mask_1] == 2))
    rotation += np.sum((individual[mask_2] != 1)) + np.sum((individual[mask_3] == 2))

    # 添加旱田水田转换的惩罚
    min_tr = (np.sum((individual[mask_2] == 2))+np.sum((individual[mask_2] == 3)))+ (np.sum((individual[mask_1] == 1)) + np.sum((individual[mask_3] == 1)))
    # total_profit -= 300*(np.sum((individual[mask_2] == 2))+np.sum((individual[mask_2] == 3)))
    # total_profit -= 300 * (np.sum((individual[mask_1] == 1)) + np.sum((individual[mask_3] == 1)))

    # Precompute binary_data once
    binary_data_shape = (coords[:, 0].max() + 1, coords[:, 1].max() + 1)
    binary_data = np.zeros(binary_data_shape, dtype=np.uint8)
    binary_data[coords[:, 0], coords[:, 1]] = individual

    all_1 = np.count_nonzero(individual == 1)
    all_2 = np.count_nonzero(individual == 2)
    all_3 = np.count_nonzero(individual == 3)

    component_score1, k_1 = find_connected_components(binary_data, 1)
    component_score2, k_2 = find_connected_components(binary_data, 2)
    component_score3, k_3 = find_connected_components(binary_data, 3)

    bj = []

    if all_1 != 0:
        bj.append(k_1 / all_1)
    if all_2 != 0:
        bj.append(k_2 / all_2)
    if all_3 != 0:
        bj.append(k_3 / all_3)


    # Objective functions
    obj1 = float(total_profit)
    obj2 = float(all_soybean_area)
    obj3 = float(component_score1+component_score2+component_score3)
    obj4 = float(np.mean(bj))
    obj5 = float(rotation)
    obj6 = float(min_tr)

    # # Add constraint penalty for soybean planting area
    # if all_soybean_area < SOYBEAN_THRESHOLD:
    #     # obj1 = -WEIGHT_ALPHA*(SOYBEAN_THRESHOLD-all_soybean_area)
    #     obj3 = -WEIGHT_ALPHA * (SOYBEAN_THRESHOLD - all_soybean_area)
    #     obj5 = -WEIGHT_ALPHA * (SOYBEAN_THRESHOLD - all_soybean_area)
    #     obj2 = obj2-WEIGHT_ALPHA*(SOYBEAN_THRESHOLD-all_soybean_area)

    raw_objs = (obj1, obj2, obj3, obj4, obj5,obj6)
    # 这里先计算原始目标

    # 再做transform，使之落入[0,1]^5
    transformed_objs = objective_transform(np.array(raw_objs))

    obj1, obj2, obj3, obj4, obj5,obj6 = tuple(transformed_objs)

    if yes:
        return obj1,obj2,obj3*1000,obj4,obj5,obj6

    # Add constraint penalty for soybean planting area
    if all_soybean_area < SOYBEAN_THRESHOLD or min_tr > num_plots*0.5:
        obj1 = 0
        obj4 = 0
        obj5 = 0
        obj2 = 0
        obj6 = 1

    # # Add constraint penalty for soybean planting area
    # if total_profit < 0:
    #     obj3 *= 0.8
    #     obj5 *= 0.8
    #     obj2 *= 0.8

    return obj5*7/27+obj1*5/27+obj2*4/27+obj4*4/27+(1-obj6)*6/27,
    # return obj1,obj2,obj4,obj5,obj6


# 可视化函数
def plot_planting_scheme(individual,coords,title="Planting Scheme"):
    binary_data_shape = (coords[:, 0].max() + 1, coords[:, 1].max() + 1)
    binary_data = np.zeros(binary_data_shape, dtype=int)

    # 使用 NumPy 索引快速填充 binary_data
    binary_data[coords[:, 0], coords[:, 1]] = individual

    plt.figure(figsize=(10, 10))
    plt.imshow(binary_data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Crop Type')
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

def softmax(weights):
    """
    Apply Softmax to convert weights to probabilities.

    Parameters:
    - weights: Array of weights.

    Returns:
    - probabilities: Array of probabilities corresponding to the weights.
    """
    exp_weights = np.exp(weights - np.max(weights))  # Stabilize to prevent overflow
    return exp_weights / exp_weights.sum()

def custom_mutation(individual, li, mj, zw, yu, initial_planting_scheme,coords, indpb=0.1):
    """
    自定义变异函数：在轮作冲突地块上增加变异概率。

    Parameters:
    - individual: 当前个体，表示种植方案（一维数组）。
    - initial_planting_scheme: 去年种植方案（一维数组），用于判断轮作冲突。
    - indpb: 基础变异概率。
    - high_mutation_factor: 冲突地块变异概率的放大倍数。
    """

    if np.random.rand()<=0.1:

        mutation_probs = np.full(len(individual), indpb)

        # 随机确定哪些地块发生变异
        mutation_flags = np.random.rand(len(individual)) < mutation_probs

        # 如果没有地块需要变异，直接返回
        if not np.any(mutation_flags):
            return individual,

        # 4.2 计算潜在作物的轮作收益矩阵（逐个作物）
        potential_crops = np.array([1, 2, 3])  # 作物编号（假设1：水稻，2：大豆，3：玉米）
        current_crops = individual[:, np.newaxis]  # 当前种植作物（列向量）
        prev_crops = initial_planting_scheme[:, np.newaxis]  # 去年种植作物（列向量）

        # 计算轮作收益矩阵
        rotation_benefits = np.zeros((len(individual), len(potential_crops)))
        rotation_benefits[:, 0] += (prev_crops.flatten() == 2) * 1  # 大豆后水稻
        rotation_benefits[:, 1] += (prev_crops.flatten() == 1) * 1  # 水稻后大豆
        rotation_benefits[:, 1] -= (prev_crops.flatten() == 2) * 1  # 大豆后大豆
        rotation_benefits[:, 2] += (prev_crops.flatten() == 3) * 1  # 玉米后大豆

        # 4.3 根据轮作规则和当前作物计算变异的概率
        valid_mutation_mask = (potential_crops[np.newaxis, :] != current_crops)
        adjusted_probs = np.where(valid_mutation_mask, rotation_benefits, 0)

        # Softmax 计算变异概率
        exp_probs = np.exp(adjusted_probs - np.max(adjusted_probs, axis=1, keepdims=True))  # 稳定数值
        normalized_probs = exp_probs / exp_probs.sum(axis=1, keepdims=True)

        # 4.4 根据调整后的概率调整变异概率
        final_probs = mutation_probs * normalized_probs.max(axis=1)

        # 随机确定哪些地块发生变异
        mutation_flags = np.random.rand(len(individual)) < final_probs

        mutated_indices = np.where(mutation_flags)[0]

        if not np.any(mutation_flags):  # 如果没有地块需要变异，直接返回
            return individual,

        # 4.6 为发生变异的地块随机选择新的作物
        mutated_individual = individual.copy()
        for idx in mutated_indices:
            new_crop = np.random.choice(potential_crops, p=normalized_probs[idx])
            mutated_individual[idx] = new_crop

        # 清除适应度，触发重新计算
        mutated_individual = creator.Individual(mutated_individual)  # 确保是 `creator.Individual` 类型
        del mutated_individual.fitness.values  # 清除适应度，触发重新计算

        return mutated_individual,

    else:
        # Precompute binary_data once
        binary_data_shape = (coords[:, 0].max() + 1, coords[:, 1].max() + 1)
        binary_data_1 = np.zeros(binary_data_shape, dtype=np.uint8)

        if np.random.rand() <= 0.5:

            # 计算作物索引
            crop_indices = individual - 1  # 假设作物编号从 1 开始

            # 获取对应的利润并计算总收益
            profits = li[np.arange(len(individual)), crop_indices]
            total_profit = np.dot(profits, mj)
            total_profit = (total_profit - obj_mins[0]) / (obj_maxs[0] - obj_mins[0])

            # 计算大豆种植面积
            is_soybean = (zw[crop_indices] == 'soybean')
            all_soybean_area = np.dot(mj, is_soybean)

            binary_data_1[coords[:, 0], coords[:, 1]] = individual

            labeled_1, num_features_1 = label(binary_data_1 == 1)
            labeled_2, num_features_2 = label(binary_data_1 == 2)
            labeled_3, num_features_3 = label(binary_data_1 == 3)

            # 合并标签数组
            merged = np.zeros_like(binary_data_1, dtype=np.int32)

            merged[labeled_1 != 0] = labeled_1[labeled_1 != 0]  # 填充值为1的区域的标签
            merged[labeled_2 != 0] = labeled_2[labeled_2 != 0] + num_features_1  # 填充值为2的区域的标签，添加偏移
            merged[labeled_3 != 0] = labeled_3[labeled_3 != 0] + num_features_1 + num_features_2  # 填充值为3的区域的标签，添加偏移

            # 2. 获取每个区域的索引
            unique_regions = np.unique(merged)
            unique_regions = unique_regions[unique_regions > 0]  # 排除背景（0）

            indpb = indpb * 2

            # if total_profit <0.5 and all_soybean_area >=yu:
            #     p = [0.25, 0.25, 0.5]
            #     indpb = indpb * 2
            if all_soybean_area <yu:
                p = [0.25, 0.5, 0.25]
                indpb = indpb * 2
            # elif total_profit <0.5 and all_soybean_area <yu:
            #     p = [0.2, 0.4, 0.4]
            #     indpb = indpb * 2
            else:
                p = [1/3,1/3,1/3]

            # 3. 保证大豆种植面积并为每个连通区域分配作物
            new_individual = np.copy(binary_data_1)  # 从去年的种植方案初始化

            swap_regions = np.random.choice(unique_regions, size=int(len(unique_regions) * indpb), replace=False)

            for region in swap_regions:
                # 对于每个区域，选择一个随机作物进行种植
                chosen_crop = np.random.choice(range(1,4), p=p)
                new_individual[merged == region] = chosen_crop  # 将区域内的作物更换为选定的作物
        else:

            binary_data_1[coords[:, 0], coords[:, 1]] = initial_planting_scheme

            labeled_1, num_features_1 = label(binary_data_1 == 1)
            labeled_2, num_features_2 = label(binary_data_1 == 2)
            labeled_3, num_features_3 = label(binary_data_1 == 3)

            # 合并标签数组
            merged = np.zeros_like(binary_data_1, dtype=np.int32)

            merged[labeled_1 != 0] = labeled_1[labeled_1 != 0]  # 填充值为1的区域的标签
            merged[labeled_2 != 0] = labeled_2[labeled_2 != 0] + num_features_1  # 填充值为2的区域的标签，添加偏移
            merged[labeled_3 != 0] = labeled_3[labeled_3 != 0] + num_features_1 + num_features_2  # 填充值为3的区域的标签，添加偏移

            # 2. 获取每个区域的索引
            unique_regions = np.unique(merged)
            unique_regions = unique_regions[unique_regions > 0]  # 排除背景（0）

            indpb = indpb * 2

            # 3. 保证大豆种植面积并为每个连通区域分配作物
            new_individual = np.copy(binary_data_1)  # 从去年的种植方案初始化

            swap_regions = np.random.choice(unique_regions, size=int(len(unique_regions) * indpb), replace=False)

            for region in swap_regions:
                # 对于每个区域，基于轮作进行种植
                now = new_individual[merged == region][0]
                if now == 1:
                    new_crop = np.random.choice(range(1,4), p=[0.2,0.4,0.4])
                elif now == 2:
                    new_crop = np.random.choice(range(1,4), p=[0.6,0.1,0.3])
                else:
                    new_crop = np.random.choice(range(1,4), p=[0.2,0.6,0.2])
                new_individual[merged == region] = new_crop  # 将区域内的作物更换为选定的作物


        # 获取binary_data中所有大于0的元素，并保持其原始值（1, 2, 3）
        new_individual = new_individual[new_individual > 0].flatten()

        # 清除适应度，触发重新计算
        mutated_individual = creator.Individual(new_individual)  # 确保是 `creator.Individual` 类型
        del mutated_individual.fitness.values  # 清除适应度，触发重新计算

        return mutated_individual,



def geography_based_crossover(ind1, ind2, coords, initial_planting_scheme, min_region_size=20, max_region_size=100):

    # Precompute binary_data once
    binary_data_shape = (coords[:, 0].max() + 1, coords[:, 1].max() + 1)
    binary_data_1 = np.zeros(binary_data_shape, dtype=np.uint8)
    binary_data_1[coords[:, 0], coords[:, 1]] = ind1
    binary_data_2 = np.zeros(binary_data_shape, dtype=np.uint8)
    binary_data_2[coords[:, 0], coords[:, 1]] = ind2

    labeled_1, num_features_1 = label(binary_data_1 == 1)
    labeled_2, num_features_2 = label(binary_data_1 == 2)
    labeled_3, num_features_3 = label(binary_data_1 == 3)

    # 合并标签数组
    merged = np.zeros_like(binary_data_1, dtype=np.int32)

    merged[labeled_1 != 0] = labeled_1[labeled_1 != 0]  # 填充值为1的区域的标签
    merged[labeled_2 != 0] = labeled_2[labeled_2 != 0] + num_features_1  # 填充值为2的区域的标签，添加偏移
    merged[labeled_3 != 0] = labeled_3[labeled_3 != 0] + num_features_1 + num_features_2  # 填充值为3的区域的标签，添加偏移

    # 2. 获取每个区域的索引
    unique_regions = np.unique(merged)
    unique_regions = unique_regions[unique_regions > 0]  # 排除背景（0）

    p = 0.5*np.random.rand()

    swap_regions = np.random.choice(unique_regions, size=int(len(unique_regions)*p),replace=False)

    for region in swap_regions:
        # Get the coordinates of the current region
        region_mask = (merged == region)
        region_coords = np.column_stack(np.where(region_mask))

        # Swap the values for the current region in both offspring
        for coord in region_coords:
            x, y = coord
            # Swap the values in offspring1 and offspring2
            binary_data_1[x, y], binary_data_2[x, y] = binary_data_2[x, y], binary_data_1[x, y]

    offspring1 = binary_data_1[binary_data_1 > 0].flatten()
    offspring2 = binary_data_2[binary_data_2 > 0].flatten()

    # Create offspring copies
    offspring1 = creator.Individual(offspring1)
    offspring2 = creator.Individual(offspring2)

    # Clear fitness values to ensure they will be re-evaluated
    del offspring1.fitness.values
    del offspring2.fitness.values

    return offspring1, offspring2




def downsample_matrix(matrix, block_size):
    """
    Downsample a matrix by aggregating values within rectangular blocks.

    Parameters:
    - matrix (np.ndarray): The input matrix to be downsampled.
    - block_size (tuple): A tuple of (height, width) specifying the block size.

    Returns:
    - np.ndarray: The downsampled matrix.
    """
    block_h, block_w = block_size
    h, w = matrix.shape

    # Determine the size of the output matrix
    new_h = h // block_h
    new_w = w // block_w

    # Initialize the output matrix
    downsampled = np.zeros((new_h, new_w), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            # Extract the block from the original matrix
            block = matrix[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]

            # Calculate the majority value (mode) in the block
            unique, counts = np.unique(block, return_counts=True)
            block_mode = unique[np.argmax(counts)]
            downsampled[i, j] = block_mode

    return downsampled

def main():
    random.seed(42)  # 设置随机种子以保证实验的可重复性

    # 读取2023年TIF文件
    with rasterio.open('fujin2019.tif') as src:
        data = src.read(1)

    # 提取binary_data（值为1, 2, 3 的区域），保留原本的1、2、3
    initial_planting_matrix = np.where(np.isin(data, [1, 2, 3]), data, 0).astype(np.uint8)
    # (31, 31)
    initial_planting_matrix_1 = downsample_matrix(initial_planting_matrix, (31, 31))

    # 获取binary_data中所有大于0的元素，并保持其原始值（1, 2, 3）
    initial_planting_scheme = initial_planting_matrix_1[initial_planting_matrix_1 > 0].flatten()

    # # 读取保存的 txt 文件
    # initial_planting_scheme = np.loadtxt('2022.txt', delimiter=',')

    # 读取保存的 txt 文件
    initial_planting_scheme_2 = np.loadtxt('2020.txt', delimiter=',')
    initial_planting_scheme_2 = np.uint8(initial_planting_scheme_2)

    initial_planting_scheme = np.uint8(initial_planting_scheme)

    # 输出一维数组
    print("2023年 binary_data 中大于0的元素一维数组：", initial_planting_scheme)

    # 获取binary_data中所有大于0元素的坐标
    coords = np.column_stack(np.where(initial_planting_matrix_1 > 0))

    binary_data_shape = (coords[:, 0].max() + 1, coords[:, 1].max() + 1)
    initial_planting_matrix = np.zeros(binary_data_shape, dtype=np.uint8)
    initial_planting_matrix[coords[:, 0], coords[:, 1]] = initial_planting_scheme

    num_plots = len(initial_planting_scheme)
    print(num_plots)

    rows, cols = initial_planting_matrix.shape

    AREAS = [1] * num_plots  # 每块地的面积

    # CROPS = ['soybean', 'wheat', 'corn']  # 作物种类
    CROPS = np.array(['wheat', 'soybean', 'corn'])  # 作物种类

    # # 定义目标权重
    # WEIGHT_ALPHA = 10   # 大豆种植不足的惩罚权重
    # bl = 0.2610
    # # 6731.6/25790
    SOYBEAN_THRESHOLD = sum(AREAS) * bl  # 大豆种植面积阈值
    # 548.1 / 987.708

    # # 计算每亩的收益
    # profits_per_mu = np.array([-75.65, -171.21, 82])  # 黑龙江每亩的利润，其中水稻使用的梗稻

    # 创建 PROFIT 数组，将每种作物的收益数据复制到每个地块
    PROFITS = np.tile(profits_per_mu, (num_plots, 1))

    # 输出结果
    print("每种作物的每亩收益（单位：元）：")
    print(profits_per_mu)

    # 目标函数和遗传算法配置
    # 更新工具箱中的交叉和变异策略
    # creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0,-1))
    creator.create("FitnessSingle", base.Fitness, weights=(1.0,))

    # 定义 Individual 为 NumPy 数组
    creator.create("Individual", np.ndarray, fitness=creator.FitnessSingle)
    # # 定义 Individual 为 NumPy 数组
    # creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

    # 创建个体时使用 NumPy 数组
    def create_individual():

        # 3. 为每个连通区域随机分配作物
        new_individual = np.copy(initial_planting_matrix_1)  # 从去年的种植方案初始化

        labeled_1, num_features_1 = label(initial_planting_matrix_1 == 1)
        labeled_2, num_features_2 = label(initial_planting_matrix_1 == 2)
        labeled_3, num_features_3 = label(initial_planting_matrix_1 == 3)

        # 合并标签数组
        merged = np.zeros_like(initial_planting_matrix_1, dtype=np.int32)

        merged[labeled_1 != 0] = labeled_1[labeled_1 != 0]  # 填充值为1的区域的标签
        merged[labeled_2 != 0] = labeled_2[labeled_2 != 0] + num_features_1  # 填充值为2的区域的标签，添加偏移
        merged[labeled_3 != 0] = labeled_3[labeled_3 != 0] + num_features_1 + num_features_2  # 填充值为3的区域的标签，添加偏移

        # 2. 获取每个区域的索引
        unique_regions = np.unique(merged)
        unique_regions = unique_regions[unique_regions > 0]  # 排除背景（0）

        for region in unique_regions:
            # 对于每个区域，选择一个随机作物进行种植
            chosen_crop = np.random.randint(1, len(CROPS) + 1)  # 随机选择作物 1、2、3
            new_individual[merged == region] = chosen_crop  # 将区域内的作物更换为选定的作物

        # 获取binary_data中所有大于0的元素，并保持其原始值（1, 2, 3）
        new_individual = new_individual[new_individual > 0].flatten()

        # 4. 将生成的种植方案转为 Individual 类型
        return creator.Individual(np.uint8(new_individual))

    # 定义工具箱
    toolbox = base.Toolbox()

    # import itertools
    #
    # def uniform_reference_points(nobj, divisions=4):
    #     """Das and Dennis方法生成参考点"""
    #     ref_dirs = []
    #     for comb in itertools.combinations(range(divisions + nobj - 1), nobj - 1):
    #         arr = np.array(comb)
    #         arr = np.concatenate([arr, [divisions + nobj - 1]])
    #         arr = arr[1:] - arr[:-1] - 1
    #         arr = arr / divisions
    #         ref_dirs.append(arr)
    #     return np.array(ref_dirs)
    #
    # ref_points = uniform_reference_points(nobj=5, divisions=12)
    #  ref_points=ref_points
    # 注册选择操作，并传入参考点
    toolbox.register("select", tools.selBest)
    toolbox.register("attr_int", random.randint, 1, len(CROPS))
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", partial(geography_based_crossover, coords=coords, initial_planting_scheme=initial_planting_scheme, min_region_size=1, max_region_size=310))
    toolbox.register("mutate", partial(custom_mutation,li=PROFITS,zw=CROPS,mj=AREAS,yu=SOYBEAN_THRESHOLD,
                     initial_planting_scheme=initial_planting_scheme,
                     indpb=0.02,  # 基础变异概率
                     coords=coords,))

    # toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", partial(evalCrops, initial_planting_scheme=initial_planting_scheme, coords=coords,PROFITS=PROFITS,AREAS=AREAS,CROPS=CROPS,SOYBEAN_THRESHOLD=SOYBEAN_THRESHOLD,WEIGHT_ALPHA = WEIGHT_ALPHA))

    # 会影响进化的速度
    k = 7

    def create_smart_initial(initial_scheme, soybean_threshold):
        individual = initial_scheme.copy()
        soybean_plots = np.sum(individual == 2)

        # 确保大豆种植达标
        while soybean_plots < soybean_threshold:
            idx = random.randint(0, len(individual) - 1)
            if individual[idx] != 2:
                individual[idx] = 2
                soybean_plots += 1

        return creator.Individual(individual)


    # 创建初始种群
    population_size = 250
    l = int(population_size/100)
    pop = toolbox.population(n=1)  # 少生成一个随机个体
    initial_individual = creator.Individual(initial_planting_scheme)  # 将 initial_planting_scheme 转为 Individual 类型

    def create_max_lr_based_on_connectivity(q):

        labeled_1, num_features_1 = label(initial_planting_matrix_1 == 1)
        labeled_2, num_features_2 = label(initial_planting_matrix_1 == 2)
        labeled_3, num_features_3 = label(initial_planting_matrix_1 == 3)

        # 合并标签数组
        merged = np.zeros_like(initial_planting_matrix_1, dtype=np.int32)

        merged[labeled_1 != 0] = labeled_1[labeled_1 != 0]  # 填充值为1的区域的标签
        merged[labeled_2 != 0] = labeled_2[labeled_2 != 0] + num_features_1  # 填充值为2的区域的标签，添加偏移
        merged[labeled_3 != 0] = labeled_3[labeled_3 != 0] + num_features_1 + num_features_2  # 填充值为3的区域的标签，添加偏移

        # 2. 获取每个区域的索引
        unique_regions = np.unique(merged)
        unique_regions = unique_regions[unique_regions > 0]  # 排除背景（0）

        # 3. 保证大豆种植面积并为每个连通区域分配作物
        new_individual = np.copy(initial_planting_matrix_1)  # 从去年的种植方案初始化

        soybean_area_needed = int(SOYBEAN_THRESHOLD)  # 需要大豆种植的最小面积
        soybean_count = 0

        for region in unique_regions:
            if soybean_count < soybean_area_needed:
                # 如果大豆面积不足，优先选择大豆
                chosen_crop = 2  # 选择大豆
                region_area = np.count_nonzero(merged == region)
                soybean_count += region_area
            else:
                # 大豆面积足够后，选择经济效益较高的作物（例如玉米）
                chosen_crop = q  # 假设玉米是最经济的
            # 将区域内的作物更换为选定的作物
            new_individual[merged == region] = chosen_crop

        # 获取binary_data中所有大于0的元素，并保持其原始值（1, 2, 3）
        new_individual = new_individual[new_individual > 0].flatten()

        # 4. 将生成的种植方案转为 Individual 类型
        return creator.Individual(np.uint8(new_individual))

    for o in range((l)):
        for i in range(len(CROPS)):
            all_planting = np.full(num_plots, i+1)
            all_planting = np.uint8(all_planting)
            pop.append(create_smart_initial(all_planting, SOYBEAN_THRESHOLD))
            pop.append(create_smart_initial(all_planting, SOYBEAN_THRESHOLD))

        # 添加最大收益种植方案
        max_lr = create_max_lr_based_on_connectivity(1)
        pop.append(create_smart_initial(max_lr, SOYBEAN_THRESHOLD))
        max_lr = create_max_lr_based_on_connectivity(2)
        pop.append(create_smart_initial(max_lr, SOYBEAN_THRESHOLD))
        max_lr = create_max_lr_based_on_connectivity(3)
        pop.append(create_smart_initial(max_lr, SOYBEAN_THRESHOLD))

        # 初始化新的轮作方案
        rotation_individual = initial_planting_scheme.copy()
        min_tr = initial_planting_scheme.copy()

        # 创建掩码来识别去年种植的不同作物类型
        mask_1 = (initial_planting_scheme == 2)  # 去年种植了大豆
        mask_2 = (initial_planting_scheme == 1)  # 去年种植了水稻
        mask_3 = (initial_planting_scheme == 3)  # 去年种植了玉米

        # 根据轮作规则为每个地块分配新的作物种植，尽量减少轮作冲突
        # 对于去年种植了大豆的地块，优先选择水稻
        rotation_individual[mask_1] = 1  # 大豆后种水稻
        min_tr[mask_1] = 3

        # 对于去年种植了水稻的地块，优先选择玉米
        rotation_individual[mask_2] = 3  # shuidao后种玉米
        min_tr[mask_2] = 1

        # 对于去年种植了玉米的地块，优先选择小麦
        rotation_individual[mask_3] = 2  # 玉米后种大豆
        min_tr[mask_3] = 2

        rotation_individual = np.uint8(rotation_individual)
        min_tr = np.uint8(rotation_individual)

        pop.append(create_smart_initial(rotation_individual, SOYBEAN_THRESHOLD))
        pop.append(create_smart_initial(rotation_individual, SOYBEAN_THRESHOLD))
        pop.append(create_smart_initial(rotation_individual, SOYBEAN_THRESHOLD))

        pop.append(create_smart_initial(min_tr, SOYBEAN_THRESHOLD))
        pop.append(create_smart_initial(min_tr, SOYBEAN_THRESHOLD))
        pop.append(create_smart_initial(min_tr, SOYBEAN_THRESHOLD))

        for i in range(int(k)):
            pop.append(initial_individual)  # 将 initial_individual 加入种群

    from pyDOE import lhs

    # 生成均匀分布的三种作物种植比例点
    def generate_lhs_points(num_points):
        # 生成三维的拉丁超立方体样本
        points = lhs(3, samples=num_points, criterion='center')

        # 由于拉丁超立方体采样生成的点在[0, 1]范围内, 将它们缩放到[0, 1]的和为1
        points = np.array([point / sum(point) for point in points])  # 确保 x + y + z = 1
        return points

    # 生成拉丁超立方体采样的作物比例
    proportions = generate_lhs_points(int(population_size - (k + len(CROPS)*2 + 3)*l)-1)

    for i in range(len(proportions)):

        proportion = proportions[i]

        # 3. 为每个连通区域随机分配作物
        new_individual = np.copy(initial_planting_matrix_1)  # 从去年的种植方案初始化

        labeled_1, num_features_1 = label(initial_planting_matrix_1 == 1)
        labeled_2, num_features_2 = label(initial_planting_matrix_1 == 2)
        labeled_3, num_features_3 = label(initial_planting_matrix_1 == 3)

        # 合并标签数组
        merged = np.zeros_like(initial_planting_matrix_1, dtype=np.int32)

        merged[labeled_1 != 0] = labeled_1[labeled_1 != 0]  # 填充值为1的区域的标签
        merged[labeled_2 != 0] = labeled_2[labeled_2 != 0] + num_features_1  # 填充值为2的区域的标签，添加偏移
        merged[labeled_3 != 0] = labeled_3[labeled_3 != 0] + num_features_1 + num_features_2  # 填充值为3的区域的标签，添加偏移

        # 2. 获取每个区域的索引
        unique_regions = np.unique(merged)
        unique_regions = unique_regions[unique_regions > 0]  # 排除背景（0）

        total_regions = len(unique_regions)

        # 根据给定的比例分配作物
        num_crop_1 = int(proportion[0] * total_regions)  # 分配给作物1的区域数量
        num_crop_2 = int(proportion[1] * total_regions)  # 分配给作物2的区域数量
        num_crop_3 = total_regions - num_crop_1 - num_crop_2  # 剩余分配给作物3

        # 随机分配连通区域给作物1、作物2、作物3
        random.shuffle(unique_regions)

        crop_assignment = {
            1: unique_regions[:num_crop_1],
            2: unique_regions[num_crop_1:num_crop_1 + num_crop_2],
            3: unique_regions[num_crop_1 + num_crop_2:]
        }

        # 将分配好的作物区域赋值给新个体
        for crop, regions in crop_assignment.items():
            for region in regions:
                new_individual[merged == region] = crop

        # 获取binary_data中所有大于0的元素，并保持其原始值（1, 2, 3）
        new_individual = new_individual[new_individual > 0].flatten()

        pop.append(create_smart_initial(new_individual, SOYBEAN_THRESHOLD))

        # pop.append(creator.Individual(np.uint8(new_individual)))

    # 精英保留策略
    hof = tools.HallOfFame(1)

    # 交叉概率
    crossover_probability = 0.7

    # 变异概率
    mutation_probability = 0.2

    # 进化代数
    number_of_generations = 300

    # 收集统计数据的配置
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    toolbox.stats = stats

    m = 5

    pop, log = algorithms.eaSimple(pop, toolbox, crossover_probability, mutation_probability,
                                   ngen=number_of_generations, m=m ,stats=stats, halloffame=hof,
                                   verbose=True)

    # # 使用混合优化算法
    # pop, log, hof = hybrid_ga_with_sa(pop, toolbox, ngen=number_of_generations, m = m, cxpb=crossover_probability, mutpb=mutation_probability, local_search_rate=0.3)


    # 可视化初始种植配置和最终种植配置
    initial_individual = pop[0]
    best_individual = hof[0]
    plot_planting_scheme(initial_planting_scheme, coords, "Initial Planting Scheme")
    print("原始种植方案: ", initial_planting_scheme)
    print("收益最大化、大豆种植面积最大化、连片种植最大化、轮作效益最大化:", evalCrops(initial_planting_scheme,initial_planting_scheme, coords=coords,PROFITS=PROFITS,AREAS=AREAS,CROPS=CROPS,SOYBEAN_THRESHOLD=SOYBEAN_THRESHOLD,WEIGHT_ALPHA = WEIGHT_ALPHA,yes=True))
    print("收益最大化、大豆种植面积最大化、连片种植最大化、轮作效益最大化:", evalCrops(initial_planting_scheme,initial_planting_scheme, coords=coords,PROFITS=PROFITS,AREAS=AREAS,CROPS=CROPS,SOYBEAN_THRESHOLD=SOYBEAN_THRESHOLD,WEIGHT_ALPHA = WEIGHT_ALPHA))

    plot_planting_scheme(best_individual, coords, "Final Optimized Planting Scheme")


    print("最优轮作种植方案: ", rotation_individual)
    print("收益最大化、大豆种植面积最大化、连片种植最大化、轮作效益最大化:", evalCrops(rotation_individual, initial_planting_scheme,coords=coords,PROFITS=PROFITS,AREAS=AREAS,CROPS=CROPS,SOYBEAN_THRESHOLD=SOYBEAN_THRESHOLD,WEIGHT_ALPHA = WEIGHT_ALPHA,yes=True))
    print("收益最大化、大豆种植面积最大化、连片种植最大化、轮作效益最大化:", evalCrops(rotation_individual, initial_planting_scheme,coords=coords,PROFITS=PROFITS,AREAS=AREAS,CROPS=CROPS,SOYBEAN_THRESHOLD=SOYBEAN_THRESHOLD,WEIGHT_ALPHA = WEIGHT_ALPHA))

    print("最优种植方案: ", best_individual)
    # 将数组保存为txt文件
    np.savetxt('2020.txt', best_individual, delimiter=',', fmt='%d')
    print("收益最大化、大豆种植面积最大化、连片种植最大化、轮作效益最大化:", evalCrops(best_individual, initial_planting_scheme,coords=coords,PROFITS=PROFITS,AREAS=AREAS,CROPS=CROPS,SOYBEAN_THRESHOLD=SOYBEAN_THRESHOLD,WEIGHT_ALPHA = WEIGHT_ALPHA,yes=True))
    print("收益最大化、大豆种植面积最大化、连片种植最大化、轮作效益最大化:", evalCrops(best_individual, initial_planting_scheme,coords=coords,PROFITS=PROFITS,AREAS=AREAS,CROPS=CROPS,SOYBEAN_THRESHOLD=SOYBEAN_THRESHOLD,WEIGHT_ALPHA = WEIGHT_ALPHA))

    # import pickle  # 在代码开头添加该导入语句
    #
    # # 原代码结尾处，在可视化和打印语句之后添加以下保存逻辑
    #
    # # 保存 log（统计信息）
    # with open('log.pkl', 'wb') as f:
    #     pickle.dump(log, f)
    # print("日志（log）已保存至 log.pkl")
    #
    # # 保存种群（pop）
    # with open('pop.pkl', 'wb') as f:
    #     pickle.dump(pop, f)
    # print("种群（pop）已保存至 pop.pkl")
    #
    # # # 读取 log
    # # with open('log.pkl', 'rb') as f:
    # #     loaded_log = pickle.load(f)
    # #
    # #
    # # # 读取 pop
    # # with open('pop.pkl', 'rb') as f:
    # #     loaded_pop = pickle.load(f)
    #
    #
    #
    #
    # # ======================
    # # 新增：种群稳定性分析
    # # ======================
    # from collections import namedtuple
    #
    # # 定义参数名称（每个地块对应一个参数，此处简化为作物类型索引）
    # param_names = [f"Plot_{i + 1}" for i in range(num_plots)]
    #
    # # 提取种群数据
    # final_population = pop
    # best_individual = hof[0] if hof else pop[0]
    # fitness_scores = np.array([ind.fitness.values[0] for ind in final_population])
    # param_values = np.array([ind for ind in final_population])  # 个体的作物分配数组
    #
    # def calculate_genotype_entropy(individuals):
    #     entropy = []
    #     for col in range(individuals.shape[1]):
    #         counts = np.bincount(individuals[:, col], minlength=4)  # 作物类型1-3，索引0为无效
    #         probs = counts[1:] / len(individuals)  # 排除背景0
    #         # 修正：计算每列的熵值并添加到列表
    #         col_entropy = -np.sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
    #         entropy.append(col_entropy)
    #     return np.mean(entropy)
    #
    # # 统计分析
    # print("\n===================== 种群统计分析 =====================")
    # print(f"最优解适应度: {best_individual.fitness.values[0]:.2f}")
    # print(f"种群平均适应度: {np.mean(fitness_scores):.2f} ± {np.std(fitness_scores):.2f}")
    # print(f"适应度变异系数: {np.std(fitness_scores) / np.mean(fitness_scores):.4f}")
    #
    # # 收敛性分析（最后10代适应度波动）
    # if len(log.chapters["fitness"]) >= 5:
    #     last_10_fitness = log.chapters["fitness"].select("max")[-10:]
    #     convergence_std = np.std(last_10_fitness)
    #     print(f"最后10代适应度波动: {convergence_std:.4f}")
    #
    # # 多样性分析（基因型熵）
    # genotype_entropy = calculate_genotype_entropy(param_values)
    # print(f"种群基因型熵: {genotype_entropy:.4f} (值越低表示越收敛)")
    #
    # # 绘制适应度分布直方图
    # plt.figure(figsize=(8, 4))
    # plt.hist(fitness_scores, bins=20, density=True, alpha=0.6, color='blue')
    # plt.axvline(x=np.mean(fitness_scores), color='red', linestyle='--', label='均值')
    # plt.axvline(x=best_individual.fitness.values[0], color='green', linestyle='--', label='最优解')
    # plt.title('适应度分布')
    # plt.xlabel('适应度值'), plt.ylabel('频率'), plt.legend()
    # plt.grid(True), plt.tight_layout(), plt.show()
    #
    # # 绘制收敛曲线
    # plt.figure(figsize=(12, 4))
    # gen = log.select("gen")
    # best_fitness = log.select("max")
    # mean_fitness = log.select("avg")
    #
    # plt.subplot(1, 2, 1)
    # print(gen)
    # print(best_fitness)
    # plt.plot(gen, best_fitness, label='最优适应度', color='red')
    # plt.plot(gen, mean_fitness, label='平均适应度', color='blue')
    # plt.title('进化收敛曲线')
    # plt.xlabel('世代'), plt.ylabel('适应度'), plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(gen, np.array(best_fitness) - np.array(mean_fitness), color='purple')
    # plt.title('最优解与均值差距')
    # plt.xlabel('世代'), plt.ylabel('差值'), plt.grid(True)
    # plt.tight_layout(), plt.show()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # 如果使用了PyInstaller，确保冻结支持
    main()

