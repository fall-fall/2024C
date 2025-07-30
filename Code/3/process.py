# -*- coding: utf-8 -*-
"""
==============================================================================
2024年数学建模C题 - 问题三：基于遗传算法的仿真优化模型
==============================================================================
版本: 1.0
作者: Gemini

功能:
1.  建立一个包含相关性随机波动和市场反馈机制的仿真系统。
2.  使用遗传算法(GA)搜索最优的7年种植计划。
3.  通过大量蒙特卡洛仿真评估每个计划的适应度（平均利润）。
4.  处理复杂的硬性约束（忌重茬、豆类种植）。
"""

import pandas as pd
import numpy as np
import os
import time
import random
import copy

# ==============================================================================
# 0. 参数设置 (可调整)
# ==============================================================================
# --- 遗传算法参数 ---
POP_SIZE = 30           # 种群大小 (建议值: 50-100)
MAX_GEN = 50            # 最大遗传代数 (建议值: 100-500)
CX_PROB = 0.8           # 交叉概率
MUT_PROB = 0.2          # 变异概率
N_SIMULATIONS = 30      # 每次适应度评估的仿真次数 (建议值: 100-500)

# --- 市场仿真参数 (需在论文中作为假设说明) ---
# 供给影响价格系数 (蔬菜和食用菌)
# 解释: 供给每增加1%，价格就下降 η 次方。η 越大，影响越显著。
SUPPLY_PRICE_ELASTICITY = 0.2 

# --- 全局变量 (将在数据加载时填充) ---
PARAMS = {}
I_PLOTS, J_CROPS, Y_YEARS, K_SEASONS = [], [], [], []
P_AREA, P_PAST, J_BEAN, S_SUITABILITY = {}, {}, [], {}
CROP_TYPES = {}
BASE_PROD, BASE_PRICE, BASE_COST = {}, {}, {}
COV_MATRIX, CHOLESKY_L = None, None

# ==============================================================================
# 1. 数据加载与预处理
# ==============================================================================
def load_data_and_setup_globals(project_root_dir):
    """加载所有数据并初始化全局参数"""
    global PARAMS, I_PLOTS, J_CROPS, Y_YEARS, K_SEASONS, \
           P_AREA, P_PAST, J_BEAN, S_SUITABILITY, CROP_TYPES, \
           BASE_PROD, BASE_PRICE, BASE_COST, COV_MATRIX, CHOLESKY_L
    
    print("--- 1. 开始加载和预处理数据 ---")
    
    try:
        data_dir = os.path.join(project_root_dir, 'Data')
        plots_df = pd.read_excel(os.path.join(data_dir, '附件1.xlsx'), sheet_name='乡村的现有耕地')
        crops_info_df = pd.read_excel(os.path.join(data_dir, '附件1.xlsx'), sheet_name='乡村种植的农作物')
        stats_df = pd.read_excel(os.path.join(data_dir, '附件2.xlsx'), sheet_name='2023年统计的相关数据')
        past_planting_df = pd.read_excel(os.path.join(data_dir, '附件2.xlsx'), sheet_name='2023年的农作物种植情况')
    except Exception as e:
        print(f"错误: 无法加载数据文件。请确保路径正确。{e}")
        return

    # --- 数据清洗 ---
    plots_df.columns = plots_df.columns.str.strip()
    crops_info_df.columns = crops_info_df.columns.str.strip()
    stats_df.columns = stats_df.columns.str.strip()
    past_planting_df.columns = past_planting_df.columns.str.strip()
    stats_df.dropna(subset=['作物名称'], inplace=True)
    numeric_cols = ['亩产量/斤', '种植成本/(元/亩)', '销售单价/(元/斤)']
    for col in numeric_cols:
        def clean_and_convert(value):
            if isinstance(value, str) and '-' in value:
                try: low, high = map(float, value.split('-')); return (low + high) / 2
                except: return None
            return value
        stats_df[col] = stats_df[col].apply(clean_and_convert)
    stats_df[numeric_cols] = stats_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    stats_df['亩产量(kg/亩)'] = stats_df['亩产量/斤'] / 2
    stats_df['销售价格(元/kg)'] = stats_df['销售单价/(元/斤)'] * 2

    I_PLOTS = plots_df['地块名称'].tolist()
    J_CROPS = sorted(stats_df['作物名称'].unique().tolist())
    Y_YEARS = list(range(2024, 2031))
    K_SEASONS = [1, 2]

    crop_data = pd.merge(stats_df, crops_info_df[['作物名称', '作物类型']], on='作物名称', how='left')
    CROP_TYPES = dict(zip(crop_data['作物名称'], crop_data['作物类型']))
    
    P_AREA = dict(zip(plots_df['地块名称'], plots_df['地块面积/亩']))
    BASE_PRICE = dict(zip(crop_data['作物名称'], crop_data['销售价格(元/kg)']))
    BASE_PROD = dict(zip(crop_data['作物名称'], crop_data['亩产量(kg/亩)']))
    BASE_COST = dict(zip(crop_data['作物名称'], crop_data['种植成本/(元/亩)']))

    J_BEAN = [j for j in J_CROPS if CROP_TYPES.get(j) == '豆类' or j in ['大豆', '绿豆', '红豆', '豌豆', '蚕豆', '黄豆']]
    
    P_PAST = {i: None for i in I_PLOTS}
    past_planting_df_nonan = past_planting_df.dropna(subset=['种植地块', '作物名称'])
    for _, row in past_planting_df_nonan.iterrows():
        if row['种植地块'] in P_PAST:
            P_PAST[row['种植地块']] = row['作物名称']
    
    S_SUITABILITY = {}
    plot_types = dict(zip(plots_df['地块名称'], plots_df['地块类型']))
    for i in I_PLOTS:
        for j in J_CROPS:
            for k in K_SEASONS:
                plot_t, crop_t, is_bean = plot_types.get(i), CROP_TYPES.get(j), j in J_BEAN
                suitable = 0
                if plot_t in ['平旱地', '梯田', '山坡地'] and (crop_t == '粮食' or is_bean) and k == 1: suitable = 1
                elif plot_t == '水浇地':
                    if (crop_t == '水稻' and k==1): suitable = 1
                    if (crop_t == '蔬菜'): suitable = 1
                elif plot_t == '普通大棚' and ((crop_t == '蔬菜' and k == 1) or (crop_t == '食用菌' and k == 2)): suitable = 1
                elif plot_t == '智慧大棚' and crop_t == '蔬菜': suitable = 1
                S_SUITABILITY[i, j, k] = suitable

    # --- 建立相关性矩阵 (重要假设) ---
    # 假设：粮食之间正相关，蔬菜之间正相关
    n_vars = len(J_CROPS)
    # 主对角线为自身方差(波动性)，这里设为0.05，即价格标准差约为sqrt(0.05)=22%
    COV_MATRIX = np.eye(n_vars) * 0.05 
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            crop1, crop2 = J_CROPS[i], J_CROPS[j]
            type1, type2 = CROP_TYPES.get(crop1), CROP_TYPES.get(crop2)
            if type1 == type2 and type1 in ['粮食', '蔬菜']:
                # 同类作物价格弱正相关
                COV_MATRIX[i, j] = COV_MATRIX[j, i] = 0.02 
    
    # 保证矩阵半正定，以进行Cholesky分解
    min_eig = np.min(np.linalg.eigvalsh(COV_MATRIX))
    if min_eig < 0:
        COV_MATRIX -= 1.01 * min_eig * np.eye(n_vars)

    CHOLESKY_L = np.linalg.cholesky(COV_MATRIX)
    print("--- 1. 数据与全局参数设置完成 ---")


# ==============================================================================
# 2. 遗传算法核心函数
# ==============================================================================

# 一个“个体”的数据结构:
# solution = { year: { plot: { season: crop_name } } }

def create_initial_solution():
    """创建一个满足硬约束的随机初始解"""
    # 随机决定每个地块每年的种植计划
    solution = {y: {i: {k: None for k in K_SEASONS} for i in I_PLOTS} for y in Y_YEARS}
    for y in Y_YEARS:
        for i in I_PLOTS:
            for k in K_SEASONS:
                possible_crops = [j for j in J_CROPS if S_SUITABILITY.get((i, j, k), 0) == 1]
                if possible_crops:
                    solution[y][i][k] = random.choice(possible_crops)
    
    # 创建后必须修复，以满足所有约束
    return repair_solution(solution)

def repair_solution(solution):
    """修复解，确保其满足所有硬约束（忌重茬和豆类）"""
    # 按年份和地块顺序修复，保证修复后的结果依然满足之前修复过的约束
    for y in Y_YEARS:
        for i in I_PLOTS:
            # --- 修复忌重茬 ---
            last_year_crops = set()
            if y > Y_YEARS[0]:
                for k_prev in K_SEASONS:
                    crop = solution[y-1][i].get(k_prev)
                    if crop: last_year_crops.add(crop)
            else: # 2024年
                crop = P_PAST.get(i)
                if crop: last_year_crops.add(crop)

            for k in K_SEASONS:
                if solution[y][i][k] in last_year_crops:
                    possible_crops = [j for j in J_CROPS if S_SUITABILITY.get((i, j, k), 0) == 1 and j not in last_year_crops]
                    solution[y][i][k] = random.choice(possible_crops) if possible_crops else None

    # --- 修复豆类约束 ---
    for i in I_PLOTS:
        # 检查每个3年窗口
        for y_start in range(2023, Y_YEARS[-1] - 1):
            window_years = list(range(y_start, y_start + 3))
            
            # 检查窗口内是否已有豆类
            planted_in_window = False
            for y_win in window_years:
                if y_win == 2023:
                    if P_PAST.get(i) in J_BEAN: planted_in_window = True; break
                else:
                    if any(solution[y_win][i].get(k) in J_BEAN for k in K_SEASONS):
                        planted_in_window = True; break
            if planted_in_window: continue

            # 如果没有，则强制种植一个
            y_fix = random.choice(window_years[1:]) # 不修改2023年的历史
            k_fix = random.choice(K_SEASONS)
            
            last_year_crops = set(solution[y_fix-1][i].values()) if y_fix > Y_YEARS[0] else {P_PAST.get(i)}
            possible_beans = [j for j in J_BEAN if S_SUITABILITY.get((i, j, k_fix), 0) == 1 and j not in last_year_crops]
            
            if possible_beans:
                solution[y_fix][i][k_fix] = random.choice(possible_beans)

    return solution

def crossover(parent1, parent2):
    """单点交叉：随机选择一个地块，交换其完整的7年种植计划"""
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    plot_to_swap = random.choice(I_PLOTS)
    
    for y in Y_YEARS:
        child1[y][plot_to_swap], child2[y][plot_to_swap] = child2[y][plot_to_swap], child1[y][plot_to_swap]
        
    return child1, child2

def mutate(solution):
    """随机改变一个种植决策"""
    y_mut = random.choice(Y_YEARS)
    i_mut = random.choice(I_PLOTS)
    k_mut = random.choice(K_SEASONS)
    
    possible_crops = [j for j in J_CROPS if S_SUITABILITY.get((i_mut, j, k_mut), 0) == 1]
    if possible_crops:
        solution[y_mut][i_mut][k_mut] = random.choice(possible_crops)

    return solution

# ==============================================================================
# 3. 仿真与适应度评估
# ==============================================================================
def evaluate_fitness(solution):
    """核心函数：运行N次仿真，计算一个solution的平均总利润"""
    total_profits = []
    for _ in range(N_SIMULATIONS):
        yearly_profits = []
        for y in Y_YEARS:
            # 生成年度相关的随机价格冲击
            uncorr_shocks = np.random.randn(len(J_CROPS))
            corr_shocks = CHOLESKY_L @ uncorr_shocks
            
            # 计算当年的总供给
            total_supply = {j: 0 for j in J_CROPS}
            for i in I_PLOTS:
                for k in K_SEASONS:
                    crop = solution[y][i][k]
                    if crop:
                        # 假设产量有 +/-10% 的随机波动
                        prod_shock = 1 + (random.random() * 0.2 - 0.1)
                        total_supply[crop] += P_AREA[i] * BASE_PROD.get(crop,0) * prod_shock

            # 计算受供需影响的年度价格和成本
            current_price = {}
            current_cost = {}
            for idx, j in enumerate(J_CROPS):
                price_shock = 1 + corr_shocks[idx]
                sim_price = BASE_PRICE.get(j,0) * price_shock
                
                # 供给影响价格 (仅对蔬菜和食用菌)
                if CROP_TYPES.get(j) in ['蔬菜', '食用菌'] and total_supply.get(j,0) > 0:
                    # 计算一个基准供给量
                    base_supply = P_AREA[list(P_AREA.keys())[0]] * 10 * BASE_PROD.get(j,0) # 简单假设
                    supply_ratio = base_supply / total_supply.get(j,1)
                    sim_price *= (supply_ratio ** SUPPLY_PRICE_ELASTICITY)

                current_price[j] = sim_price
                current_cost[j] = BASE_COST.get(j,0) * (1.05 ** (y - 2023))

            # 计算年度总利润
            year_revenue = 0
            year_cost = 0
            for i in I_PLOTS:
                for k in K_SEASONS:
                    crop = solution[y][i][k]
                    if crop:
                        revenue = P_AREA[i] * BASE_PROD.get(crop,0) * current_price.get(crop,0)
                        cost = P_AREA[i] * current_cost.get(crop,0)
                        year_revenue += revenue
                        year_cost += cost
            
            yearly_profits.append(year_revenue - year_cost)
        total_profits.append(sum(yearly_profits))
        
    return np.mean(total_profits) if total_profits else 0

# ==============================================================================
# 4. 主执行流程
# ==============================================================================
def main():
    """主函数，运行遗传算法"""
    # --- 路径设置 ---
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    except NameError:
        current_dir = os.getcwd()
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    load_data_and_setup_globals(project_root)

    print("\n--- 2. 开始遗传算法优化 ---")
    print(f"参数: 种群={POP_SIZE}, 代数={MAX_GEN}, 交叉率={CX_PROB}, 变异率={MUT_PROB}, 仿真次数={N_SIMULATIONS}")

    # --- 初始化种群 ---
    print("正在创建初始种群...")
    population = [create_initial_solution() for _ in range(POP_SIZE)]
    best_solution = None
    best_fitness = -np.inf

    # --- 迭代进化 ---
    for gen in range(MAX_GEN):
        start_time = time.time()
        print(f"\n--- 第 {gen+1}/{MAX_GEN} 代 ---")
        
        # 评估适应度
        fitnesses = [evaluate_fitness(sol) for sol in population]
        
        current_best_fitness = max(fitnesses)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = copy.deepcopy(population[fitnesses.index(current_best_fitness)])
        
        gen_time = time.time() - start_time
        print(f"本代最高适应度 (平均利润): {current_best_fitness:,.2f} 元")
        print(f"历史最优适应度: {best_fitness:,.2f} 元")
        print(f"本代耗时: {gen_time:.2f} 秒")

        # --- 选择 (锦标赛选择) ---
        next_gen_pop_pre_cx = []
        for _ in range(POP_SIZE):
            i, j = random.sample(range(POP_SIZE), 2)
            winner = i if fitnesses[i] > fitnesses[j] else j
            next_gen_pop_pre_cx.append(population[winner])
        
        # --- 交叉与变异 ---
        next_gen_pop = []
        for i in range(0, POP_SIZE, 2):
            p1, p2 = next_gen_pop_pre_cx[i], next_gen_pop_pre_cx[i+1]
            if random.random() < CX_PROB:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            
            if random.random() < MUT_PROB: c1 = mutate(c1)
            if random.random() < MUT_PROB: c2 = mutate(c2)

            next_gen_pop.append(repair_solution(c1))
            next_gen_pop.append(repair_solution(c2))
        
        population = next_gen_pop

    print("\n--- 3. 优化完成 ---")
    print(f"找到的最优策略的预期平均总利润为: {best_fitness:,.2f} 元")

    # --- 格式化并保存结果 ---
    output = []
    for y, plots in best_solution.items():
        for i, seasons in plots.items():
            for k, j in seasons.items():
                if j:
                    output.append({
                        '年份': y, '季节': k, '地块编号': i, '作物名称': j, 
                        '种植面积（亩）': P_AREA[i] 
                    })
    
    result_df = pd.DataFrame(output)
    # 结果保存在当前脚本所在目录下的 'results' 文件夹内
    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'result3.xlsx')
    print(f"\n正在保存最优策略至: {output_path}")
    result_df.to_excel(output_path, index=False)
    print("保存成功！")


if __name__ == '__main__':
    main()