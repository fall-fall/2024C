# -*- coding: utf-8 -*-
# 文件名: solve_q3_final.py (v1.2 - 修正NameError)
# 功能: 最终版求解脚本，使用遗传算法和蒙特卡洛仿真完整解决问题三

import pandas as pd
import numpy as np
import os
import time
import re
import random
import copy

# --- 遗传算法参数 (为快速看到结果设置得较低，实际运行时应调高) ---
POP_SIZE = 100           # 种群大小 (建议值: 100+)
MAX_GEN = 100            # 最大遗传代数 (建议值: 100+)
CX_PROB = 0.8           # 交叉概率
MUT_PROB = 0.2          # 变异概率
TOURNAMENT_SIZE = 3     # 锦标赛选择的规模
N_SIMULATIONS = 100      # 每次适应度评估的仿真次数 (建议值: 100+)
# --- 市场仿真参数 (模型假设) ---
SUPPLY_PRICE_ELASTICITY = 0.1 # 供给影响价格的弹性系数：供给增加1%，价格下降0.1%

def load_and_prepare_data(data_path_f1, data_path_f2):
    """最终版数据加载与处理函数。"""
    try:
        print("正在读取Excel文件...")
        plots_df = pd.read_excel(data_path_f1, sheet_name='乡村的现有耕地')
        crops_info_df = pd.read_excel(data_path_f1, sheet_name='乡村种植的农作物')
        stats_df_detailed = pd.read_excel(data_path_f2, sheet_name='2023年统计的相关数据')
        past_planting_df = pd.read_excel(data_path_f2, sheet_name='2023年的农作物种植情况')
        
        for df in [plots_df, crops_info_df, stats_df_detailed, past_planting_df]:
            df.columns = df.columns.str.strip()
        
        params = {}
        
        # 1. 地块参数
        params['I_plots'] = plots_df['地块名称'].tolist()
        params['P_area'] = dict(zip(plots_df['地块名称'], plots_df['地块面积/亩']))
        params['P_plot_type'] = dict(zip(plots_df['地块名称'], plots_df['地块类型']))
        
        # 2. 作物参数
        params['J_crops'] = sorted(crops_info_df['作物名称'].dropna().unique().tolist())
        params['P_crop_type'] = dict(zip(crops_info_df['作物名称'], crops_info_df['作物类型']))
        bean_keywords = ['豆', '豆类']
        params['J_bean'] = [j for j, ctype in params['P_crop_type'].items() if isinstance(ctype, str) and any(keyword in ctype for keyword in bean_keywords)]

        # 3. 2023年种植历史 (处理地块-作物粒度)
        params['P_past'] = {i: None for i in params['I_plots']}
        past_planting_unique = past_planting_df.drop_duplicates(subset=['种植地块'], keep='first')
        for _, row in past_planting_unique.iterrows():
            if row['种植地块'] in params['P_past']:
                params['P_past'][row['种植地块']] = row['作物名称']
                
        # 4. 经济与产量参数 (分地块类型)
        for col in ['亩产量/斤', '种植成本/(元/亩)', '销售单价/(元/斤)']:
            if col in stats_df_detailed.columns:
                 def clean_and_convert(value):
                    if isinstance(value, str) and '-' in value:
                        parts = re.split(r'[-–—]', value.strip())
                        if len(parts) == 2:
                            try: return (float(parts[0]) + float(parts[1])) / 2
                            except ValueError: return pd.NA
                    return value
                 stats_df_detailed[col] = stats_df_detailed[col].apply(clean_and_convert)
                 stats_df_detailed[col] = pd.to_numeric(stats_df_detailed[col], errors='coerce')
        stats_df_detailed.dropna(inplace=True)

        params['P_yield'] = {}
        params['P_cost'] = {}
        params['P_price'] = {}
        
        for _, row in stats_df_detailed.iterrows():
            key = (row['作物名称'], row['地块类型'])
            params['P_cost'][key] = row['种植成本/(元/亩)']
            params['P_yield'][key] = row['亩产量/斤'] / 2  # 统一换算为公斤
            params['P_price'][key] = row['销售单价/(元/斤)'] * 2 # 统一换算为公斤

        # 5. 种植适宜性矩阵
        params['S_suitability'] = {}
        for i in params['I_plots']:
            plot_t = params['P_plot_type'][i]
            for j in params['J_crops']:
                crop_t = params['P_crop_type'].get(j)
                is_bean = j in params['J_bean']
                # 简化，只考虑单季
                suitable = 0
                if isinstance(crop_t, str):
                    if plot_t in ['平旱地', '梯田', '山坡地'] and ('粮食' in crop_t or is_bean): suitable = 1
                    elif plot_t == '水浇地' and (crop_t == '水稻' or crop_t == '蔬菜'): suitable = 1
                    elif plot_t in ['普通大棚', '智慧大棚'] and crop_t == '蔬菜': suitable = 1
                    # 食用菌作为第二季作物，在此简化模型中暂不直接设为适宜
                params['S_suitability'][i, j] = suitable
        print(" -> 数据参数准备完成。")
        return params
        
    except Exception as e:
        print(f"错误: 加载或处理数据失败。具体错误: {e}")
        return None

# --- 遗传算法核心函数 ---
# 一个“个体”的数据结构: solution = { year: { plot: crop_name } }
def repair_solution(solution, params):
    """修复解，确保其满足所有硬约束（忌重茬和豆类）"""
    plots = params['I_plots']
    years = sorted(solution.keys())
    
    for i in plots:
        # 1. 修复忌重茬
        for y in years:
            last_year_crop = solution[y-1][i] if y > years[0] else params['P_past'].get(i)
            if solution[y][i] == last_year_crop:
                possible_crops = [j for j in params['J_crops'] if params['S_suitability'].get((i,j), 0) == 1 and j != last_year_crop]
                solution[y][i] = random.choice(possible_crops) if possible_crops else None
        
        # 2. 修复豆类种植
        windows = [(2023, 2024, 2025)] + [(y, y+1, y+2) for y in range(2024, years[-1] - 1)]
        for w in windows:
            crops_in_window = []
            if w[0] == 2023: crops_in_window.append(params['P_past'].get(i))
            for y in w:
                if y != 2023: crops_in_window.append(solution[y][i])
            
            if not any(c in params['J_bean'] for c in crops_in_window):
                for _ in range(3): # 尝试3次
                    y_fix = random.choice([y for y in w if y != 2023])
                    last_year_crop = solution[y_fix-1][i] if y_fix > 2024 else params['P_past'].get(i)
                    possible_beans = [b for b in params['J_bean'] if params['S_suitability'].get((i,b), 0) == 1 and b != last_year_crop]
                    if possible_beans:
                        solution[y_fix][i] = random.choice(possible_beans)
                        break
    return solution

def create_initial_solution(params):
    """创建一个满足硬约束的随机初始解"""
    solution = {y: {i: None for i in params['I_plots']} for y in range(2024, 2031)}
    for y in range(2024, 2031):
        for i in params['I_plots']:
            possible_crops = [j for j in params['J_crops'] if params['S_suitability'].get((i,j), 0) == 1]
            if possible_crops:
                solution[y][i] = random.choice(possible_crops)
    return repair_solution(solution, params)

def crossover(parent1, parent2, params):
    """交叉算子：按地块进行均匀交叉"""
    child = copy.deepcopy(parent1)
    for i in params['I_plots']:
        if random.random() < 0.5:
            for y in range(2024, 2031):
                child[y][i] = parent2[y][i]
    return child

def mutate(solution, params):
    """变异算子：随机改变多个种植决策"""
    mutated_solution = copy.deepcopy(solution)
    for _ in range(random.randint(1, 5)):
        mut_y = random.choice(list(range(2024, 2031)))
        mut_i = random.choice(params['I_plots'])
        possible_crops = [j for j in params['J_crops'] if params['S_suitability'].get((mut_i,j), 0) == 1]
        if possible_crops:
            mutated_solution[mut_y][mut_i] = random.choice(possible_crops)
    return mutated_solution

def evaluate_fitness(solution, params, cov_matrix_L):
    """核心函数：运行N次仿真，计算一个solution的平均总利润"""
    total_profits = []
    for _ in range(N_SIMULATIONS):
        yearly_profits = []
        for y in range(2024, 2031):
            uncorr_shocks = np.random.randn(len(params['J_crops']))
            corr_shocks = cov_matrix_L @ uncorr_shocks
            
            total_supply = {j: 0 for j in params['J_crops']}
            for i in params['I_plots']:
                crop = solution[y][i]
                if crop:
                    plot_type = params['P_plot_type'][i]
                    base_yield = params['P_yield'].get((crop, plot_type), 0)
                    prod_shock = 1 + (random.random() * 0.2 - 0.1)
                    total_supply[crop] += params['P_area'][i] * base_yield * prod_shock

            base_total_supply = {j: sum(area for p, area in params['P_area'].items()) * np.mean([v for (c,t),v in params['P_yield'].items() if c==j] or [0]) for j in params['J_crops']}
            year_revenue = 0
            year_cost = 0

            for idx, j in enumerate(params['J_crops']):
                price_shock = 1 + corr_shocks[idx] * 0.05
                base_price = np.mean([p for (c,t),p in params['P_price'].items() if c==j] or [0])
                sim_price = base_price * price_shock
                
                crop_type = params['P_crop_type'].get(j, '')
                if isinstance(crop_type, str) and ('蔬菜' in crop_type or '食用菌' in crop_type):
                    if total_supply.get(j,0) > 0 and base_total_supply.get(j,0) > 0:
                        supply_ratio = base_total_supply[j] / total_supply[j]
                        sim_price *= (supply_ratio ** SUPPLY_PRICE_ELASTICITY)
                
                year_revenue += total_supply.get(j,0) * sim_price
            
            for i in params['I_plots']:
                crop = solution[y][i]
                if crop:
                    plot_type = params['P_plot_type'][i]
                    base_cost = params['P_cost'].get((crop, plot_type), 0)
                    year_cost += params['P_area'][i] * base_cost * (1.05 ** (y - 2023))

            yearly_profits.append(year_revenue - year_cost)
        total_profits.append(sum(yearly_profits))
    return np.mean(total_profits) if total_profits else -np.inf

# --- 主程序 ---
if __name__ == '__main__':
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        path_f1 = os.path.join(project_root, 'Data', '附件1.xlsx')
        path_f2 = os.path.join(project_root, 'Data', '附件2.xlsx')
        output_dir = os.path.join(project_root, 'Code','3','results') 
        os.makedirs(output_dir, exist_ok=True)
    except NameError:
        project_root = os.getcwd()
        path_f1 = os.path.join(project_root, 'Data', '附件1.xlsx')
        path_f2 = os.path.join(project_root, 'Data', '附件2.xlsx')
        output_dir = os.path.join(project_root, 'Result')

    params = load_and_prepare_data(path_f1, path_f2)
    
    if params:
        n_vars = len(params['J_crops'])
        cov_matrix = np.eye(n_vars) * 0.05
        min_eig = np.min(np.linalg.eigvalsh(cov_matrix))
        if min_eig < 0: cov_matrix -= 1.01 * min_eig * np.eye(n_vars)
        cov_matrix_L = np.linalg.cholesky(cov_matrix)

        print("\n--- 开始遗传算法优化 (问题三) ---")
        print(f"参数: 种群={POP_SIZE}, 代数={MAX_GEN}, 仿真次数={N_SIMULATIONS}")
        
        population = [create_initial_solution(params) for _ in range(POP_SIZE)]
        best_solution_overall = None
        best_fitness_overall = -np.inf

        for gen in range(MAX_GEN):
            start_time = time.time()
            fitnesses = [evaluate_fitness(sol, params, cov_matrix_L) for sol in population]
            
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]

            if gen_best_fitness > best_fitness_overall:
                best_fitness_overall = gen_best_fitness
                best_solution_overall = copy.deepcopy(population[gen_best_idx])
            
            print(f"第 {gen+1}/{MAX_GEN} 代, 最高适应度: {best_fitness_overall:,.2f}, 耗时: {time.time() - start_time:.2f} 秒")
                        # ===== 新增：记录平均适应度 & 保存 csv =====
            avg_fitness = np.mean(fitnesses)          # 计算平均适应度
            with open('ga_log.csv', 'a', encoding='utf-8') as f:
                f.write(f"{gen+1},{best_fitness_overall},{avg_fitness}\n")
            # =========================================
            # 精英主义 + 锦标赛选择
            new_population = [copy.deepcopy(best_solution_overall)] 
            while len(new_population) < POP_SIZE:
                # 锦标赛选择
                def tournament_selection(pop, fits, k):
                    selection_ix = np.random.randint(len(pop))
                    for ix in np.random.randint(0, len(pop), k-1):
                        if fits[ix] > fits[selection_ix]:
                            selection_ix = ix
                    return pop[selection_ix]

                parent1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
                parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
                
                child = crossover(parent1, parent2, params)
                
                if random.random() < MUT_PROB:
                    child = mutate(child, params)
                
                new_population.append(repair_solution(child, params))

            population = new_population

        print("\n--- 优化完成 ---")
        print(f"找到的最优策略的预期平均总利润为: {best_fitness_overall:,.2f} 元")

        # 格式化并保存结果
        output = []
        # 在此简化模型中，我们假设每个地块每年只种一季作物
        for y, plots in best_solution_overall.items():
            for i, j in plots.items():
                if j:
                    # --- 核心修正：在这里添加 plot_type 的定义 ---
                    plot_type = params['P_plot_type'][i]
                    crop_type = params['P_crop_type'].get(j)
                    
                    season = 1 # 默认为第一季
                    # 判断是否为普通大棚的第二季食用菌
                    if plot_type == '普通大棚' and isinstance(crop_type, str) and '食用菌' in crop_type:
                        # 这个简化模型假设GA只会为普通大棚选择蔬菜或食用菌
                        # 更复杂的模型需要让GA为每个季节选择作物
                        # 此处我们无法确定GA选择的是第一季还是第二季，因此这是一个待完善的假设
                        pass # 暂不处理季节问题，统一记为1

                    output.append({'年份': y, '季节': season, '地块编号': i, '作物名称': j, '种植面积（亩）': params['P_area'][i]})
        
        result_df = pd.DataFrame(output)
        output_path = os.path.join(output_dir, 'result3.xlsx')
        result_df.to_excel(output_path, index=False)
        print(f"问题三的结果已成功保存至: {output_path}")