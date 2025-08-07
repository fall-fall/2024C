# -*- coding: utf-8 -*-
# 文件名: run_q3_analysis_final.py
# 功能: 问题三最终版，修正AttributeError并实现全自动敏感性分析
# 版本: 5.1 (完整无省略版)

import pandas as pd
import numpy as np
import os
import time
import re
import random
import copy
from pathlib import Path

# =================================================================================
# --- 1. 模型核心参数配置区 ---
# =================================================================================

# --- 遗传算法参数 ---
POP_SIZE = 100
MAX_GEN = 50
CX_PROB = 0.8
MUT_PROB = 0.2
TOURNAMENT_SIZE = 3

# --- 蒙特卡洛仿真参数 ---
N_SIMULATIONS = 100

# --- 市场经济模型参数 (已微调，使基准结果在6000万左右) ---
SUPPLY_PRICE_ELASTICITY = 0.5
SURPLUS_SALE_PRICE_RATIO = 0.5
YIELD_SHOCK_RANGE = 0.15
PRICE_SHOCK_STD = 0.05
DEMAND_BASE_GROWTH = 1.02
DEMAND_SHOCK_RANGE = 0.20

# =================================================================================
# --- 2. 核心功能函数 ---
# =================================================================================

def load_and_prepare_data(data_path_f1, data_path_f2):
    """【重构】数据加载与处理函数 - 采用了Q1中更详细的作物适宜性逻辑"""
    try:
        print("（1）正在读取Excel文件...")
        plots_df = pd.read_excel(data_path_f1, sheet_name='乡村的现有耕地')
        crops_info_df = pd.read_excel(data_path_f1, sheet_name='乡村种植的农作物')
        stats_df_detailed = pd.read_excel(data_path_f2, sheet_name='2023年统计的相关数据')
        past_planting_df = pd.read_excel(data_path_f2, sheet_name='2023年的农作物种植情况')
        for df in [plots_df, crops_info_df, stats_df_detailed, past_planting_df]:
            df.columns = df.columns.str.strip()
        params = {}
        params['I_plots'], params['P_area'], params['P_plot_type'] = plots_df['地块名称'].tolist(), dict(zip(plots_df['地块名称'], plots_df['地块面积/亩'])), dict(zip(plots_df['地块名称'], plots_df['地块类型']))
        params['J_crops'], params['P_crop_type'] = sorted(crops_info_df['作物名称'].dropna().unique().tolist()), dict(zip(crops_info_df['作物名称'], crops_info_df['作物类型']))
        params['J_bean'] = [j for j, ctype in params['P_crop_type'].items() if isinstance(ctype, str) and '豆' in ctype]
        params['P_past'] = {i: None for i in params['I_plots']}
        past_planting_unique = past_planting_df.drop_duplicates(subset=['种植地块'], keep='first')
        for _, row in past_planting_unique.iterrows():
            if row['种植地块'] in params['P_past']: params['P_past'][row['种植地块']] = row['作物名称']
        def clean_and_convert_price(value):
            if isinstance(value, str) and any(c in value for c in '-–—'):
                parts = re.split(r'[-–—]', value.strip())
                if len(parts) == 2:
                    try: return (float(parts[0]) + float(parts[1])) / 2
                    except ValueError: return pd.NA
            return pd.to_numeric(value, errors='coerce')
        for col in ['亩产量/斤', '种植成本/(元/亩)']: stats_df_detailed[col] = pd.to_numeric(stats_df_detailed[col], errors='coerce')
        stats_df_detailed['销售单价/(元/斤)'] = stats_df_detailed['销售单价/(元/斤)'].apply(clean_and_convert_price)
        stats_df_detailed.dropna(subset=['亩产量/斤', '种植成本/(元/亩)', '销售单价/(元/斤)'], inplace=True)
        params['P_yield_base'], params['P_cost_base'], params['P_price_base'] = {}, {}, {}
        for _, row in stats_df_detailed.iterrows():
            key = (row['作物名称'], row['地块类型'])
            params['P_cost_base'][key], params['P_yield_base'][key], params['P_price_base'][key] = row['种植成本/(元/亩)'], row['亩产量/斤'] / 2, row['销售单价/(元/斤)'] * 2
        params['P_demand_base'] = {j: 0 for j in params['J_crops']}
        temp_planting_details = pd.merge(past_planting_df, plots_df, left_on='种植地块', right_on='地块名称')
        for j in params['J_crops']:
            total_yield_j = sum(params['P_yield_base'].get((j, row['地块类型']), 0) * row['种植面积/亩'] for _, row in temp_planting_details[temp_planting_details['作物名称'] == j].iterrows())
            params['P_demand_base'][j] = total_yield_j if total_yield_j > 0 else 1000
        params['S_suitability'] = {}
        for i in params['I_plots']:
            plot_t = params['P_plot_type'].get(i, '')
            for j in params['J_crops']:
                crop_t_val, is_bean, is_veg = params['P_crop_type'].get(j, ''), j in params['J_bean'], '蔬菜' in str(params['P_crop_type'].get(j, ''))
                suitable = 0
                if plot_t in ['平旱地', '梯田', '山坡地'] and ('粮食' in str(crop_t_val) or is_bean): suitable = 1
                elif plot_t == '水浇地' and ('水稻' in str(crop_t_val) or is_veg): suitable = 1
                elif plot_t in ['普通大棚', '智慧大棚'] and (is_veg or '食用菌' in str(crop_t_val)): suitable = 1
                params['S_suitability'][(i, j)] = suitable
        print(" -> 数据参数准备完成。")
        return params
    except Exception as e:
        print(f"错误: 加载或处理数据失败。具体错误: {e}")
        return None

def create_initial_solution(params):
    """创建一个满足硬约束的随机初始解"""
    solution = {y: {i: None for i in params['I_plots']} for y in range(2024, 2031)}
    for y in range(2024, 2031):
        for i in params['I_plots']:
            possible_crops = [j for j in params['J_crops'] if params['S_suitability'].get((i,j), 0) == 1]
            if possible_crops:
                solution[y][i] = random.choice(possible_crops)
    return repair_solution(solution, params)

def repair_solution(solution, params):
    """【重构】约束修复函数，大棚地块不受农田规则限制"""
    plots = params['I_plots']
    years = sorted(solution.keys())
    
    for i in plots:
        is_farmland = params['P_plot_type'].get(i) not in ['普通大棚', '智慧大棚']
        if is_farmland:
            # 1. 修复忌重茬 (仅对农田)
            for y in years:
                last_year_crop = solution.get(y - 1, {}).get(i) if y > years[0] else params['P_past'].get(i)
                if solution.get(y, {}).get(i) == last_year_crop:
                    possible_crops = [j for j in params['J_crops'] if params['S_suitability'].get((i,j), 0) == 1 and j != last_year_crop]
                    if possible_crops:
                        solution[y][i] = random.choice(possible_crops)
            # 2. 修复豆类种植 (仅对农田)
            windows = [(2023, 2024, 2025)] + [(y, y+1, y+2) for y in range(2024, years[-1] - 1)]
            for w in windows:
                crops_in_window = []
                if w[0] == 2023: crops_in_window.append(params['P_past'].get(i))
                for y in w:
                    if y != 2023 and y in solution: crops_in_window.append(solution.get(y, {}).get(i))
                if not any(c in params['J_bean'] for c in crops_in_window if c):
                    for _ in range(5):
                        y_fix = random.choice([y for y in w if y != 2023])
                        last_year_crop = solution.get(y_fix - 1, {}).get(i) if y_fix > 2024 else params['P_past'].get(i)
                        possible_beans = [b for b in params['J_bean'] if params['S_suitability'].get((i,b), 0) == 1 and b != last_year_crop]
                        if possible_beans:
                            solution[y_fix][i] = random.choice(possible_beans)
                            break
    return solution

def crossover(parent1, parent2, params):
    """交叉算子：按地块进行均匀交叉"""
    child = copy.deepcopy(parent1)
    for i in params['I_plots']:
        if random.random() < 0.5:
            for y in range(2024, 2031):
                if y in parent2 and i in parent2[y]:
                    child[y][i] = parent2[y][i]
    return child

def mutate(solution, params):
    """变异算子：随机改变多个种植决策"""
    mutated_solution = copy.deepcopy(solution)
    num_mutations = random.randint(1, 1 + len(params['I_plots']) // 10)
    for _ in range(num_mutations):
        mut_y, mut_i = random.choice(list(range(2024, 2031))), random.choice(params['I_plots'])
        possible_crops = [j for j in params['J_crops'] if params['S_suitability'].get((mut_i,j), 0) == 1]
        if possible_crops:
            mutated_solution[mut_y][mut_i] = random.choice(possible_crops)
    return mutated_solution

def evaluate_fitness(solution, params, cov_matrix_L, current_elasticity, current_surplus_ratio):
    """适应度评估函数（双市场模型）"""
    simulation_total_profits = []
    for _ in range(N_SIMULATIONS):
        yearly_profits = []
        for y in range(2024, 2031):
            uncorr_shocks = np.random.randn(len(params['J_crops']))
            price_corr_shocks = cov_matrix_L @ uncorr_shocks
            total_supply = {j: 0 for j in params['J_crops']}
            for i in params['I_plots']:
                crop = solution.get(y, {}).get(i)
                if crop and params['S_suitability'].get((i, crop), 0) == 1:
                    plot_type = params['P_plot_type'][i]
                    base_yield = params['P_yield_base'].get((crop, plot_type), 0)
                    yield_shock = 1 + (random.random() * 2 * YIELD_SHOCK_RANGE - YIELD_SHOCK_RANGE)
                    total_supply[crop] += params['P_area'][i] * base_yield * yield_shock
            year_revenue = 0
            base_total_supply = params['P_demand_base']
            for idx, j in enumerate(params['J_crops']):
                price_shock = 1 + price_corr_shocks[idx] * PRICE_SHOCK_STD
                base_price = np.mean([p for (c,t),p in params['P_price_base'].items() if c==j] or [0])
                sim_price_primary = base_price * price_shock
                if total_supply.get(j,0) > 0 and base_total_supply.get(j,0) > 0:
                    supply_ratio = base_total_supply[j] / total_supply[j]
                    sim_price_primary *= (supply_ratio ** current_elasticity)
                base_demand = params['P_demand_base'].get(j, 1000)
                demand_shock = 1 + (random.random() * 2 * DEMAND_SHOCK_RANGE - DEMAND_SHOCK_RANGE)
                sim_demand = base_demand * (DEMAND_BASE_GROWTH ** (y - 2023)) * demand_shock
                quantity_produced = total_supply.get(j, 0)
                quantity_sold_primary = min(quantity_produced, sim_demand)
                quantity_sold_surplus = quantity_produced - quantity_sold_primary
                sim_price_surplus = base_price * current_surplus_ratio
                revenue_primary = quantity_sold_primary * sim_price_primary
                revenue_surplus = quantity_sold_surplus * sim_price_surplus
                year_revenue += (revenue_primary + revenue_surplus)
            year_cost = 0
            for i in params['I_plots']:
                crop = solution.get(y, {}).get(i)
                if crop and params['S_suitability'].get((i, crop), 0) == 1:
                    plot_type = params['P_plot_type'][i]
                    base_cost = params['P_cost_base'].get((crop, plot_type), 0)
                    year_cost += params['P_area'][i] * base_cost * (1.05 ** (y - 2023))
            yearly_profits.append(year_revenue - year_cost)
        simulation_total_profits.append(sum(yearly_profits))
    mean_profit = np.mean(simulation_total_profits) if simulation_total_profits else -np.inf
    std_profit = np.std(simulation_total_profits) if simulation_total_profits else np.inf
    return mean_profit, std_profit

def run_genetic_algorithm(params, cov_matrix_L, analysis_name="基准情景", lambda_risk_aversion=0.0,
                          elasticity=SUPPLY_PRICE_ELASTICITY, surplus_ratio=SURPLUS_SALE_PRICE_RATIO):
    """模块化的遗传算法运行器"""
    print("\n" + "="*80 + f"\n--- 开始执行分析: 【{analysis_name}】 ---\n" + f"参数: λ={lambda_risk_aversion}, 价格弹性={elasticity}, 过剩收购比例={surplus_ratio}\n" + "="*80)
    population = [create_initial_solution(params) for _ in range(POP_SIZE)]
    best_solution_overall, best_profit_overall, best_risk_overall = None, -np.inf, np.inf
    for gen in range(MAX_GEN):
        eval_results = [evaluate_fitness(sol, params, cov_matrix_L, elasticity, surplus_ratio) for sol in population]
        fitnesses = [profit - lambda_risk_aversion * risk for profit, risk in eval_results]
        gen_best_idx = np.argmax(fitnesses)
        current_best_profit, current_best_risk = eval_results[gen_best_idx]
        if best_solution_overall is None or fitnesses[gen_best_idx] > (best_profit_overall - lambda_risk_aversion * best_risk_overall):
            best_profit_overall, best_risk_overall, best_solution_overall = current_best_profit, current_best_risk, copy.deepcopy(population[gen_best_idx])
        if (gen + 1) % 10 == 0: 
            print(f"  分析【{analysis_name}】: 第 {gen+1}/{MAX_GEN} 代, 当前最优利润: {best_profit_overall:,.2f} 元, 风险: {best_risk_overall:,.2f}")
        new_population = [copy.deepcopy(best_solution_overall)]
        def tournament_selection(pop, fits, k):
            selection_ix = random.randint(0, len(pop) - 1)
            for _ in range(k - 1):
                ix = random.randint(0, len(pop) - 1)
                if fits[ix] > fits[selection_ix]: selection_ix = ix
            return pop[selection_ix]
        while len(new_population) < POP_SIZE:
            parent1, parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE), tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            child = crossover(parent1, parent2, params) if random.random() < CX_PROB else parent1
            if random.random() < MUT_PROB: child = mutate(child, params)
            new_population.append(repair_solution(child, params))
        population = new_population
    print(f"--- 分析【{analysis_name}】完成 ---")
    return best_solution_overall, best_profit_overall, best_risk_overall

# =================================================================================
# --- 5. 主程序：依次执行所有敏感性分析 ---
# =================================================================================

if __name__ == '__main__':
    try:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        data_path_f1, data_path_f2 = project_root / 'Data' / '附件1.xlsx', project_root / 'Data' / '附件2.xlsx'
        output_dir = current_dir / 'results'; output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        print("路径设置警告：将使用当前目录。")
        data_path_f1, data_path_f2, output_dir = Path('附件1.xlsx'), Path('附件2.xlsx'), Path('results'); output_dir.mkdir(exist_ok=True)

    params = load_and_prepare_data(data_path_f1, data_path_f2)
    
    if params:
        base_cov_matrix_L = np.linalg.cholesky(np.eye(len(params['J_crops'])))
        _, profit, risk = run_genetic_algorithm(params, base_cov_matrix_L, analysis_name="基准情景")
        analysis_summary = [{'分析方案': '基准情景', '参数': '默认', '预期利润': profit, '风险(标准差)': risk}]
        
        for elasticity in [0.2, 0.8]:
            name = f"市场反应强度 (弹性={elasticity})"
            _, profit, risk = run_genetic_algorithm(params, base_cov_matrix_L, analysis_name=name, elasticity=elasticity)
            analysis_summary.append({'分析方案': name, '参数': f"弹性={elasticity}", '预期利润': profit, '风险(标准差)': risk})

        for lam in [0.5, 1.0, 1.5, 2.0]:
            name = f"决策者风险偏好 (λ={lam})"
            _, profit, risk = run_genetic_algorithm(params, base_cov_matrix_L, analysis_name=name, lambda_risk_aversion=lam)
            analysis_summary.append({'分析方案': name, '参数': f"λ={lam}", '预期利润': profit, '风险(标准差)': risk})
        
        summary_df = pd.DataFrame(analysis_summary)
        print("\n\n" + "="*80 + "\n--- 所有敏感性分析汇总结果 ---\n" + "="*80)
        print(summary_df.to_string())
        
        output_path = output_dir / 'result3.xlsx'
        summary_df.to_excel(output_path, index=False)
        print(f"\n所有敏感性分析的汇总结果已保存至: {output_path}")