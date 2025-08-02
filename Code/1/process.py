# -*- coding: utf-8 -*-
# 文件名: solve_q1_final.py (v1.9 - 修复超时结果输出问题)

import pandas as pd
import pyomo.environ as pyo
import os
import time
import re
import numpy as np
import copy
import warnings
from pathlib import Path

def load_and_prepare_data(data_path_f1, data_path_f2):
    """
    数据加载与处理函数。
    - 统一处理列名。
    - 清理区间数据，并转换为数值。
    - 填充缺失的'作物-地块类型'组合的参数，使用中位数作为默认值。
    - 识别豆类作物，并处理历史种植数据。
    - 构建适宜性参数。
    """
    try:
        print("正在读取Excel文件...")
        plots_df = pd.read_excel(data_path_f1, sheet_name='乡村的现有耕地')
        crops_info_df = pd.read_excel(data_path_f1, sheet_name='乡村种植的农作物')
        stats_df_detailed = pd.read_excel(data_path_f2, sheet_name='2023年统计的相关数据')
        past_planting_df = pd.read_excel(data_path_f2, sheet_name='2023年的农作物种植情况')
        print(" -> Excel文件读取成功。")

    except Exception as e:
        print(f"错误: 读取Excel文件失败。具体错误: {e}")
        return None

    # 数据清洗：统一列名
    for df in [plots_df, crops_info_df, stats_df_detailed, past_planting_df]:
        df.columns = df.columns.str.strip()

    params = {}
    params['I_plots'] = plots_df['地块名称'].tolist()
    params['P_area'] = dict(zip(plots_df['地块名称'], plots_df['地块面积/亩']))
    params['P_plot_type'] = dict(zip(plots_df['地块名称'], plots_df['地块类型']))

    params['J_crops'] = crops_info_df['作物名称'].unique().tolist()
    params['P_crop_type'] = dict(zip(crops_info_df['作物名称'], crops_info_df['作物类型']))
    bean_keywords = ['豆', '豆类']
    params['J_bean'] = [
        j for j, ctype in params['P_crop_type'].items()
        if isinstance(ctype, str) and any(keyword in ctype for keyword in bean_keywords)
    ]

    # 历史种植情况
    params['P_past'] = {(i, j): 0 for i in params['I_plots'] for j in params['J_crops']}
    for _, row in past_planting_df.iterrows():
        if row['种植地块'] in params['I_plots'] and row['作物名称'] in params['J_crops']:
            params['P_past'][row['种植地块'], row['作物名称']] = 1

    # 处理区间值并填充缺失值
    def clean_and_convert(value):
        if isinstance(value, str) and ('-' in value or '–' in value or '—' in value):
            parts = re.split(r'[-–—]', value.strip())
            try:
                if len(parts) == 2:
                    return (float(parts[0]) + float(parts[1])) / 2
            except ValueError:
                return pd.NA
        try:
            return float(value)
        except (ValueError, TypeError):
            return pd.NA

    for col in ['亩产量/斤', '种植成本/(元/亩)', '销售单价/(元/斤)']:
        if col in stats_df_detailed.columns:
            stats_df_detailed[col] = stats_df_detailed[col].apply(clean_and_convert)
            stats_df_detailed[col] = pd.to_numeric(stats_df_detailed[col], errors='coerce')
    stats_df_detailed.dropna(inplace=True)
    
    # 计算默认值
    default_yield = stats_df_detailed['亩产量/斤'].median() / 2 if not stats_df_detailed.empty else 500  # 斤转公斤
    default_cost = stats_df_detailed['种植成本/(元/亩)'].median() if not stats_df_detailed.empty else 500
    default_price = stats_df_detailed['销售单价/(元/斤)'].median() * 2 if not stats_df_detailed.empty else 10 # 元/斤转元/公斤

    # 初始化并填充生产参数
    params['P_yield'] = {}
    params['P_cost'] = {}
    params['P_price'] = {}
    
    for _, row in stats_df_detailed.iterrows():
        key = (row['作物名称'], row['地块类型'])
        params['P_cost'][key] = row['种植成本/(元/亩)']
        params['P_yield'][key] = row['亩产量/斤'] / 2
        params['P_price'][key] = row['销售单价/(元/斤)'] * 2

    missing_combinations = 0
    for crop in params['J_crops']:
        for plot_type in plots_df['地块类型'].unique():
            key = (crop, plot_type)
            if key not in params['P_yield']:
                params['P_yield'][key] = default_yield
                params['P_cost'][key] = default_cost
                params['P_price'][key] = default_price
                missing_combinations += 1
    if missing_combinations > 0:
        warnings.warn(f"警告: 为{missing_combinations}个作物-地块类型组合使用了默认值")

    # 计算需求基准
    params['P_demand'] = {j: 0 for j in params['J_crops']}
    temp_planting_details = pd.merge(past_planting_df, plots_df, left_on='种植地块', right_on='地块名称')
    for j in params['J_crops']:
        total_yield_j = 0
        crop_plantings = temp_planting_details[temp_planting_details['作物名称'] == j]
        for _, planting_row in crop_plantings.iterrows():
            plot_type = planting_row['地块类型']
            area = planting_row['种植面积/亩']
            key = (j, plot_type)
            if key in params['P_yield']:
                total_yield_j += params['P_yield'][key] * area
        params['P_demand'][j] = total_yield_j if total_yield_j > 0 else 1000

    # 地块适宜性规则
    params['S_suitability'] = {}
    for i in params['I_plots']:
        plot_t = params['P_plot_type'].get(i)
        if not plot_t: continue
        for j in params['J_crops']:
            crop_t = params['P_crop_type'].get(j)
            if not isinstance(crop_t, str): continue
            is_bean = j in params['J_bean']
            for k in [1, 2]:
                suitable = 0
                if plot_t in ['平旱地', '梯田', '山坡地'] and ('粮食' in crop_t or is_bean) and k == 1: suitable = 1
                elif plot_t == '水浇地':
                    if (crop_t == '水稻' and k == 1) or ('蔬菜' in crop_t): suitable = 1
                elif plot_t == '普通大棚' and (('蔬菜' in crop_t and k == 1) or ('食用菌' in crop_t and k == 2)): suitable = 1
                elif plot_t == '智慧大棚' and ('蔬菜' in crop_t): suitable = 1
                params['S_suitability'][i, j, k] = suitable

    print(" -> 数据参数准备完成。")
    return params

def build_model(params, case_type):
    """构建优化模型"""
    model = pyo.ConcreteModel(f"Q1_Model_{case_type}")
    
    # 集合定义
    model.I = pyo.Set(initialize=params['I_plots'])  # 地块
    model.J = pyo.Set(initialize=params['J_crops'])  # 作物
    model.Y = pyo.Set(initialize=list(range(2024, 2031)))  # 年份
    model.K = pyo.Set(initialize=[1, 2])             # 季节
    
    # 决策变量
    model.x = pyo.Var(model.I, model.J, model.K, model.Y, 
                     domain=pyo.NonNegativeReals, bounds=(0, 150))  # 种植面积
    model.u = pyo.Var(model.I, model.J, model.K, model.Y, 
                     domain=pyo.Binary)  # 是否种植
    model.z = pyo.Var(model.I, model.J, model.Y, 
                     domain=pyo.Binary)  # 年度种植标记
    
    # 销售变量
    model.Sales = pyo.Var(model.J, model.K, model.Y, 
                         domain=pyo.NonNegativeReals)
    if case_type == 'discount':
        model.OverSales = pyo.Var(model.J, model.K, model.Y, 
                                 domain=pyo.NonNegativeReals)
    
    # 目标函数
    def objective_rule(m):
        # 收入
        revenue = sum(params['P_price'].get((j, params['P_plot_type'].get(i)), 0) * m.Sales[j,k,y]
                      for i in m.I for j in m.J for k in m.K for y in m.Y)
        if case_type == 'discount':
            revenue += sum(0.5 * params['P_price'].get((j, params['P_plot_type'].get(i)), 0) * m.OverSales[j,k,y]
                           for i in m.I for j in m.J for k in m.K for y in m.Y)
        # 成本
        total_cost = sum(params['P_cost'].get((j, params['P_plot_type'].get(i)), 0) * m.x[i,j,k,y]
                         for i in m.I for j in m.J for k in m.K for y in m.Y)
        return revenue - total_cost
    
    model.profit = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # 约束条件
    def dispersion_rule(m, j, k, y): return sum(m.u[i, j, k, y] for i in m.I) <= 10
    model.dispersion_con = pyo.Constraint(model.J, model.K, model.Y, rule=dispersion_rule)
    def production_rule(m, j, k, y):
        total_prod = sum(params['P_yield'].get((j, params['P_plot_type'][i]), 0) * m.x[i, j, k, y] for i in m.I)
        if case_type == 'waste': return m.Sales[j, k, y] <= total_prod
        else: return m.Sales[j, k, y] + m.OverSales[j, k, y] == total_prod
    model.prod_con = pyo.Constraint(model.J, model.K, model.Y, rule=production_rule)
    def demand_rule(m, j, k, y): return m.Sales[j, k, y] <= params['P_demand'].get(j, 0)
    model.demand_con = pyo.Constraint(model.J, model.K, model.Y, rule=demand_rule)
    def area_rule(m, i, k, y): return sum(m.x[i, j, k, y] for j in m.J) <= params['P_area'][i]
    model.area_con = pyo.Constraint(model.I, model.K, model.Y, rule=area_rule)
    def suitability_rule(m, i, j, k, y):
        if params['S_suitability'].get((i, j, k), 0) == 0: return m.x[i, j, k, y] == 0
        return pyo.Constraint.Skip
    model.suitability_con = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=suitability_rule)
    def u_link_upper_rule(m, i, j, k, y): return m.x[i, j, k, y] <= params['P_area'][i] * m.u[i, j, k, y]
    model.u_link_upper = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=u_link_upper_rule)
    A_min = 0.1
    def u_link_lower_rule(m, i, j, k, y): return m.x[i, j, k, y] >= A_min * m.u[i, j, k, y]
    model.u_link_lower = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=u_link_lower_rule)
    def z_link_rule(m, i, j, y): return m.z[i, j, y] >= sum(m.u[i, j, k, y] for k in m.K)
    model.z_link_con = pyo.Constraint(model.I, model.J, model.Y, rule=z_link_rule)
    def rotation_past_rule(m, i, j):
        if params['P_past'].get((i, j), 0) > 0: return m.z[i, j, 2024] <= 0
        return pyo.Constraint.Skip
    model.rotation_past_con = pyo.Constraint(model.I, model.J, rule=rotation_past_rule)
    def rotation_future_rule(m, i, j, y):
        if y < 2030: return m.z[i, j, y] + m.z[i, j, y + 1] <= 1
        return pyo.Constraint.Skip
    model.rotation_future_con = pyo.Constraint(model.I, model.J, model.Y, rule=rotation_future_rule)
    model.bean_con = pyo.ConstraintList()
    for i in model.I:
        if i in params['P_plot_type']:
            past_bean = sum(params['P_past'].get((i, j), 0) for j in params['J_bean'])
            model.bean_con.add(past_bean + sum(model.z[i, j, y] for j in params['J_bean'] for y in [2024, 2025]) >= 1)
            for y_start in range(2024, 2031 - 2):
                model.bean_con.add(sum(model.z[i, j, y] for j in params['J_bean'] for y in range(y_start, y_start + 3)) >= 1)

    print(" -> 模型构建完成。")
    return model

def build_and_solve_once(params, case_type, solver_path, timeout=600):
    """
    构建并求解优化模型，只进行一次求解器调用，超时后输出当前最佳解。
    """
    print(f"\n--- 正在构建并求解问题一 (情况: {case_type}) ---")
    print(f"-> 求解时间限制: {timeout} 秒")
    
    model = build_model(params, case_type)

    start_time = time.time()
    solver = pyo.SolverFactory('cbc', executable=solver_path)
    solver.options['sec'] = timeout
    
    try:
        results = solver.solve(model, tee=True)
    except Exception as e:
        print(f"求解过程中出错: {str(e)}")
        return None
    end_time = time.time()
    
    # 检查求解结果。如果终止状态是 maxTimeLimit，即使 status 是 aborted，也认为是一个可接受的解决方案。
    if (results.solver.termination_condition == pyo.TerminationCondition.maxTimeLimit or
        (results.solver.status == pyo.SolverStatus.ok and
         results.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible])):
        
        print(f"\n求解成功！终止状态: {results.solver.termination_condition}")
        print(f"求解耗时: {end_time - start_time:.2f}秒")
        
        # 检查是否找到了可行解
        if pyo.value(model.profit) is not None:
            print(f"最终目标值: {pyo.value(model.profit):.2f}")
            
            output = []
            for v in model.x.values():
                val = pyo.value(v, exception=False)
                if val is not None and val > 0.01:
                    i, j, k, y = v.index()
                    output.append({
                        '年份': y,
                        '季节': k,
                        '地块编号': i,
                        '作物名称': j,
                        '种植面积（亩）': round(val, 4)
                    })
            return pd.DataFrame(output)
        else:
            print("警告: 求解器在超时前未能找到任何可行解。")
            return None
    else:
        print(f"\n求解失败，状态: {results.solver.termination_condition}")
        return None

# --- 主程序 ---
if __name__ == '__main__':
    try:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        path_f1 = project_root / 'Data' / '附件1.xlsx'
        path_f2 = project_root / 'Data' / '附件2.xlsx'
        output_dir = project_root / 'Code' / '1' / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)

        solver_path = project_root / 'CBC' / 'bin' / 'cbc.exe'
        if not solver_path.exists():
            print("="*50)
            print(f"错误：未找到求解器！请检查路径：\n{solver_path}")
            print("请确保您已将CBC文件夹解压到项目根目录'2024C'下。")
            print("="*50)
            exit()

        params = load_and_prepare_data(path_f1, path_f2)
        if params is None:
            raise RuntimeError("数据加载失败，无法继续。")
            
        # 情况一：浪费模式
        result_waste = build_and_solve_once(copy.deepcopy(params), 'waste', str(solver_path))
        if result_waste is not None and not result_waste.empty:
            output_path1 = output_dir / 'result1_1.xlsx'
            result_waste.to_excel(output_path1, index=False)
            print(f"情况一的结果已成功保存至: {output_path1}")
        else:
            print("情况一未能找到可行的种植方案。")

        # 情况二：折扣模式
        result_discount = build_and_solve_once(copy.deepcopy(params), 'discount', str(solver_path))
        if result_discount is not None and not result_discount.empty:
            output_path2 = output_dir / 'result1_2.xlsx'
            result_discount.to_excel(output_path2, index=False)
            print(f"情况二的结果已成功保存至: {output_path2}")
        else:
            print("情况二未能找到可行的种植方案。")
            
        print("\n所有求解完成！")

    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()