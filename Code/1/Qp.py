# -*- coding: utf-8 -*-
# 文件名: solve_q1_with_sensitivity_analysis_cost_relative_path.py

import pandas as pd
import pyomo.environ as pyo
import os
import time
import re
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter

# =============================================================================
# Part 1: 模型与数据加载代码 (与之前相同)
# =============================================================================

def load_and_prepare_data(data_path_f1, data_path_f2):
    """
    最终版数据加载与处理函数。
    """
    try:
        print("正在读取Excel文件...")
        plots_df = pd.read_excel(data_path_f1, sheet_name='乡村的现有耕地')
        crops_info_df = pd.read_excel(data_path_f1, sheet_name='乡村种植的农作物')
        stats_df_detailed = pd.read_excel(data_path_f2, sheet_name='2023年统计的相关数据')
        past_planting_df = pd.read_excel(data_path_f2, sheet_name='2023年的农作物种植情况')
        print(" -> Excel文件读取成功。")

        for df in [plots_df, crops_info_df, stats_df_detailed, past_planting_df]:
            df.columns = df.columns.str.strip()
        
        params = {}
        
        # 1. 地块参数
        params['I_plots'] = plots_df['地块名称'].tolist()
        params['P_area'] = dict(zip(plots_df['地块名称'], plots_df['地块面积/亩']))
        params['P_plot_type'] = dict(zip(plots_df['地块名称'], plots_df['地块类型']))
        
        # 2. 作物参数
        params['J_crops'] = crops_info_df['作物名称'].unique().tolist()
        params['P_crop_type'] = dict(zip(crops_info_df['作物名称'], crops_info_df['作物类型']))
        bean_keywords = ['豆', '豆类']
        
        params['J_bean'] = [
            j for j, ctype in params['P_crop_type'].items() 
            if isinstance(ctype, str) and any(keyword in ctype for keyword in bean_keywords)
        ]

        # 3. 2023年种植历史
        params['P_past'] = {(i, j): 0 for i in params['I_plots'] for j in params['J_crops']}
        for _, row in past_planting_df.iterrows():
            if row['种植地块'] in params['I_plots'] and row['作物名称'] in params['J_crops']:
                params['P_past'][row['种植地块'], row['作物名称']] = 1
                
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
            # 转换为公斤
            params['P_yield'][key] = row['亩产量/斤'] / 2 
            # 转换为元/公斤
            params['P_price'][key] = row['销售单价/(元/斤)'] * 2
            
        # 5. 估算预期销售量 (基于2023年真实总产量，单位：公斤)
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
            # 如果某作物2023年没种，给一个基础需求量避免为0
            params['P_demand'][j] = total_yield_j if total_yield_j > 0 else 1000

        # 6. 种植适宜性矩阵
        params['S_suitability'] = {}
        for i in params['I_plots']:
            plot_t = params['P_plot_type'][i]
            for j in params['J_crops']:
                crop_t = params['P_crop_type'].get(j)
                is_bean = j in params['J_bean']
                for k in [1, 2]:
                    suitable = 0
                    if isinstance(crop_t, str):
                        if plot_t in ['平旱地', '梯田', '山坡地'] and ('粮食' in crop_t or is_bean) and k == 1: suitable = 1
                        elif plot_t == '水浇地':
                            if (crop_t == '水稻' and k==1): suitable = 1
                            if (crop_t == '蔬菜'): suitable = 1
                        elif plot_t == '普通大棚' and ((crop_t == '蔬菜' and k == 1) or (crop_t == '食用菌' and k == 2)): suitable = 1
                        elif plot_t == '智慧大棚' and crop_t == '蔬菜': suitable = 1
                    params['S_suitability'][i, j, k] = suitable

        print(" -> 数据参数准备完成。")
        return params
        
    except FileNotFoundError as e:
        print(f"错误: 文件未找到。请检查您的文件结构，确保脚本能找到'Data'文件夹: {e.filename}")
        return None
    except Exception as e:
        print(f"错误: 加载或处理数据失败。具体错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def solve_q1_model(params, case_type='waste', tee_output=False):
    """
    修改版求解函数，增加tee_output控制是否打印求解器日志，并返回总利润。
    """
    # 此函数与上一版本完全相同，为保持完整性而保留
    # ... (省略)
    print(f"\n--- 正在构建并求解问题一 (情况: {case_type}) ---")
    
    model = pyo.ConcreteModel(f"Q1_Model_{case_type}")
    
    # --- 集合与参数 ---
    model.I = pyo.Set(initialize=params['I_plots'])
    model.J = pyo.Set(initialize=params['J_crops'])
    model.Y = pyo.Set(initialize=list(range(2024, 2031)))
    model.K = pyo.Set(initialize=[1, 2])
    
    # --- 决策变量 ---
    model.x = pyo.Var(model.I, model.J, model.K, model.Y, domain=pyo.NonNegativeReals, bounds=(0, 150))
    model.u = pyo.Var(model.I, model.J, model.K, model.Y, domain=pyo.Binary)
    model.z = pyo.Var(model.I, model.J, model.Y, domain=pyo.Binary)
    model.Sales = pyo.Var(model.J, model.K, model.Y, domain=pyo.NonNegativeReals)
    if case_type == 'discount':
        model.OverSales = pyo.Var(model.J, model.K, model.Y, domain=pyo.NonNegativeReals)

    # --- 目标函数 ---
    def objective_rule(m):
        # 使用不同地块类型的平均价格作为代表价格
        avg_price = {j: np.mean([p for (crop, _), p in params['P_price'].items() if crop == j] or [0]) for j in m.J}
        revenue = sum(avg_price[j] * m.Sales[j,k,y] for j in m.J for k in m.K for y in m.Y)
        if case_type == 'discount':
            revenue += sum(0.5 * avg_price[j] * m.OverSales[j,k,y] for j in m.J for k in m.K for y in m.Y)
        
        total_cost = sum(params['P_cost'].get((j, params['P_plot_type'][i]), 1e9) * m.x[i,j,k,y] 
                         for i in m.I for j in m.J for k in m.K for y in m.Y)
        return revenue - total_cost
    model.profit = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # --- 约束 ---
    def production_rule(m, j, k, y):
        total_prod = sum(params['P_yield'].get((j, params['P_plot_type'][i]), 0) * m.x[i,j,k,y] for i in m.I)
        if case_type == 'waste': return m.Sales[j,k,y] <= total_prod
        else: return m.Sales[j,k,y] + m.OverSales[j,k,y] == total_prod
    model.prod_con = pyo.Constraint(model.J, model.K, model.Y, rule=production_rule)
    
    def demand_rule(m, j, k, y): return m.Sales[j,k,y] <= params['P_demand'].get(j, 0)
    model.demand_con = pyo.Constraint(model.J, model.K, model.Y, rule=demand_rule)
    
    def area_rule(m, i, k, y): return sum(m.x[i,j,k,y] for j in m.J) <= params['P_area'][i]
    model.area_con = pyo.Constraint(model.I, model.K, model.Y, rule=area_rule)

    def suitability_rule(m, i, j, k, y):
        if params['S_suitability'].get((i,j,k), 0) == 0: return m.x[i,j,k,y] == 0
        return pyo.Constraint.Skip
    model.suitability_con = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=suitability_rule)

    A_min = 0.1
    def u_link_upper_rule(m, i, j, k, y): return m.x[i,j,k,y] <= params['P_area'][i] * m.u[i,j,k,y]
    model.u_link_upper = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=u_link_upper_rule)
    def u_link_lower_rule(m, i, j, k, y): return m.x[i,j,k,y] >= A_min * m.u[i,j,k,y]
    model.u_link_lower = pyo.Constraint(model.I, model.J, model.K, model.Y, rule=u_link_lower_rule)

    def z_link_rule(m, i, j, y): return m.z[i,j,y] >= sum(m.u[i,j,k,y] for k in m.K)
    model.z_link_con = pyo.Constraint(model.I, model.J, model.Y, rule=z_link_rule)
    
    def rotation_past_rule(m, i, j):
        if params['P_past'].get((i,j), 0) > 0: return m.z[i,j,2024] <= 0
        return pyo.Constraint.Skip
    model.rotation_past_con = pyo.Constraint(model.I, model.J, rule=rotation_past_rule)
    
    def rotation_future_rule(m, i, j, y):
        if y < 2030: return m.z[i,j,y] + m.z[i,j,y+1] <= 1
        return pyo.Constraint.Skip
    model.rotation_future_con = pyo.Constraint(model.I, model.J, model.Y, rule=rotation_future_rule)

    model.bean_con = pyo.ConstraintList()
    for i in model.I:
        past_bean = sum(params['P_past'].get((i,j), 0) for j in params['J_bean'])
        model.bean_con.add(past_bean + sum(model.z[i,j,y] for j in params['J_bean'] for y in [2024, 2025]) >= 1)
        for y_start in range(2024, 2031 - 2):
            model.bean_con.add(sum(model.z[i,j,y] for j in params['J_bean'] for y in range(y_start, y_start+3)) >= 1)

    # --- 求解 ---
    # 配置cbc求解器绝对路径
    cbc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'CBC', 'bin', 'cbc.exe'))
    solver = pyo.SolverFactory('cbc', executable=cbc_path)
    solver.options['sec'] = 600
    results = solver.solve(model, tee=tee_output)

    # --- 结果处理 (核心修改) ---
    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible]):
        profit_value = pyo.value(model.profit)
        print(f"求解成功！总利润: {profit_value:,.2f} 元")
        return profit_value
    else:
        print(f"求解失败: {results.solver.termination_condition}")
        return None

# =============================================================================
# Part 2: 灵敏度分析模块
# =============================================================================

def run_sensitivity_analysis(base_params, target_crop, param_to_vary, variation_range, case='discount'):
    """
    对指定参数进行灵敏度分析。
    """
    # 此函数与上一版本完全相同，为保持完整性而保留
    # ... (省略)
    print(f"\n===== 开始对 '{target_crop}' 的 '{param_to_vary}' 进行灵敏度分析 =====")
    
    profits_list = []
    
    # 获取基准值 (以所有地块类型的平均值为基准)
    base_values = [v for (c, _), v in base_params[param_to_vary].items() if c == target_crop]
    if not base_values:
        print(f"错误: 在参数 '{param_to_vary}' 中找不到作物 '{target_crop}'。")
        return [], []
    
    for variation in variation_range:
        print(f"\n--- 分析波动: {variation:+.0%} ---")
        
        # 使用深拷贝以防修改原始参数
        analysis_params = copy.deepcopy(base_params)
        
        # 修改目标作物的参数值
        for key, base_value in analysis_params[param_to_vary].items():
            crop_name, _ = key
            if crop_name == target_crop:
                analysis_params[param_to_vary][key] = base_value * (1 + variation)
        
        # 求解模型并获取利润
        # 在灵敏度分析中关闭详细求解日志以保持界面清洁 (tee_output=False)
        profit = solve_q1_model(analysis_params, case_type=case, tee_output=False)
        
        if profit is not None:
            profits_list.append(profit)
        else:
            # 如果某点求解失败, 可以用None或np.nan占位
            profits_list.append(np.nan)
            
    print("\n===== 灵敏度分析完成 =====")
    # 过滤掉求解失败的点
    valid_indices = [i for i, p in enumerate(profits_list) if not np.isnan(p)]
    valid_variations = [variation_range[i] for i in valid_indices]
    valid_profits = [profits_list[i] for i in valid_indices]

    return valid_variations, valid_profits


# =============================================================================
# Part 3: 高级可视化模块
# =============================================================================

def plot_sensitivity_gradient(variations, profits, title, xlabel):
    """
    使用渐变色线图可视化灵敏度分析结果。
    """
    # 此函数与上一版本完全相同，为保持完整性而保留
    # ... (省略)
    if not variations or not profits:
        print("没有可供可视化的有效数据。")
        return

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- 核心：创建渐变色线图 ---
    x = np.array(variations) * 100  # 转换为百分比
    y = np.array(profits) / 10000   # 转换为万元

    # 将数据点连接成线段
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 创建一个颜色映射 (viridis, plasma, inferno, magma 都是不错的选择)
    # 对于成本分析，利润是反向的，我们可以反转颜色映射，或者直接用y值着色，效果同样直观
    cmap = plt.get_cmap('viridis') 
    norm = plt.Normalize(y.min(), y.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(y)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    
    # 可以在线上方添加散点以突出数据点
    ax.scatter(x, y, c=y, cmap=cmap, zorder=10, s=50, edgecolors='black', linewidth=0.5)

    # --- 格式化图表 ---
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('七年总利润 (万元)', fontsize=12)
    
    # 设置坐标轴格式
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val:+.0f}%'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val:,.0f}'))

    # 添加网格线，样式模仿参考图
    ax.grid(True, which='major', linestyle='--', linewidth=0.7, color='lightgray')
    
    # 设置背景色
    ax.set_facecolor('#f8f8f8')
    fig.patch.set_facecolor('white')

    # 调整边距
    plt.tight_layout()
    
    # 保存图像
    output_filename = "sensitivity_analysis_cost_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {output_filename}")
    
    plt.show()


# =============================================================================
# Main: 主程序入口
# =============================================================================
if __name__ == '__main__':
    # --- 核心修改：文件路径已恢复为原始的相对路径逻辑 ---
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 假设脚本在'Code/YourFolder'下，数据在'Data'下，项目根目录是两者的父目录
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        path_f1 = os.path.join(project_root, 'Data', '附件1.xlsx')
        path_f2 = os.path.join(project_root, 'Data', '附件2.xlsx')
    except NameError:
        # 如果在交互式环境（如Jupyter）中运行, 假设数据文件夹与notebook在同一级
        project_root = os.getcwd()
        path_f1 = os.path.join(project_root, 'Data', '附件1.xlsx')
        path_f2 = os.path.join(project_root, 'Data', '附件2.xlsx')

    # 1. 加载数据
    params = load_and_prepare_data(path_f1, path_f2)
    
    if params:
        # 2. 定义分析参数
        # !!! 请在这里修改为您数据中的主要粮食作物名称 !!!
        # 例如: '玉米', '小麦', '水稻' 等
        MAJOR_GRAIN_CROP = '玉米' 
        
        variations_to_test = np.arange(-0.20, 0.21, 0.05)
        
        # 3. 运行灵敏度分析 (以情况二'discount'为例)
        variations, profits = run_sensitivity_analysis(
            base_params=params,
            target_crop=MAJOR_GRAIN_CROP,
            param_to_vary='P_cost', # 分析种植成本
            variation_range=variations_to_test,
            case='discount'
        )
        
        # 4. 可视化结果
        plot_sensitivity_gradient(
            variations, 
            profits,
            title=f'图2：{MAJOR_GRAIN_CROP}种植成本变化与总利润的关联性',
            xlabel=f'{MAJOR_GRAIN_CROP} 种植成本基准值波动'
        )