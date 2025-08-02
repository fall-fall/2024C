# -*- coding: utf-8 -*-
"""
violin_plot_generator.py
完整的小提琴图生成代码，处理数据并可视化两种方案的利润分布
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path
import os

# ==================== 配置部分 ====================
# 设置中文字体 (修改为你的系统字体路径)
FONT_PATH = 'C:/Windows/Fonts/msyh.ttc'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 核心函数 ====================
def load_and_prepare_data(data_dir):
    """加载并准备所有需要的数据"""
    # 读取耕地数据
    plots = pd.read_excel(data_dir / '附件1.xlsx', sheet_name='乡村的现有耕地')
    
    # 读取统计数据并清理列名
    stats = pd.read_excel(data_dir / '附件2.xlsx', sheet_name='2023年统计的相关数据')
    stats.columns = stats.columns.str.strip()
    
    # 处理统计数据的单位转换
    stats['亩产量'] = stats['亩产量/斤'].apply(parse_num) * 0.5  # 斤转公斤
    stats['成本'] = stats['种植成本/(元/亩)'].apply(parse_num)
    stats['单价'] = stats['销售单价/(元/斤)'].apply(parse_num) * 2  # 元/斤转元/公斤
    
    # 返回处理好的参数
    return plots, stats[['作物名称', '地块类型', '亩产量', '成本', '单价']].dropna()

def parse_num(x):
    """解析数字（处理区间和空值）"""
    if pd.isna(x):
        return pd.NA
    x = str(x).strip()
    if '-' in x:
        parts = x.split('-')
        return (float(parts[0]) + float(parts[1])) / 2
    return float(x)

def calculate_profit(plan_df, plots_df, params_df):
    """计算单个方案的利润"""
    # 合并地块信息
    merged = plan_df.merge(
        plots_df[['地块名称', '地块类型']], 
        left_on='地块编号', 
        right_on='地块名称', 
        how='left'
    )
    
    # 合并作物参数
    merged = merged.merge(
        params_df, 
        on=['作物名称', '地块类型'], 
        how='left'
    )
    
    # 计算各项指标
    merged['产量'] = merged['亩产量'] * merged['种植面积（亩）']
    merged['收入'] = merged['产量'] * merged['单价'] / 10000  # 转换为万元
    merged['成本'] = merged['成本'] * merged['种植面积（亩）'] / 10000  # 转换为万元
    merged['profit'] = merged['收入'] - merged['成本']
    
    return merged

def process_scheme(file_path, scheme_name, plots_df, params_df):
    """处理单个方案文件"""
    print(f"正在处理方案: {scheme_name}")
    plan = pd.read_excel(file_path)
    df = calculate_profit(plan, plots_df, params_df)
    
    # 按年份和季节汇总利润
    result = df.groupby(['年份', '季节'])['profit'].sum().reset_index()
    result['方案'] = scheme_name
    
    # 添加调试信息
    print(f"{scheme_name} 方案数据样例:")
    print(result.head())
    print(f"总利润统计:\n{result['profit'].describe()}")
    
    return result

# ==================== 主程序 ====================
def main():
    try:
        # 设置文件路径
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent.parent / 'Data'  # 假设项目结构为2024C/Data/
        
        # 检查文件是否存在
        required_files = ['附件1.xlsx', '附件2.xlsx', 'result2.xlsx', 'result3.xlsx']
        for file in required_files:
            if not (data_dir / file).exists():
                raise FileNotFoundError(f"未找到文件: {data_dir/file}")
        
        # 加载基础数据
        plots, params = load_and_prepare_data(data_dir)
        
        # 处理两个方案
        result2 = process_scheme(
            data_dir / 'result2.xlsx', 
            'Q2-Robust', 
            plots, 
            params
        )
        
        result3 = process_scheme(
            data_dir / 'result3.xlsx', 
            'P3-GA', 
            plots, 
            params
        )
        
        # 合并结果
        df = pd.concat([result2, result3])
        print("\n合并后的数据统计:")
        print(df.groupby('方案')['profit'].describe())
        
        # ============== 绘制小提琴图 ==============
        plt.figure(figsize=(10, 6))
        
        # 创建小提琴图
        ax = sns.violinplot(
            data=df, 
            x='方案', 
            y='profit',
            palette='pastel',
            inner='quartile',  # 显示四分位数线
            cut=0,            # 紧贴数据范围
            scale='width'      # 统一宽度
        )
        
        # 添加统计标记
        for i, scheme in enumerate(df['方案'].unique()):
            median = df[df['方案'] == scheme]['profit'].median()
            ax.text(i, median, f'{median:.2f}', 
                   ha='center', va='center', 
                   fontweight='bold', color='white',
                   bbox=dict(facecolor='black', alpha=0.5))
        
        # 设置图表样式
        plt.title('两种种植方案的利润分布对比\n(单位：万元)', 
                 fontsize=16, pad=20)
        plt.xlabel('方案类型', fontsize=12)
        plt.ylabel('利润 (万元)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存和显示
        output_dir = data_dir / 'result'
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / 'profit_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存至: {output_path}")
        plt.show()
        
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()