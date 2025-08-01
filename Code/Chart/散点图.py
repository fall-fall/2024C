# 文件名: 散点图.py, 功能同上, 路径已修正
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_scatter_plot():
    # 路径设置
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'result')
        os.makedirs(output_dir, exist_ok=True)
    except NameError:
        output_dir = 'result'
        os.makedirs(output_dir, exist_ok=True)
        
    # !!重要提示!!: 此处需要您对三个最终方案进行评估后的总结数据。
    print("注意：正在使用内置的示例数据。")
    data = {
        '方案名称': ['Q1: 确定性解', 'Q2: 风险规避解', 'Q3: 仿真优化策略'],
        '预期收益 (万元)': [120, 110, 135],
        '风险 (利润标准差)': [40, 15, 25]
    }
    df_summary = pd.DataFrame(data)

    # 可视化
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 'SimHei' 字体未找到。")
        
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    scatter_plot = sns.scatterplot(
        data=df_summary, x='风险 (利润标准差)', y='预期收益 (万元)',
        hue='方案名称', s=200, style='方案名称',
        palette='cividis', edgecolor='black', alpha=0.8
    )
    plt.title('三大核心方案风险-收益对比', fontsize=18, pad=15)
    plt.xlabel('风险 (利润标准差)', fontsize=14)
    plt.ylabel('预期收益 (平均利润)', fontsize=14)
    plt.legend(title='方案', fontsize=12)

    for i in range(df_summary.shape[0]):
        plt.text(df_summary['风险 (利润标准差)'][i] + 0.5, 
                 df_summary['预期收益 (万元)'][i], 
                 df_summary['方案名称'][i])

    plt.axhline(y=df_summary['预期收益 (万元)'].mean(), color='grey', linestyle='--', linewidth=0.8)
    plt.axvline(x=df_summary['风险 (利润标准差)'].mean(), color='grey', linestyle='--', linewidth=0.8)
    min_risk_idx = df_summary['风险 (利润标准差)'].idxmin()
    max_return_idx = df_summary['预期收益 (万元)'].idxmax()
    plt.annotate('理想区域\n(高收益,低风险)', 
                 xy=(df_summary.loc[min_risk_idx, '风险 (利润标准差)'], df_summary.loc[max_return_idx, '预期收益 (万元)']), 
                 xytext=(df_summary['风险 (利润标准差)'].min() - 5, df_summary['预期收益 (万元)'].max() + 5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=12, ha='right')

    # 保存
    output_filename = os.path.join(output_dir, '7_最终方案对比_散点图.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图像已成功保存至: {output_filename}")
    plt.show()

if __name__ == '__main__':
    generate_scatter_plot()