import pandas as pd
import os

def create_yearly_summary_tables(result_df, all_plots_list, all_crops_list, output_excel_path):
    """
    为每一年生成一个格式完全固定的汇总表。
    - 行：地块编号 + 季节 (按季节优先排序)
    - 列：所有作物
    - 单元格：种植面积
    """
    try:
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            
            if '年份' not in result_df.columns:
                print("错误：原始结果文件中缺少'年份'列。")
                return
            
            years = sorted(result_df['年份'].unique())
            all_seasons = sorted(result_df['季节'].unique())

            for year in years:
                print(f"--> 正在处理 {year} 年的数据...")
                
                df_year = result_df[result_df['年份'] == year]

                # ##################################################################
                # 关键逻辑修正：将表格变回“纵向”布局
                # 行索引(index)为地块+季节，列索引(columns)为作物
                # ##################################################################
                summary_table = df_year.pivot_table(
                    index=['地块编号', '季节'],
                    columns='作物名称',
                    values='种植面积（亩）',
                    aggfunc='sum',
                    fill_value=0
                )
                
                # --- 格式化与补全 ---
                
                # 1. 补全缺失的作物列，并按指定顺序排序
                summary_table = summary_table.reindex(columns=all_crops_list, fill_value=0)
                
                # 2. 补全缺失的地块+季节行
                full_multi_index = pd.MultiIndex.from_product(
                    [all_plots_list, all_seasons], 
                    names=['地块编号', '季节']
                )
                summary_table = summary_table.reindex(index=full_multi_index, fill_value=0)

                # 3. 按“季节”优先，再按“地块编号”对行进行排序
                summary_table.sort_index(level=['季节', '地块编号'], inplace=True)

                # 4. Sheet命名使用纯数字年份
                summary_table.to_excel(writer, sheet_name=str(year))

        print(f"\n所有年份的汇总表已成功保存至: {output_excel_path}")

    except Exception as e:
        print(f"在创建年度汇总表时发生错误: {e}")


def main():
    """
    主函数，读取原始结果并调用格式化函数。
    """
    print("--- 开始运行最终格式转换脚本 ---")
    
    try:
        current_dir = os.path.dirname(__file__)
        results_dir = os.path.join(current_dir, 'results')
        data_dir = os.path.join(current_dir, '..', '..', 'Data')

        print("正在从附件读取完整的地块和作物列表...")
        path_f1 = os.path.join(data_dir, '附件1.xlsx')
        
        plots_df = pd.read_excel(path_f1, sheet_name='乡村的现有耕地')
        plots_df.columns = plots_df.columns.str.strip()
        all_plots_list = plots_df['地块名称'].tolist()

        crops_info_df = pd.read_excel(path_f1, sheet_name='乡村种植的农作物')
        crops_info_df.columns = crops_info_df.columns.str.strip()
        all_crops_list = crops_info_df['作物名称'].tolist()
        print("模板列表加载完毕。")
        
        files_to_process = ['result2.xlsx']
        
        for filename in files_to_process:
            input_path = os.path.join(results_dir, filename)
            
            if not os.path.exists(input_path):
                print(f"\n警告：找不到原始结果文件 '{input_path}'，已跳过。")
                continue

            print(f"\n>>> ------------------------------------")
            print(f">>> 正在处理文件: {input_path}")
            original_result_df = pd.read_excel(input_path)
            
            base, ext = os.path.splitext(filename)
            output_path = os.path.join(results_dir, f"{base}_final_summary{ext}")
            
            create_yearly_summary_tables(original_result_df, all_plots_list, all_crops_list, output_path)

    except Exception as e:
        print(f"\n处理过程中发生意外错误: {e}")

    print("\n--- 所有任务已完成 ---")


if __name__ == '__main__':
    main()