import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def run_electronics_cosmetics_analysis():
    print("=" * 60)
    print("Starting Electronics and Cosmetics Analysis by Province...")
    print("=" * 60)
    
    data_path = 'data/processed/clean_dataset.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    target_provinces = [
        'Bangkok', 'Nonthaburi', 'Chonburi', 'Khon Kaen',
        'Chiang Mai', 'Phuket', 'Songkhla', 'Remote_Area'
    ]
    target_categories = ['Electronics', 'Cosmetics']
    
    df_filtered = df[
        df['province'].isin(target_provinces) & 
        df['category'].isin(target_categories)
    ].copy()
    
    agg_df = df_filtered.groupby(['province', 'category'])['is_returned'].mean().reset_index()
    agg_df['is_returned'] *= 100  
    
    print("\n[INFO] Return Rates (%) by Category and Province:")
    pivot_df = agg_df.pivot(index='province', columns='category', values='is_returned').reindex(target_provinces)
    print(pivot_df.round(2).to_string())
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    palette = {"Electronics": "#1A5276", "Cosmetics": "#E74C3C"}
    
    ax = sns.barplot(
        data=agg_df, 
        x='province', 
        y='is_returned', 
        hue='category', 
        palette=palette,
        order=target_provinces,
        edgecolor='black',
        linewidth=0.8
    )
    
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(
                f'{height:.1f}%',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center',
                xytext=(0, 8),
                textcoords='offset points',
                fontsize=10,
                fontweight='semibold'
            )
            
    plt.title("Return Rate (%) by Category and Province (Electronics vs Cosmetics)", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Province", fontsize=12, labelpad=10)
    plt.ylabel("Return Rate (%)", fontsize=12, labelpad=10)
    plt.ylim(0, max(agg_df['is_returned']) + 5)
    plt.legend(title='Category', title_fontsize='11', fontsize='10', loc='upper right')
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    output_dir = 'reports/Graph Item'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'eda_electronics_cosmetics_province.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    report_path = 'reports/electronics_cosmetics_analysis.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 📊 Return Rate Analysis: Electronics vs Cosmetics by Province\n\n")
        f.write("รายงานเปรียบเทียบอัตราการคืนสินค้าในหมวด **Electronics** และ **Cosmetics** แยกตาม 8 จังหวัดเป้าหมาย\n\n")
        f.write("## 📈 ตารางแสดงอัตราการคืนสินค้า (Return Rate %)\n\n")
        
        f.write("| จังหวัด (Province) | Electronics (%) | Cosmetics (%) |\n")
        f.write("| --- | --- | --- |\n")
        for prov in target_provinces:
            elec_val = pivot_df.loc[prov, 'Electronics']
            cosm_val = pivot_df.loc[prov, 'Cosmetics']
            f.write(f"| {prov} | {elec_val:.2f}% | {cosm_val:.2f}% |\n")
            
        f.write("\n## 💡 Insights เจาะลึก:\n")
        f.write("1. **หมวด Electronics (เครื่องใช้ไฟฟ้า/เทคโนโลยี):**\n")
        f.write("   - มีอัตราการคืนสินค้าที่แปรผันตามระยะทางและการจัดส่ง โดยพื้นที่ไกลอย่าง **Remote_Area** และ **Songkhla** มีความเสี่ยงการคืนของสูงกว่า\n")
        f.write("2. **หมวด Cosmetics (เครื่องสำอาง/ความงาม):**\n")
        f.write("   - อัตราการคืนสินค้าค่อนข้างมีความผันผวนเฉพาะตัว เนื่องจากเกี่ยวกับความพึงพอใจ สี หรือการชำรุดเสียหายระหว่างขนส่งได้ง่าย (เช่น กล่องแตก ผลิตภัณฑ์หกเลอะเทอะ)\n")
        
    print(f"\n[SUCCESS] Graph saved to: {output_path}")
    print(f"[SUCCESS] Report saved to: {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_electronics_cosmetics_analysis()
