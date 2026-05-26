import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def run_gender_province_analysis():
    print("=" * 60)
    print("Starting Gender vs Province Return Rate Analysis...")
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
    
    df_filtered = df[df['province'].isin(target_provinces)].copy()

    print("Unique genders in dataset:", df_filtered['gender'].unique())
    
    agg_df = df_filtered.groupby(['province', 'gender'])['is_returned'].mean().reset_index()
    agg_df['is_returned'] *= 100  
    
    overall_gender = df_filtered.groupby('gender')['is_returned'].mean().reset_index()
    overall_gender['is_returned'] *= 100
    print("\n[INFO] Overall Return Rate by Gender:")
    print(overall_gender.to_string(index=False))
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 8))
    
    palette = {
        "Male": "#3498DB",      
        "Female": "#E74C3C",
        "Other": "#9B59B6"      
    }
    
    genders_in_data = [g for g in ["Male", "Female", "Other"] if g in df_filtered['gender'].unique()]
    
    ax = sns.barplot(
        data=agg_df, 
        x='province', 
        y='is_returned', 
        hue='gender', 
        palette=palette,
        order=target_provinces,
        hue_order=genders_in_data,
        edgecolor='black',
        linewidth=0.6
    )
    
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f'{height:.1f}%',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center',
                xytext=(0, 7),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold',
                color='#2C3E50'
            )
            
    plt.title("Return Rate (%) by Gender across Provinces", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Province", fontsize=13, labelpad=12)
    plt.ylabel("Return Rate (%)", fontsize=13, labelpad=12)
    plt.ylim(0, 50) 

    plt.legend(
        title='Customer Gender', 
        title_fontsize='12', 
        fontsize='11', 
        loc='upper right',
        frameon=True,
        facecolor='white',
        edgecolor='gray',
        framealpha=0.9
    )
    
    plt.xticks(rotation=15, fontsize=11, fontweight='semibold')
    plt.yticks(fontsize=11)
    plt.tight_layout()
    
    output_dir = 'reports/Graph Item'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'eda_province_gender_return_rate.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    report_path = 'reports/province_gender_return_rate_summary.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 🚻 Gender & Province Return Rate Interaction Analysis\n\n")
        f.write("รายงานนี้แสดงการวิเคราะห์อัตราการส่งคืนสินค้า (Return Rate) เมื่อแยกตามเพศ (Gender) และจังหวัดปลายทาง (Province) ")
        f.write("เพื่อดูว่าพฤติกรรมการคืนสินค้ามีความแตกต่างทางกายภาพและภูมิประชากรศาสตร์อย่างไร\n\n")
        
        f.write("## 📌 1. ตารางอัตราการส่งคืนแยกตามเพศและจังหวัด (% Return Rate)\n\n")
    
        pivot_df = agg_df.pivot(index='province', columns='gender', values='is_returned')
        pivot_df = pivot_df.reindex(target_provinces)[genders_in_data]
        
        headers = ["Province"] + list(pivot_df.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for idx, row in pivot_df.iterrows():
            row_values = [str(idx)] + [f"{val:.2f}%" if not np.isnan(val) else "N/A" for val in row]
            f.write("| " + " | ".join(row_values) + " |\n")
            
        f.write("\n\n## 📌 2. อัตราการส่งคืนภาพรวมแยกตามเพศ (Overall Return Rate by Gender)\n\n")
        f.write("| Gender | Overall Return Rate (%) |\n")
        f.write("| --- | --- |\n")
        for _, row in overall_gender.iterrows():
            f.write(f"| {row['gender']} | {row['is_returned']:.2f}% |\n")
            
        f.write("\n\n## 💡 ข้อค้นพบสำคัญ (Key Takeaways):\n")
        f.write("1. **ความต่างระหว่างเพศภาพรวม:** ตรวจสอบความแตกต่างระหว่างอัตราการส่งคืนของ Male vs Female vs Other\n")
        f.write("2. **ความสัมพันธ์รายพื้นที่:** ดูว่ามีพื้นที่ใดบ้างที่เพศเฉพาะเจาะจงมีอัตราการส่งคืนที่สูงหรือต่ำเป็นพิเศษ เพื่อใช้ปรับปรุงกลยุทธ์จัดส่งหรือเสนอโปรโมชั่น\n")
        
    print("\n[INFO] Complete Return Rate (%) Matrix by Gender & Province:")
    print(pivot_df.round(2).to_string())
    print(f"\n[SUCCESS] Gender & Province analysis completed and saved to: {report_path}")
    print(f"[SUCCESS] Visualization saved to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_gender_province_analysis()
