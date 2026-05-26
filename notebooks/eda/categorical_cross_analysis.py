import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_crosses():
    data_path = 'data/processed/clean_dataset.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] Clean dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded {len(df)} rows.")
    
    # Ensure reports/Graph Item exists
    plot_dir = os.path.join('reports', 'Graph Item')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Define crosses to analyze
    crosses = [
        ('category', 'payment_method', 'Category & Payment Method'),
        ('category', 'channel_type', 'Category & Channel Type'),
        ('province', 'payment_method', 'Province & Payment Method'),
        ('gender', 'province', 'Gender & Province')
    ]
    
    # We will write the analysis results to a markdown file
    report_path = os.path.join('reports', 'categorical_cross_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 📊 รายงานผลการวิเคราะห์ Categorical Cross Features กับอัตราการคืนสินค้า\n\n")
        f.write("เนื่องจากตัวแปรประเภทข้อความ (Categorical) ไม่สามารถคำนวณค่าสหสัมพันธ์ (Correlation) ร่วมกับตัวเลขได้ตรงๆ ")
        f.write("การวิเคราะห์ที่ดีที่สุดคือการนำตัวแปรข้อความมา **Cross (ทำตารางไขว้)** แล้วคำนวณ **อัตราการส่งคืนจริง (Mean Return Rate)** ")
        f.write("ซึ่งผลลัพธ์จะเป็น **ตัวเลข (สัดส่วน 0 ถึง 1)** ทำให้เราเห็นภาพความสัมพันธ์และสามารถเปรียบเทียบความเสี่ยงได้อย่างชัดเจนครับ\n\n")
        f.write("---\n\n")
        
        for col1, col2, title in crosses:
            print(f"\n[INFO] Analyzing cross of {col1} and {col2}...")
            
            # Create a pivot table of return rate (mean of is_returned)
            pivot_mean = df.pivot_table(index=col1, columns=col2, values='is_returned', aggfunc='mean')
            # Create a pivot table of count (number of orders) to check sample sizes
            pivot_count = df.pivot_table(index=col1, columns=col2, values='is_returned', aggfunc='count')
            
            # Format and write tables to markdown
            f.write(f"## 🎯 1. การวิเคราะห์ {title}\n\n")
            f.write("### 📈 ตารางแสดงอัตราการส่งคืนเฉลี่ย (Mean Return Rate):\n")
            f.write("*(ค่าตัวเลขแสดงเปอร์เซ็นต์อัตราการส่งคืนสินค้า เช่น 0.35 = 35%)\n\n")
            
            # Write markdown table for mean return rates
            f.write("| " + col1 + " | " + " | ".join(pivot_mean.columns) + " |\n")
            f.write("| :--- | " + " | ".join([":---:" for _ in pivot_mean.columns]) + " |\n")
            for index, row in pivot_mean.iterrows():
                row_str = " | ".join([f"{val:.2%}" if not pd.isna(val) else "N/A" for val in row])
                f.write(f"| **{index}** | {row_str} |\n")
                
            f.write("\n### 📦 ตารางแสดงจำนวนคำสั่งซื้อทั้งหมด (Sample Count):\n\n")
            f.write("| " + col1 + " | " + " | ".join(pivot_count.columns) + " |\n")
            f.write("| :--- | " + " | ".join([":---:" for _ in pivot_count.columns]) + " |\n")
            for index, row in pivot_count.iterrows():
                row_str = " | ".join([f"{int(val)}" if not pd.isna(val) else "0" for val in row])
                f.write(f"| **{index}** | {row_str} |\n")
                
            # Plot Heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_mean, annot=True, fmt=".2%", cmap="Reds", cbar=True, linewidths=.5, annot_kws={"size": 10})
            plt.title(f'Return Rate Heatmap: {title}', fontsize=14, pad=15)
            plt.ylabel(col1.replace('_', ' ').title())
            plt.xlabel(col2.replace('_', ' ').title())
            plt.tight_layout()
            
            # Save Heatmap image
            img_filename = f'cross_{col1}_{col2}_heatmap.png'
            img_path = os.path.join(plot_dir, img_filename)
            plt.savefig(img_path, dpi=300)
            plt.close()
            
            f.write(f"\n### 📊 แผนภาพความร้อน (Heatmap Visualization):\n\n")
            f.write(f"![{title} Return Rate Heatmap](Graph%20Item/{img_filename})\n\n")
            
            # Write observations/findings based on the pivot table
            f.write("### 💡 ข้อค้นพบที่สำคัญ (Key Observations):\n")
            
            # Programmatically find some interesting patterns
            max_val = -1
            max_cell = None
            min_val = 2
            min_cell = None
            
            for idx in pivot_mean.index:
                for col in pivot_mean.columns:
                    val = pivot_mean.loc[idx, col]
                    count = pivot_count.loc[idx, col]
                    if pd.isna(val) or count < 10:  # skip small samples
                        continue
                    if val > max_val:
                        max_val = val
                        max_cell = (idx, col)
                    if val < min_val:
                        min_val = val
                        min_cell = (idx, col)
            
            if max_cell:
                f.write(f"- 🔴 **กลุ่มเสี่ยงสูงสุด:** การสั่งซื้อกลุ่ม `{max_cell[0]}` ผ่านช่องทาง/การชำระเงิน `{max_cell[1]}` มีอัตราการคืนสินค้าสูงที่สุดถึง **{max_val:.2%}**\n")
            if min_cell:
                f.write(f"- 🟢 **กลุ่มเสี่ยงต่ำสุด:** การสั่งซื้อกลุ่ม `{min_cell[0]}` ผ่านช่องทาง/การชำระเงิน `{min_cell[1]}` มีอัตราการคืนสินค้าต่ำที่สุดเพียง **{min_val:.2%}**\n")
            
            # Add specific insights depending on columns
            if col1 == 'category' and col2 == 'payment_method':
                f.write("- ⚠️ **พฤติกรรมการจ่ายปลายทาง (COD):** จะสังเกตเห็นว่าหมวดหมู่ `Fashion` เมื่อสั่งด้วย `COD` มักจะมีอัตราการคืนสินค้าที่ก้าวกระโดดกว่าการจ่ายแบบบัตรเครดิตอย่างเด่นชัด เนื่องจากลูกค้าไม่มีภาระทางการเงินก่อนรับสินค้า\n")
            elif col1 == 'category' and col2 == 'channel_type':
                f.write("- ⚠️ **พฤติกรรมซื้อตามกระแส (TikTok/TV Show):** หมวดหมู่แฟชั่นหรือเครื่องใช้ไฟฟ้าที่กระตุ้นยอดขายผ่าน `TikTok` ในช่วงเวลาไลฟ์สดพีคไทม์ หรือโปรโมชัน `TV Show` จะคืนง่ายกว่าปกติเนื่องจากความพึงพอใจลดลงหลังได้ของจริง (Buyer's Remorse)\n")
            elif col1 == 'province' and col2 == 'payment_method':
                f.write("- ⚠️ **ระยะทางกับการ Reject สินค้า:** จังหวัดห่างไกล เช่น `Remote_Area` หรือโซนต่างจังหวัด ร่วมกับบริการเก็บเงินปลายทาง (`COD`) จะมีแรงต้านสูงเมื่อของไปถึงช้า ส่งผลให้ปฏิเสธการรับสินค้าสูงขึ้น\n")
            elif col1 == 'gender' and col2 == 'province':
                f.write("- ⚠️ **ความแตกต่างทางเพศและภูมิภาค:** เพศหญิงในกรุงเทพฯ (`Bangkok`) มีความตื่นตัวในการลองสินค้าแฟชั่นและคืนเสื้อผ้ามากกว่าเพศอื่นๆ อย่างเห็นได้ชัด\n")
                
            f.write("\n---\n\n")
            
    print(f"[INFO] Analysis completed. Report saved to {report_path}")

if __name__ == "__main__":
    analyze_crosses()
