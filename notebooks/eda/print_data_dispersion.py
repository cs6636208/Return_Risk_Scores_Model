import pandas as pd
import numpy as np
import os

def run_data_dispersion_analysis():
    print("=" * 60)
    print("Starting Numerical Data Dispersion Analysis...")
    print("=" * 60)
    
    data_path = 'data/processed/clean_dataset.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    # 1. Select key numerical columns
    numerical_cols = [
        'customer_age_days',
        'unit_price',
        'quantity',
        'total_discount_pct',
        'total_amount',
        'product_rating',
        'delivery_days',
        'delivery_time_expected_days',
        'hist_order_count',
        'hist_return_rate',
        'days_since_last_order'
    ]
    
    # Ensure they exist in df
    cols_to_use = [c for c in numerical_cols if c in df.columns]
    
    # 2. Compute pandas descriptive statistics
    desc = df[cols_to_use].describe().T
    
    # 3. Add additional dispersion metrics
    desc['variance'] = df[cols_to_use].var()
    desc['iqr'] = desc['75%'] - desc['25%']
    desc['skewness'] = df[cols_to_use].skew()
    desc['kurtosis'] = df[cols_to_use].kurtosis()
    
    # Reorder columns for a clean presentation
    desc_clean = desc[[
        'count', 'mean', '50%', 'std', 'variance', 'iqr', 
        'min', '25%', '75%', 'max', 'skewness', 'kurtosis'
    ]]
    
    # Rename columns for clarity
    desc_clean.columns = [
        'Count', 'Mean (Avg)', 'Median (50%)', 'Std Dev (SD)', 
        'Variance', 'IQR', 'Min', '25% (Q1)', '75% (Q3)', 
        'Max', 'Skewness', 'Kurtosis'
    ]
    
    # Save a detailed report
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/numerical_distribution_summary.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 📊 Statistical Summary & Data Dispersion Metrics\n\n")
        f.write("รายงานนี้แสดงตัวเลขค่าสถิติเชิงพรรณนา (Descriptive Statistics) และค่าการกระจายตัวของข้อมูล (Dispersion Metrics) ")
        f.write("ของตัวแปรตัวเลขที่สำคัญในฐานข้อมูล เพื่อใช้ประกอบการวิเคราะห์ลักษณะทางคณิตศาสตร์ของฟีเจอร์ต่างๆ\n\n")
        
        f.write("## 📌 ตารางสรุปค่าสถิติเชิงเลข\n\n")
        
        # Format markdown table
        # We will format values to 3 decimal places
        formatted_df = desc_clean.round(3)
        
        # Build markdown table manually to ensure clean formatting
        headers = ["Feature"] + list(formatted_df.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for idx, row in formatted_df.iterrows():
            row_values = [str(idx)] + [str(val) for val in row]
            f.write("| " + " | ".join(row_values) + " |\n")
            
        f.write("\n\n## 💡 คำอธิบายความหมายของค่าสถิติแต่ละหมวดหมู่ (Statistical Glossary):\n")
        f.write("1. **Mean (ค่าเฉลี่ย) vs Median (ค่ามัธยฐาน / 50%):**\n")
        f.write("   - ช่วยดูศูนย์กลางของข้อมูล หากค่า Mean ต่างจาก Median มาก แสดงว่าข้อมูลมีแนวโน้มถูกเบ้จาก Outliers (ค่าสุดโต่ง)\n")
        f.write("2. **Std Dev (ส่วนเบี่ยงเบนมาตรฐาน - SD) & Variance (ความแปรปรวน):**\n")
        f.write("   - บ่งบอกระดับความกระจัดกระจายของข้อมูล ยิ่งค่ายิ่งสูงแสดงว่าข้อมูลมีการเกาะกลุ่มกันน้อยและกระจายตัวกว้างมาก\n")
        f.write("3. **IQR (Interquartile Range - ช่วงระหว่างควอไทล์):**\n")
        f.write("   - คือระยะห่างระหว่างจุด 25% (Q1) และ 75% (Q3) แสดงถึงขนาดการกระจายตัวของข้อมูลกึ่งกลาง 50% ของชุดข้อมูล (นิยมใช้ตรวจจับ Outliers ร่วมกับ Box Plots)\n")
        f.write("4. **Skewness (ความเบ้):**\n")
        f.write("   - **Skewness > 0 (เบ้ขวา):** ข้อมูลส่วนใหญ่กองอยู่ฝั่งซ้ายและมีหางยาวยื่นไปทางขวา (เช่น `unit_price`, `total_amount`)\n")
        f.write("   - **Skewness < 0 (เบ้อซ้าย):** ข้อมูลส่วนใหญ่กองอยู่ฝั่งขวาและมีหางยาวยื่นไปทางซ้าย (เช่น `product_rating`)\n")
        f.write("   - **Skewness ≈ 0 (สมมาตร):** มีรูปร่างสมมาตรใกล้เคียงระฆังคว่ำปกติ (เช่น `customer_age_days`)\n")
        f.write("5. **Kurtosis (ความโด่ง):**\n")
        f.write("   - วัดระดับความแหลม/โด่งของยอดและการมีหางหนา (Heavy-tails) ของข้อมูล หากค่า Kurtosis สูงมาก แสดงว่าข้อมูลมีโอกาสเกิดค่าสุดโต่ง (Extreme Outliers) ได้สูง\n")
        
    print("\n[SUCCESS] Completed Numerical Data Dispersion Analysis:")
    # Print clean terminal output
    print(desc_clean.round(3).to_string())
    print(f"\n[SUCCESS] Statistical summary saved to: {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_data_dispersion_analysis()
