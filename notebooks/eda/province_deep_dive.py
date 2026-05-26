import pandas as pd
import numpy as np
import os

def run_province_analysis():
    print("=" * 60)
    print("Starting Province Deep Dive Analysis...")
    
    data_path = 'data/processed/clean_dataset.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)

    df['delivery_gap'] = df['delivery_days'] - df['delivery_time_expected_days']
    df['is_late'] = df['delivery_gap'] > 0
    df['is_cod'] = df['payment_method'] == 'COD'
  
    prov_stats = df.groupby('province').agg(
        total_orders=('order_id', 'count'),
        return_rate=('is_returned', lambda x: x.mean() * 100),
        avg_delivery_days=('delivery_days', 'mean'),
        avg_delivery_gap=('delivery_gap', 'mean'),
        late_delivery_rate=('is_late', lambda x: x.mean() * 100),
        cod_usage_rate=('is_cod', lambda x: x.mean() * 100),
        top_courier=('courier_name', lambda x: x.mode()[0] if not x.empty else 'N/A'),
        avg_courier_damage_rate=('damage_rate', lambda x: x.mean() * 100)
    ).round(2)
    
    prov_stats = prov_stats.sort_values('return_rate', ascending=False)
    
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/province_deep_dive.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Province Deep Dive Analysis (เจาะลึกที่มาของแต่ละพื้นที่)\n\n")
        f.write("ตารางด้านล่างนี้แสดงข้อมูลเชิงลึก (ไส้ใน) ของแต่ละจังหวัด เพื่อให้เห็นสาเหตุว่าทำไมบางพื้นที่ถึงมีการคืนของสูง\n\n")
        f.write(prov_stats.to_string())
        f.write("\n\n### ข้อสังเกตที่ได้จากข้อมูล (Insights):\n")
        f.write("1. **เวลาจัดส่ง (Delivery Days / Gap):** จังหวัดที่มี Return Rate สูง มักจะมีค่า `avg_delivery_gap` (ส่งช้ากว่ากำหนด) ที่สูงตามไปด้วย\n")
        f.write("2. **รูปแบบการชำระเงิน (COD Usage):** หากพื้นที่ไหนมีสัดส่วนการใช้ COD (เก็บเงินปลายทาง) สูงร่วมกับการส่งช้า จะเกิดแรงบวกทำให้การปฏิเสธรับสินค้าพุ่งสูงขึ้น\n")
        f.write("3. **ขนส่ง (Top Courier):** ขนส่งที่รับผิดชอบในพื้นที่นั้นๆ อาจมี `damage_rate` สูง ทำให้กล่องพัสดุเสียหายระหว่างทางที่ไกล\n")
        
    print(f"Analysis complete! Report saved to {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_province_analysis()
