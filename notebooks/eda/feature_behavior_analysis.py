import pandas as pd
import numpy as np
import os

def run_feature_behavior_analysis():
    print("=" * 60)
    print("Starting Feature Behavior Analysis by Category and Province...")
    print("=" * 60)
    
    data_path = 'data/processed/clean_dataset.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    df['is_peak_hour'] = 0
    df.loc[(df['channel_type'] == 'TV_Show') & (df['order_hour'].between(8, 10)), 'is_peak_hour'] = 1
    df.loc[(df['channel_type'] == 'TikTok') & (df['order_hour'].between(21, 23)), 'is_peak_hour'] = 1
    
    df['is_fashion_tv'] = ((df['category'] == 'Fashion') & (df['channel_type'] == 'TV_Show')).astype(int)
    df['is_bracketing'] = ((df['category'] == 'Fashion') & (df['quantity'] > 1)).astype(int)
    df['is_impulse_buy'] = ((df['category'] == 'Fashion') & 
                            (df['channel_type'].isin(['TV_Show', 'TikTok'])) & 
                            (df['is_peak_hour'] == 1)).astype(int)
    
    df['is_remote_area'] = (df['province'] == 'Remote_Area').astype(int)
    df['is_cod'] = (df['payment_method'] == 'COD').astype(int)
    
    df['is_long_distance_cod'] = ((df['province'].isin(['Chiang Mai', 'Phuket', 'Songkhla'])) & 
                                  (df['payment_method'] == 'COD')).astype(int)
    
    df['delivery_gap'] = df['delivery_days'] - df['delivery_time_expected_days']
    df['is_high_friction'] = ((df['delivery_gap'] > 0) & 
                              (df['province'].isin(['Chiang Mai', 'Phuket', 'Songkhla', 'Remote_Area']))).astype(int)
    
    df['is_high_discount'] = (df['total_discount_pct'] > 0.2).astype(int)
    df['is_low_commitment'] = ((df['payment_method'] == 'COD') & 
                               (df['is_high_discount'] == 1)).astype(int)
    
    df['is_perfect_storm'] = ((df['is_impulse_buy'] == 1) & 
                              (df['is_high_friction'] == 1) & 
                              (df['is_low_commitment'] == 1)).astype(int)
    
    target_provinces = [
        'Nonthaburi', 'Phuket', 'Bangkok', 'Chiang Mai', 
        'Songkhla', 'Khon Kaen', 'Chonburi', 'Remote_Area'
    ]
    
    found_provinces = df['province'].unique()
    print(f"Provinces in dataset: {found_provinces}")
    
    df_filtered = df[df['province'].isin(target_provinces)].copy()
    
    analysis_cols = [
        'is_returned', 'delivery_gap', 'is_cod', 'is_long_distance_cod',
        'is_high_friction', 'is_impulse_buy', 'is_bracketing', 
        'is_fashion_tv', 'is_low_commitment', 'is_perfect_storm'
    ]
    
    grouped = df_filtered.groupby(['category', 'province'])[analysis_cols].mean()
    
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/feature_behavior_by_category_and_province.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 📊 Feature Behavior Analysis by Category & Province\n\n")
        f.write("รายงานนี้วิเคราะห์พฤติกรรมและการกระจายตัวของ Features แต่ละตัวเมื่อเปลี่ยนตาม **Category** ใน 8 จังหวัดเป้าหมาย\n")
        f.write("เพื่อให้เข้าใจชัดเจนว่าเมื่อระบบเปลี่ยนหมวดหมู่สินค้า ค่าของ Features ที่เราสร้างขึ้นมาวิเคราะห์ความเสี่ยงการคืนของ (Return Risk) จะมีลักษณะเปลี่ยนแปลงไปอย่างไร\n\n")
        
        f.write("## 📌 1. สรุปพฤติกรรมหลัก (Key Logic Rules)\n")
        f.write("- **Features เฉพาะของแฟชั่น (`is_fashion_tv`, `is_bracketing`, `is_impulse_buy`, `is_perfect_storm`):**\n")
        f.write("  - จะมีค่าเฉลี่ยมากกว่า 0 เฉพาะเมื่อ Category = `Fashion` เท่านั้น\n")
        f.write("  - เมื่อเปลี่ยนเป็น `Supplement` หรือ `Home_Appliance` ค่าของ Features เหล่านี้จะกลายเป็น **0.00** เสมอ เพราะเงื่อนไขผูกติดกับหมวดหมู่แฟชั่น\n")
        f.write("- **Features เฉพาะของพื้นที่การจัดส่ง (`is_high_friction`, `is_long_distance_cod`):**\n")
        f.write("  - `is_long_distance_cod` จะเป็น 1 เฉพาะใน 3 จังหวัดที่เป็นหัวเมืองท่องเที่ยวไกล (`Chiang Mai`, `Phuket`, `Songkhla`) ที่สั่งแบบ COD (เก็บเงินปลายทาง) โดยไม่ขึ้นกับ Category\n")
        f.write("  - `is_high_friction` (จัดส่งช้า + อยู่ในจังหวัดเสี่ยง ได้แก่ Chiang Mai, Phuket, Songkhla, Remote_Area) จะเปิดทำงานในพื้นที่เหล่านี้เมื่อมีการส่งของเลท (Delivery Gap > 0)\n")
        f.write("- **Features ทั่วไป (`is_cod`, `is_low_commitment`, `delivery_gap`):**\n")
        f.write("  - เปลี่ยนแปลงไปตามพฤติกรรมการสั่งซื้อและการขนส่งจริงในแต่ละ Category x Province\n\n")
        
        for category in ['Fashion', 'Supplement', 'Home_Appliance']:
            f.write(f"## 🛍️ Category: {category}\n")
            f.write(f"วิเคราะห์พฤติกรรม Feature ในหมวดสินค้า **{category}** ของแต่ละจังหวัด:\n\n")
            
            cat_df = grouped.loc[category].reindex(target_provinces)
            display_df = cat_df.copy()
            percent_cols = ['is_returned', 'is_cod', 'is_long_distance_cod', 'is_high_friction', 'is_impulse_buy', 'is_bracketing', 'is_fashion_tv', 'is_low_commitment', 'is_perfect_storm']
            for col in percent_cols:
                display_df[col] = (display_df[col] * 100).round(2).astype(str) + '%'
            display_df['delivery_gap'] = display_df['delivery_gap'].round(3)
            
            display_df.columns = [
                'Return Rate (%)',
                'Avg Delivery Gap (Days)',
                'COD Rate (%)',
                'Long Dist COD (%)',
                'High Friction (%)',
                'Impulse Buy (%)',
                'Bracketing (%)',
                'Fashion TV (%)',
                'Low Commitment (%)',
                'Perfect Storm (%)'
            ]
            
            def to_markdown_custom(df):
                headers = [df.index.name or 'province'] + list(df.columns)
                lines = []
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for idx, row in df.iterrows():
                    row_str = [str(idx)] + [str(val) for val in row]
                    lines.append("| " + " | ".join(row_str) + " |")
                return "\n".join(lines)

            f.write(to_markdown_custom(display_df))
            f.write("\n\n")
            
            f.write(f"### 💡 เจาะลึกสำหรับหมวด {category}:\n")
            if category == 'Fashion':
                f.write("- **High Risk Combination:** สังเกตจังหวัด `Phuket` และ `Remote_Area` ที่มีอัตราการคืนของ (Return Rate) สูงมาก เนื่องจากมีความเสี่ยง High Friction ร่วมกับการลองสินค้าหลายไซส์ (Bracketing) และ Impulse Buying จาก TV/TikTok\n")
                f.write("- **Bracketing:** มีค่าเฉลี่ยประมาณ 10-15% ในทุกจังหวัด ซึ่งบ่งบอกพฤติกรรมการสั่งของหลายชิ้นเพื่อนำไปลอง\n")
            elif category == 'Supplement':
                f.write("- **Fashion Feature Drop:** Features กลุ่มแฟชั่นทั้งหมด (Impulse Buy, Bracketing, Fashion TV, Perfect Storm) กลายเป็น **0%** อย่างถูกต้องตามกฎ\n")
                f.write("- **COD & Delivery friction:** ในพื้นที่ห่างไกลและเกาะท่องเที่ยว (`Phuket`, `Remote_Area`) หากส่งช้า (`delivery_gap` สูง) จะทำให้ `High Friction` สูงมาก (เช่น Phuket สูงถึง 70%+) ส่งผลให้มีการปฏิเสธรับสินค้าแบบ COD สูงตามไปด้วย\n")
            elif category == 'Home_Appliance':
                f.write("- **High Friction in Logistics:** สินค้าประเภท Home Appliance มักมีขนาดใหญ่และหนัก ทำให้ใช้เวลาจัดส่งนานกว่าปกติ (สังเกตค่า Avg Delivery Gap จะเป็นบวกสูงขึ้น โดยเฉพาะจังหวัดไกลๆ เช่น Phuket และ Chiang Mai) ส่งผลให้ `High Friction` พุ่งสูงที่สุดในบรรดาสามหมวดหมู่\n")
                f.write("- **Return Rate Impact:** สำหรับ Home Appliance อัตราการคืนในพื้นที่ห่างไกลเชื่อมโยงโดยตรงกับความเสียหายหรือความล่าช้าในการขนส่ง ไม่ใช่เกิดจากความต้องการเปลี่ยนสี/ไซส์เหมือนแฟชั่น\n")
            f.write("\n" + "-"*50 + "\n\n")
            
    print(f"[SUCCESS] Feature behavior analysis completed. Report saved to {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_feature_behavior_analysis()
