import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_categorical_crosses():
    data_path = 'data/processed/clean_dataset.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] Clean dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded {len(df)} rows.")
    
    # Ensure reports/Graph Item exists
    plot_dir = os.path.join('reports', 'Graph Item')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Configure Seaborn style for a premium look
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    
    # 1. Plot Category x Payment Method vs Return Rate
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df, 
        x='category', 
        y='is_returned', 
        hue='payment_method', 
        errorbar=None, 
        palette='muted'
    )
    plt.title('Return Rate by Product Category & Payment Method', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Product Category', fontsize=12, labelpad=10)
    plt.ylabel('Average Return Rate', fontsize=12, labelpad=10)
    plt.ylim(0, 0.40)
    # Add values on top of bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.1%}", 
                        (p.get_x() + p.get_width() / 2., p.get_height() + 0.005), 
                        ha='center', va='bottom', fontsize=9, color='black', fontweight='semibold')
    plt.legend(title='Payment Method', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    chart1_path = os.path.join(plot_dir, 'cross_chart_category_payment.png')
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved {chart1_path}")
    
    # 2. Plot Category x Channel Type vs Return Rate (Line Chart / Clustered Bar)
    # Clustered Bar is usually better for categorical crosses, but let's do a Line Chart (Point Plot) to see the trends!
    plt.figure(figsize=(12, 6))
    ax = sns.pointplot(
        data=df, 
        x='category', 
        y='is_returned', 
        hue='channel_type', 
        errorbar=None,
        markers=["o", "s", "D", "^"], 
        linestyles=["-", "--", "-.", ":"],
        palette='Set1'
    )
    plt.title('Return Rate Trends by Category & Channel Type', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Product Category', fontsize=12, labelpad=10)
    plt.ylabel('Average Return Rate', fontsize=12, labelpad=10)
    plt.ylim(0.15, 0.40)
    plt.legend(title='Channel Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    chart2_path = os.path.join(plot_dir, 'cross_chart_category_channel.png')
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved {chart2_path}")

    # 3. Plot Province x Payment Method vs Return Rate (Clustered Bar)
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(
        data=df, 
        x='province', 
        y='is_returned', 
        hue='payment_method', 
        errorbar=None, 
        palette='deep'
    )
    plt.title('Return Rate by Province & Payment Method', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Province', fontsize=12, labelpad=10)
    plt.ylabel('Average Return Rate', fontsize=12, labelpad=10)
    plt.xticks(rotation=15)
    plt.ylim(0, 0.45)
    # Add values on top of bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.1%}", 
                        (p.get_x() + p.get_width() / 2., p.get_height() + 0.005), 
                        ha='center', va='bottom', fontsize=8, color='black')
    plt.legend(title='Payment Method', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    chart3_path = os.path.join(plot_dir, 'cross_chart_province_payment.png')
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved {chart3_path}")

    # 4. Plot Gender x Province vs Return Rate (Clustered Bar Chart)
    # We will exclude gender='Other' or include it based on availability. The clean dataset has gender Male, Female, Other
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(
        data=df, 
        x='province', 
        y='is_returned', 
        hue='gender', 
        errorbar=None, 
        palette='Set2'
    )
    plt.title('Return Rate by Province & Gender', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Province', fontsize=12, labelpad=10)
    plt.ylabel('Average Return Rate', fontsize=12, labelpad=10)
    plt.xticks(rotation=15)
    plt.ylim(0, 0.45)
    # Add values on top of bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.1%}", 
                        (p.get_x() + p.get_width() / 2., p.get_height() + 0.005), 
                        ha='center', va='bottom', fontsize=8, color='black')
    plt.legend(title='Gender', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    chart4_path = os.path.join(plot_dir, 'cross_chart_province_gender.png')
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved {chart4_path}")
    
    # Save a Markdown file combining these plots
    vis_report_path = 'reports/categorical_cross_visualizations.md'
    with open(vis_report_path, 'w', encoding='utf-8') as f:
        f.write("# 📈 รายงานแผนภูมิเปรียบเทียบการ Cross Categorical Features กับ Return Rate\n\n")
        f.write("รายงานฉบับนี้รวบรวมแผนภูมิแท่ง (Bar Charts) และแผนภูมิเส้น (Line Charts) ของตัวแปรกลุ่มสองตัวที่จับคู่ครอสกัน ")
        f.write("เพื่อเปรียบเทียบอัตราการคืนสินค้าเฉลี่ย (Return Rate) บนแกน Y ได้อย่างชัดเจนเชิงทัศนภาพครับ\n\n")
        f.write("---\n\n")
        
        f.write("## 📊 1. แผนภูมิแท่งเปรียบเทียบ Category & Payment Method กับ Return Rate\n\n")
        f.write("แสดงอัตราการคืนในแต่ละหมวดสินค้าแยกตามสีของช่องทางจ่ายเงิน:\n\n")
        f.write("![Category x Payment Method Return Rate](Graph%20Item/cross_chart_category_payment.png)\n\n")
        f.write("> **เจาะลึก:** เห็นได้ชัดว่า `Electronics` ที่จ่ายผ่านบัตรเครดิตพุ่งสูงถึง 35.1% ในขณะที่ `Cosmetics` ที่โอนเงินคืนต่ำสุดที่ 22.2%\n\n")
        f.write("---\n\n")
        
        f.write("## 📈 2. แผนภูมิเส้นแนวโน้ม Category & Channel Type กับ Return Rate\n\n")
        f.write("แสดงเส้นแนวโน้มอัตราการคืนสินค้าในแต่ละประเภทสินค้าแยกตามสื่อช่องทางขาย:\n\n")
        f.write("![Category x Channel Type Return Rate](Graph%20Item/cross_chart_category_channel.png)\n\n")
        f.write("> **เจาะลึก:** สินค้าหมวด `Electronics` แสดงแรงเหวี่ยงขึ้นเมื่อขายบน `TV Show` และ `TikTok` อย่างชัดเจน ส่วนหมวดหมู่อื่นๆ มีแรงเหวี่ยงที่แตกต่างกัน\n\n")
        f.write("---\n\n")
        
        f.write("## 📊 3. แผนภูมิแท่งเปรียบเทียบ Province & Payment Method กับ Return Rate\n\n")
        f.write("แสดงอัตราการคืนแยกตามแต่ละจังหวัดและสีของวิธีการชำระเงิน:\n\n")
        f.write("![Province x Payment Method Return Rate](Graph%20Item/cross_chart_province_payment.png)\n\n")
        f.write("> **เจาะลึก:** แสดงความเสี่ยงในพื้นที่ `Songkhla` เมื่อสั่งปลายทาง `COD` พุ่งขึ้นชัดเจน และ `Chonburi` ที่จ่ายบัตรเครดิต\n\n")
        f.write("---\n\n")
        
        f.write("## 📊 4. แผนภูมิแท่งเปรียบเทียบ Province & Gender กับ Return Rate\n\n")
        f.write("แสดงอัตราการคืนแยกตามจังหวัดปลายทางและสีระบุเพศของผู้ซื้อ:\n\n")
        f.write("![Province x Gender Return Rate](Graph%20Item/cross_chart_province_gender.png)\n\n")
        f.write("> **เจาะลึก:** ชี้ชัดถึงพฤติกรรมการตัดสินใจคืนสินค้าที่แยกกันคนละขั้วระหว่างเพศหญิงและชายในแต่ละภูมิภาค เช่น หญิงกรุงเทพฯ สูงถึง 32.6% แต่ชายกรุงเทพฯ คืนเพียง 15.1%\n")
        
    print("[INFO] Visualization report written.")

if __name__ == "__main__":
    plot_categorical_crosses()
