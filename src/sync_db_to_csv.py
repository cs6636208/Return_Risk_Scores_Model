import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

def main():
    print("=" * 60)
    print("🔄 RETURN RISK DATA SYNC: PostgreSQL ➔ clean_dataset.csv")
    print("=" * 60)
    
    load_dotenv()
    
    try:
        db_user = os.getenv('DB_USER')
        db_pass = os.getenv('DB_PASS')
        db_host = os.getenv('DB_HOST')
        db_port = os.getenv('DB_PORT')
        db_name = os.getenv('DB_NAME')
        
        engine_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(engine_url)
        
        df = pd.read_sql("SELECT * FROM order_history_complete_v2", engine)
        
        if 'province' in df.columns:
            nonthaburi_count = len(df[df['province'] == 'Nonthaburi'])
            print(f"📊 ตรวจพบข้อมูลจังหวัด Nonthaburi ในฐานข้อมูล = {nonthaburi_count} ออเดอร์")
        
        output_path = 'data/processed/clean_dataset.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"💾 อัปเดตข้อมูลและเขียนทับลงไฟล์ '{output_path}' เรียบร้อยแล้ว!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อฐานข้อมูล: {e}")
        print("กรุณาตรวจสอบว่า PostgreSQL เปิดใช้งานอยู่และค่าใน .env ถูกต้อง")
        print("=" * 60)

if __name__ == '__main__':
    main()
