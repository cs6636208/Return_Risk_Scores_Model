# คู่มือเปิด PostgreSQL ด้วย Docker สำหรับส่งต่องาน

เอกสารนี้ใช้สำหรับคนที่รับโปรเจ็กต์ต่อ เพื่อเปิดฐานข้อมูล PostgreSQL และ pgAdmin ด้วย Docker โดยไม่ต้องติดตั้ง PostgreSQL ลงเครื่องเอง

## ไฟล์ที่ต้องมี

```text
05_Environment/
  docker-compose.yml
  .env.example
  requirements.txt
  README_DOCKER_POSTGRES.md
```

ความหมาย:

- `docker-compose.yml` = ตั้งค่า PostgreSQL และ pgAdmin
- `.env.example` = ตัวอย่าง username/password/database ห้ามใส่ password จริง
- `requirements.txt` = Python package สำหรับรัน model/pipeline
- `README_DOCKER_POSTGRES.md` = คู่มือฉบับนี้

## เตรียมเครื่องก่อน

คนรับงานต่อต้องมี:

1. Docker Desktop
2. Git
3. Python 3.10 ขึ้นไป ถ้าจะรัน model ด้วย

## วิธีเปิด PostgreSQL

เข้าโฟลเดอร์นี้:

```powershell
cd "C:\path\to\return-risk-prediction\Model Use\05_Environment"
```

คัดลอกไฟล์ config ตัวอย่าง:

```powershell
copy .env.example .env
```

เปิดไฟล์ `.env` แล้วแก้ password ตามต้องการ เช่น:

```env
DB_HOST=127.0.0.1
DB_PORT=5433
DB_USER=admin
DB_PASS=password123
DB_NAME=gmm_oshopping_db
PGADMIN_EMAIL=admin@example.com
PGADMIN_PASS=admin123
```

เปิด PostgreSQL และ pgAdmin:

```powershell
docker compose up -d
```

ตรวจว่า container ทำงาน:

```powershell
docker ps
```

ควรเห็น:

```text
oshopping_postgres
oshopping_pgadmin
```

## ข้อมูลสำหรับเชื่อมต่อ PostgreSQL

ถ้าเชื่อมจากเครื่องเรา เช่น Python, DBeaver, pgAdmin ที่อยู่นอก Docker:

```text
Host: 127.0.0.1
Port: 5433
Database: gmm_oshopping_db
User: admin
Password: ตามค่า DB_PASS ใน .env
```

ถ้าเชื่อมจาก pgAdmin container ไป PostgreSQL container:

```text
Host: postgres_db
Port: 5432
Database: gmm_oshopping_db
User: admin
Password: ตามค่า DB_PASS ใน .env
```

## วิธีเปิด pgAdmin

เปิด browser:

```text
http://localhost:5050
```

Login ด้วย:

```text
Email: PGADMIN_EMAIL
Password: PGADMIN_PASS
```

จากนั้น Add New Server:

```text
Name: oshopping_postgres
Host name/address: postgres_db
Port: 5432
Maintenance database: gmm_oshopping_db
Username: admin
Password: ตามค่า DB_PASS
```

## คำสั่งที่ใช้บ่อย

เปิด database:

```powershell
docker compose up -d
```

ปิด database:

```powershell
docker compose down
```

ดู log:

```powershell
docker compose logs -f
```

เข้า PostgreSQL shell:

```powershell
docker exec -it oshopping_postgres psql -U admin -d gmm_oshopping_db
```

ดู table:

```sql
\dt
```

ออก:

```sql
\q
```

## ถ้าจะนำ Dataset เข้า PostgreSQL

Dataset สำหรับส่งต่อควรอยู่ที่:

```text
Model Use/02_Datasets/clean_dataset_s1.csv
Model Use/02_Datasets/real_dataset_s1.csv
```

แนะนำให้ตั้งชื่อตาราง:

```text
clean_dataset_s1
real_dataset_s1
```

วิธีง่ายสำหรับคนทั่วไป:

1. เปิด pgAdmin
2. สร้าง table ตาม columns ของ CSV
3. ใช้เมนู Import/Export
4. เลือก CSV
5. ตั้ง Header = Yes
6. Import เข้า database `gmm_oshopping_db`

ถ้าใช้ command line:

```powershell
psql -h 127.0.0.1 -p 5433 -U admin -d gmm_oshopping_db
```

ตัวอย่างคำสั่ง import:

```sql
\copy clean_dataset_s1 FROM 'C:/path/to/clean_dataset_s1.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');
```

หมายเหตุ: ต้องสร้าง table ก่อนถึงจะ `\copy` ได้ ถ้ายังไม่มี table ให้ใช้ pgAdmin import หรือเขียน Python/Pandas import แทน

## ข้อควรระวัง

- ห้ามส่ง `.env` ที่มี password จริงขึ้น GitHub
- ให้ส่ง `.env.example` แล้วให้คนรับงาน copy เป็น `.env` เอง
- ถ้าเป็นข้อมูลจริงของบริษัท ห้าม commit CSV จริงลง GitHub
- ไฟล์ CSV ใหญ่มากควรใช้ Git LFS หรือเก็บแยกนอก repo
- ถ้าขึ้น production จริงควรเปลี่ยน password และตั้ง backup database

## สรุปคำสั่งเริ่มต้น

```powershell
cd "Model Use\05_Environment"
copy .env.example .env
docker compose up -d
docker ps
```
