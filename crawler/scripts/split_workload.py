import sqlite3
import os
import shutil

def split_db(source_db_path, num_splits=20):
    if not os.path.exists(source_db_path):
        print(f"Fehler: {source_db_path} nicht gefunden!")
        return

    conn = sqlite3.connect(source_db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT municipality_id FROM seed_jobs WHERE status = 'pending' ORDER BY municipality_id")
    all_pending = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT * FROM seed_jobs WHERE municipality_id = '09162000'")
    munich_data = cursor.fetchone()

    dist_dir = 'distribution_packages'
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
    os.makedirs(dist_dir)

    if not all_pending:
        print("Keine pending Jobs gefunden!")
        return

    avg = len(all_pending) // num_splits
    
    for i in range(num_splits):
        start = i * avg
        end = None if i == num_splits - 1 else (i + 1) * avg
        subset = all_pending[start:end]
        
        pkg_name = f"pkg_{i+1:02d}"
        pkg_path = os.path.join(dist_dir, pkg_name)
        os.makedirs(pkg_path)
        
        db_path = os.path.join(pkg_path, "crawl.sqlite")
        new_conn = sqlite3.connect(db_path)

        for table in ["seed_jobs", "documents_raw", "segments"]:
            res = conn.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'").fetchone()
            if res:
                new_conn.execute(res[0])
        
        if munich_data:
            placeholders_munich = ','.join(['?'] * len(munich_data))
            new_conn.execute(f"INSERT INTO seed_jobs VALUES ({placeholders_munich})", munich_data)
        
        placeholders_subset = ','.join(['?'] * len(all_pending[0:1]))
        ids_subset = subset
        placeholders_ids = ','.join(['?'] * len(ids_subset))
        jobs_data = conn.execute(f"SELECT * FROM seed_jobs WHERE municipality_id IN ({placeholders_ids})", ids_subset).fetchall()
        
        if jobs_data:
            placeholders_row = ','.join(['?'] * len(jobs_data[0]))
            new_conn.executemany(f"INSERT INTO seed_jobs VALUES ({placeholders_row})", jobs_data)
        
        new_conn.commit()
        new_conn.close()

        with open(os.path.join(pkg_path, "ANLEITUNG.txt"), "w") as f:
            f.write(f"PROJEKT: KIT KlimaCrawler\nPAKET: {pkg_name}\n")
            f.write("-" * 30 + "\n")
            f.write("1. Kopiere 'crawl.sqlite' in deinen Ordner: crawler/data/db/\n")
            f.write("2. Starte den Crawl mit: caffeinate -i python3 -m crawler.scripts.run_worker\n")
            f.write("3. Wenn fertig, lade die Datei als 'crawl_{pkg_name}_DONE_Name.sqlite' hoch.\n")

    print(f"✅ 20 Pakete in '{dist_dir}' erstellt (basierend auf {len(munich_data)} Spalten).")
    conn.close()

if __name__ == "__main__":
    split_db("crawler/data/db/crawl.sqlite")