# KIT KlimaCrawler 🌍

Pipeline zur systematischen Erfassung von Energie- und Klimarichtlinien bayerischer Kommunen für die **GraphRAG-Analyse**.

Distributed Packages: https://bwsyncandshare.kit.edu/s/4gTz25xpaYbw6BW 
---

## 🛠 Voraussetzungen

- Python 3.9+ (empfohlen: 3.10)
- macOS, Linux oder Windows
- Stabile Internetverbindung
- Ausreichend Laufzeit (Großstädte: > 10 Stunden)

---

## 📦 Setup & Installation

### 1️⃣ Repository klonen

```bash
git clone <repository-url>
cd KIT_KlimaCrawler
```

---

### 2️⃣ Virtuelle Umgebung erstellen (wi2026)

#### Windows (PowerShell)

```powershell
python -m venv wi2026
.\wi2026\Scripts\Activate.ps1
```

#### Windows (CMD)

```cmd
python -m venv wi2026
wi2026\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv wi2026
source wi2026/bin/activate
```

#### Alternative: Conda

```bash
conda create -n wi2026 python=3.10 -y
conda activate wi2026
```

---

### 3️⃣ Abhängigkeiten installieren

Im Ordner "crawler" (über cd crawler navigieren) dann:

```bash
pip install -r requirements.txt
```

---

## 👯 Verteiltes Arbeiten (Sharding)

Um Bayern parallel zu crawlen, werden vorab aufgeteilte Datenbank-Pakete verwendet.

> ⚠️ **WICHTIG:**  
> In jedem Paket-Ordner heißt die Datei identisch: `crawl.sqlite`.  
> Der Dateiname muss exakt so bleiben, da der Code darauf referenziert.  
> Der Inhalt ist jedoch pro Paket einzigartig (unterschiedliche Kommunen).

### Ablauf

1. **Reservieren**  
   Trage deinen Namen im Repository in die Datei `TRACKER.md` beim gewählten Paket ein (z. B. `pkg_05`) und pushe die Änderung. -> Warnhinweis -> Create Fork

2. **Paket laden**  
   Lade den entsprechenden Ordner aus der Cloud/GitHub herunter.

3. **Platzieren**  
   Kopiere die enthaltene `crawl.sqlite` in:

   ```
   crawler/data/db/
   ```

   (Bestehende Datei überschreiben.)

   Den Ordner muss man manuell anlegen, der ist nicht im GitHub!

4. **Starten (ohne Limit)**  

   ```bash
   caffeinate -i python3 -m crawler.scripts.run_worker
   ```

5. **Upload nach Abschluss**  
   Datei umbenennen in:

   ```
   pkg_XX_DONE_Name.sqlite
   ```

   und wieder hochladen.

---

## 🏃‍♂️ Crawl starten (Allgemein)

Falls unabhängig von Paketen gearbeitet wird, kann der Umfang manuell gesteuert werden.

### Einzel-Lauf (eine Kommune)

```bash
python3 -m crawler.scripts.run_worker --limit 1
```

### Batch-Lauf (z. B. 100 Kommunen)

```bash
python3 -m crawler.scripts.run_worker --limit 100
```

---

## ☕ WICHTIG: Standby verhindern

Wenn der Rechner in den Ruhezustand geht, bricht die Netzwerkverbindung ab und der Crawl stoppt.

### macOS

```bash
caffeinate -i python3 -m crawler.scripts.run_worker
```

### Windows

 ### Nutze Tools wie:

- Caffeine  
- PowerToys Awake  

### 🔹 Installation

1. Gehe auf die offizielle GitHub-Seite von
Microsoft PowerToys
2. Klicke auf Releases
3. Lade die Datei
PowerToysSetup-xxx-x64.exe herunter
4. Installieren → fertig

### 🔹 Aktivieren von „Awake“

1. PowerToys öffnen
2. Links Awake auswählen
3. Aktivieren
4. Modus wählen:

Keep awake indefinitely

Keep awake for a time interval

Keep awake until expiration

-> Das ist die sauberste Lösung – kein Script, kein Hack.

---

## 📊 Monitoring & Erfolgskontrolle

### Fortschritt prüfen (Extraktions-Statistik)

```bash
sqlite3 crawler/data/db/crawl.sqlite "SELECT segment_type, COUNT(*) FROM segments GROUP BY segment_type;"
```

### Erfolgreicher Lauf

- `run_worker` beendet sich ohne Fehlermeldung  
- `seed_jobs.status` steht auf `done`  
- Tabellen `documents_raw` und `segments` sind befüllt  

- Fehler können aufkommen! Um eine Übersicht zu haben, könnt ihr euer Terminal splitten und 

```bash
while true; do
  clear;
  echo "=== CRAWLER PULS ($(date +%H:%M:%S)) ===";
  sqlite3 crawler/data/db/crawl.sqlite "
    SELECT 'Status ' || status || ': ' || count(*) FROM seed_jobs GROUP BY status;
    SELECT 'Gesamt-Dokumente: ' || count(*) FROM documents_raw;
    SELECT 'Davon PDFs:       ' || count(*) FROM documents_raw WHERE url_canonical LIKE '%.pdf';
  ";
  echo "----------------------------------------";
  echo "Zuletzt gefundene PDFs:";
  sqlite3 crawler/data/db/crawl.sqlite "SELECT url_canonical FROM documents_raw WHERE url_canonical LIKE '%.pdf' ORDER BY rowid DESC LIMIT 5;";
  sleep 30;
done
```

einfügen. 

### Oder

```bash
brew install watch
```

```bash
watch -n 30 "echo '=== CRAWLER DASHBOARD ===' && \
sqlite3 crawler/data/db/crawl.sqlite \"SELECT 'Status ' || status || ': ' || count(*) FROM seed_jobs GROUP BY status;\" && \
echo '---' && \
echo 'Gesamt-Dokumente:' && \
sqlite3 crawler/data/db/crawl.sqlite \"SELECT count(*) FROM documents_raw;\" && \
echo '---' && \
echo 'Neueste Nachhaltigkeits-PDFs:' && \
sqlite3 crawler/data/db/crawl.sqlite \"SELECT municipality_id, url_canonical FROM documents_raw WHERE (url_canonical LIKE '%klima%' OR url_canonical LIKE '%solar%' OR url_canonical LIKE '%mobil%' OR url_canonical LIKE '%haushalt%') AND url_canonical LIKE '%.pdf' ORDER BY rowid DESC LIMIT 5;\""
```
---

## ⚠ Fehlerquellen

- **Netzwerk:** VPN-Abbruch oder instabiles WLAN beendet den Prozess  
- **Tools:** `pdftotext` muss im Systempfad installiert sein  
- **Pfad-Fehler:** Die Datenbank muss exakt in `crawler/data/db/` liegen  

---

## 📌 Kurzfassung (TL;DR)

```bash
git clone <repository-url>
cd KIT_KlimaCrawler
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
caffeinate -i python3 -m crawler.scripts.run_worker
```

