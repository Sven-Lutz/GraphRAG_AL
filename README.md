KIT KlimaCrawler & GraphRAG Pipeline 🌍🧠
=========================================

Dieses Repository enthält die vollständige Pipeline zur systematischen Erfassung und Analyse von Energie- und Klimarichtlinien bayerischer Kommunen.

Das Projekt kombiniert einen verteilten **High-Recall Web-Scraper** (Stage 0) mit einer modernen **GraphRAG-Architektur** (Retrieval-Augmented Generation auf Basis von Knowledge Graphen) (Stage 1-5). So können komplexe semantische Suchen, thematische Analysen und genaue Zählungen in kommunalen Dokumenten durchgeführt werden.

* * * * *

🛠 Voraussetzungen
------------------

Bevor du startest, stelle sicher, dass folgende Software installiert ist:

-   **Python 3.9+** (empfohlen: 3.10)

-   **Neo4j Datenbank** (Desktop oder Server) für den Knowledge Graph

-   **Ollama** (für lokale LLM-Inferenz, datenschutzkonform, z.B. für `llama3.1:8b`)

-   Ausreichend Laufzeit und ggf. eine GPU (für das lokale Generieren von Embeddings & Graphen)

* * * * *

📦 1. Setup & Installation
--------------------------

### 1.1 Repository klonen

git clone cd KIT_KlimaCrawler

### 1.2 Virtuelle Umgebung erstellen

Es wird dringend empfohlen, eine virtuelle Umgebung zu nutzen.

**Windows (PowerShell):** python -m venv venv .\venv\Scripts\Activate.ps1

**macOS / Linux:** python3 -m venv venv source venv/bin/activate

### 1.3 Abhängigkeiten installieren

Installiere alle notwendigen Pakete aus der `requirements.txt`:

pip install -r crawler/requirements.txt

*(Hinweis: Für lokale Embeddings mit GPU-Unterstützung muss PyTorch ggf. entsprechend der Systemarchitektur separat installiert werden).*

* * * * *

🚀 2. Die Pipeline: Schritt-für-Schritt Anleitung
-------------------------------------------------

Die Architektur ist in 5 Hauptphasen unterteilt. Führe die folgenden Befehle nacheinander aus, um vom nackten Datensatz bis zum interaktiven Chat-Agenten zu gelangen.

### Stage 0: Crawling (Datenbeschaffung)

Der Crawler sammelt HTML-Seiten und PDFs von kommunalen Webseiten und speichert sie in einer SQLite-Datenbank (`crawler/data/db/crawl.sqlite`).

Initialisiere die Seed-Jobs (Startpunkte) und starte den Worker:

python -m crawler.scripts.init_seed_jobs python -m crawler.scripts.run_worker

> **Tipp (Mac):** Nutze `caffeinate -i python -m crawler.scripts.run_worker`, um den Mac am Einschlafen zu hindern. (Für Windows: Nutze *Microsoft PowerToys Awake*).

*(Für verteiltes Crawlen mit vorab aufgeteilten Datenbank-Paketen siehe unten im Abschnitt "Verteiltes Arbeiten").*

### Stage 1: Chunking (Textzerlegung)

Um die Dokumente für das Sprachmodell verdaulich zu machen, werden sie in handliche Token-Blöcke (Chunks) zerlegt (hier: 600 Tokens mit 60 Tokens Überlappung).

python -m crawler.scripts.chunk_documents --chunk-size 600 --overlap 60 --min-score 15

### Stage 2: Embeddings & Knowledge Graph Extraktion

In dieser Phase wird der Text durchsuchbar gemacht und in strukturierte Entitäten und Beziehungen übersetzt.

**2a. Vector Embeddings generieren (Lokales Modell):** python -m crawler.scripts.generate_embeddings --model paraphrase-multilingual-MiniLM-L12-v2 --neo4j

**2b. Entitäten & Beziehungen extrahieren (via lokales Ollama):** *Stelle sicher, dass Ollama im Hintergrund läuft und das Modell geladen ist.* python -m crawler.scripts.extract_graph_ollama --model llama3.1:8b --gleaning 1 --chunk-mode

**2c. Export in die Neo4j Datenbank:** Überträgt die extrahierten Daten (Entitäten, Beziehungen, Chunks) in deine laufende Neo4j-Instanz. *(Konfiguriere vorher deine Neo4j-Zugangsdaten in der `crawler/core/config.py` oder per `.env`).* python -m crawler.scripts.export_graph_to_neo4j

### Stage 3: Community Detection (Netzwerkanalyse)

Gruppiert verwandte Entitäten im Knowledge Graph zu "Communities" und lässt das LLM automatische Reports über diese Themen-Cluster schreiben.

python -m crawler.scripts.detect_communities --algorithm leiden --min-size 3

### Stage 4: Querying (Interaktiver Agent)

Starte den Multi-Agenten, der Nutzerfragen über einen "Router" entgegennimmt und entscheidet, ob er Vektor-Suche (Local Search), Map-Reduce über Community-Reports (Global Search) oder Cypher-Datenbankabfragen (Text2Cypher) nutzt.

python -m crawler.scripts.query_agent --interactive

* * * * *

👯 Verteiltes Arbeiten beim Crawling (Sharding)
-----------------------------------------------

Wenn wir Bayern im Team parallel crawlen wollen, nutzen wir vorab aufgeteilte Datenbank-Pakete.

> ⚠️ **WICHTIG:** In jedem Paket-Ordner heißt die Datei identisch: `crawl.sqlite`. Der Name muss exakt so bleiben.

1.  **Reservieren:** Trage deinen Namen in die `TRACKER.md` beim gewählten Paket ein (z.B. `pkg_05`) und pushe die Änderung.

2.  **Paket laden:** Lade den Ordner herunter.

3.  **Platzieren:** Erstelle den Ordner `crawler/data/db/` falls er noch nicht existiert und lege die heruntergeladene `crawl.sqlite` genau dort ab.

4.  **Starten:** caffeinate -i python -m crawler.scripts.run_worker

5.  **Upload nach Abschluss:** Wenn der Worker durch ist, benenne die Datei um (z.B. in `pkg_05_DONE_Name.sqlite`) und lade sie wieder in die Cloud.

* * * * *

📊 Monitoring & Erfolgskontrolle (Stage 0)
------------------------------------------

Um während eines laufenden Crawls den Überblick zu behalten, öffne ein zweites Terminalfenster und nutze dieses Monitoring-Script:

**Einfacher Puls-Check (Mac/Linux):**

watch -n 30 "echo '=== CRAWLER DASHBOARD ===' &&

sqlite3 crawler/data/db/crawl.sqlite "SELECT 'Status ' || status || ': ' || count() FROM seed_jobs GROUP BY status;" &&

echo '---' &&

echo 'Gesamt-Dokumente:' &&

sqlite3 crawler/data/db/crawl.sqlite "SELECT count() FROM documents_raw;""

📈 Evaluation (Stage 5)
-----------------------

Um die RAG-Architektur wissenschaftlich zu überprüfen (Context Recall, Faithfulness, Claim Accuracy), kann eine synthetische Question-Answering-Evaluation gestartet werden:

python -m crawler.scripts.evaluate --questions 30 --output eval_results.json

⚠ Typische Fehlerquellen
------------------------

-   **Prozess stoppt plötzlich:** Rechner ist in den Standby gegangen. Unbedingt Tools wie *caffeinate* (Mac) oder *Awake* (Windows) nutzen!

-   **Neo4j Fehler beim Export:** Stelle sicher, dass die Neo4j-Datenbank aktiv ist und die Credentials in der Konfigurationsdatei stimmen.

-   **Ollama Timeout:** Beim ersten Mal muss das Modell (`llama3.1:8b`) erst heruntergeladen werden (`ollama run llama3.1:8b` vorab im Terminal ausführen).

-   **Pfad-Fehler:** Die SQLite-Datenbank muss immer unter exakt `crawler/data/db/crawl.sqlite` liegen.
