import sqlite3
import re
import textwrap
import csv
from pathlib import Path

COLOR_MONEY = '\033[92m'  # Grün
COLOR_ACTOR = '\033[96m'  # Cyan
COLOR_RESET = '\033[0m'

def highlight_text(text: str) -> str:
    # Optimierter Regex: Erkennt jetzt auch TEUR, Mio., Mrd. und glatte Beträge besser
    money_pattern = r'(\d{1,3}(?:\.\d{3})*(?:,\d+)?\s*(?:€|Euro|Mio\.?|Mrd\.?|Tsd\.?|TEUR))'
    text = re.sub(money_pattern, rf"{COLOR_MONEY}\1{COLOR_RESET}", text, flags=re.IGNORECASE)
    
    # Deine exzellente Liste an Netzwerk- und Finanz-Triggern
    trigger_words = [
        'Förder', 'Zuschuss', 'Invest', 'Finanz', 'Spende', 'Stiftung', 
        'Kooperation', 'unterstützt', 'gefördert', 'Beteiligung', 'Genossenschaft',
        'GmbH', 'AG', 'Verein', 'Bund', 'Land', 'EU-', 'KfW'
    ]
    for trigger in trigger_words:
        text = re.sub(rf'({trigger}[a-zA-ZäöüÄÖÜß]*)', rf"{COLOR_ACTOR}\1{COLOR_RESET}", text, flags=re.IGNORECASE)
        
    return text

def analyze_finances(export_csv: bool = True):
    db_path = Path("crawler/data/db/crawl.sqlite")
    
    if not db_path.exists():
        print(f"❌ Datenbank nicht gefunden unter: {db_path}")
        print("Bitte zuerst den Crawler starten und einige Daten sammeln!")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Optimierte Query: Nutzt deine Klima/Finanz-Filter + ignoriert Impressum-Rauschen (is_negative)
    query = """
        SELECT d.municipality_id, d.url_canonical, s.text, s.impact_score 
        FROM segments s
        JOIN documents_raw d ON s.document_id = d.document_id
        WHERE (s.text LIKE '%€%' OR s.text LIKE '%Euro%' OR s.text LIKE '%Mio%' OR s.text LIKE '%TEUR%')
          AND (s.text LIKE '%KfW%' OR s.text LIKE '%EU-%' OR s.text LIKE '%Bund%' 
               OR s.text LIKE '%Förder%' OR s.text LIKE '%Investition%' OR s.text LIKE '%Zuschuss%')
          AND (s.text LIKE '%Klima%' OR s.text LIKE '%Energie%' OR s.text LIKE '%Wärme%' OR s.text LIKE '%Solar%')
          AND COALESCE(s.is_negative, 0) = 0
        ORDER BY s.impact_score DESC
        LIMIT 50; -- Limit etwas höher, um einen guten CSV-Export zu haben
    """
    
    try:
        cursor.execute(query)
        results = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"❌ SQL-Fehler: {e}")
        return
    finally:
        conn.close()

    print(f"\n{'='*80}")
    print(f"🚀 FINANZ-DETEKTOR: {len(results)} GraphRAG-Kandidaten gefunden!")
    print(f"{'='*80}\n")

    if not results:
        print("Bisher noch keine passenden Segmente gecrawlt. Lass den Crawler noch etwas laufen!")
        return

    # CSV-Export für den HPC vorbereiten
    if export_csv:
        export_path = Path("hpc_finance_claims.csv")
        with open(export_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["municipality_id", "url", "impact_score", "text"])
            for muni_id, url, text, score in results:
                writer.writerow([muni_id, url, score, text])
        print(f"💾 Export erfolgreich: {len(results)} Datensätze in {export_path} gespeichert!\n")

    # Nur die Top 15 im Terminal anzeigen, um es übersichtlich zu halten
    for muni_id, url, text, score in results[:15]:
        print(f"📍 Gemeinde:  {muni_id} (Score: {score})")
        print(f"🔗 Quelle:    {url}")
        print("-" * 80)
        
        clean_text = " ".join(text.split())
        
        # WICHTIG: ERST umbrechen, DANN färben!
        wrapped_text = textwrap.fill(clean_text, width=100)
        highlighted = highlight_text(wrapped_text)
        
        print(highlighted)
        print(f"\n{'='*80}\n")

if __name__ == "__main__":
    analyze_finances()