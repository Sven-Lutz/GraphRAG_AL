from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

DB_PATH = Path("crawler/data/db/crawl.sqlite")
OUT_DIR = Path("topic_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. NEU: Erweiterte Liste für deutsche Stoppwörter (Behörden-Rauschen filtern)
GERMAN_STOP_WORDS = [
    "der", "die", "das", "und", "in", "den", "von", "zu", "im", "für", "mit", 
    "auf", "des", "sich", "als", "ein", "eine", "ist", "werden", "aus", "dem", 
    "bei", "um", "zur", "oder", "über", "sind", "auch", "am", "an", "nach", 
    "wie", "wir", "dass", "durch", "zum", "es", "nicht", "noch", "was", "wird",
    "da", "kann", "können", "mehr", "nur", "so", "hat", "haben", "bis", "sehr"
]

def load_scored_segments(
    db_path: Path,
    limit: int = 60_000,
    min_len: int = 150,
    min_score: int = 15,
    per_doc: int = 4,
) -> pd.DataFrame:
    """
    Lädt nur die relevantesten Segmente mithilfe der Crawler-Scores.
    Sorgt dafür, dass keine einzelne Gemeinde/Dokument das Modell dominiert (per_doc Limit).
    """
    query = """
    WITH ranked AS (
      SELECT
        s.text,
        s.document_id,
        d.municipality_id,
        COALESCE(s.impact_score, 0) AS impact_score,
        COALESCE(s.is_negative, 0) AS is_negative,
        ROW_NUMBER() OVER (
          PARTITION BY s.document_id
          ORDER BY COALESCE(s.impact_score, 0) DESC
        ) AS rn
      FROM segments s
      JOIN documents_raw d ON d.document_id = s.document_id
      WHERE length(s.text) >= ?
        AND COALESCE(s.is_negative, 0) = 0
        AND COALESCE(s.impact_score, 0) >= ?
    )
    SELECT text, document_id, municipality_id, impact_score
    FROM ranked
    WHERE rn <= ?
    ORDER BY impact_score DESC
    LIMIT ?;
    """
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=(min_len, min_score, per_doc, limit))

    # Minimal Cleanup
    df["text"] = (
        df["text"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)  # Mehrfache Leerzeichen (Whitespace-Rauschen) filtern
        .str.strip()
    )
    df = df[df["text"].str.len() >= min_len]
    df = df.drop_duplicates(subset=["text"])
    return df.reset_index(drop=True)


def main():
    print("Lade Daten aus der SQLite-Datenbank...")
    # ---- Load corpus (scored) ----
    df = load_scored_segments(
        DB_PATH,
        limit=120_000,
        min_len=150,
        min_score=15,
        per_doc=6,
    )
    if df.empty:
        raise RuntimeError("Keine Segmente gefunden! Stelle sicher, dass der Crawler mit Scores gelaufen ist.")

    docs = df["text"].tolist()
    print(f"{len(docs):,} bewertete Segmente für das Topic Modeling geladen.")

    # 2. NEU: GPU embedding (Automatische Erkennung für HPC (cuda) & Mac (mps))
    if torch.cuda.is_available():
        device = "cuda"  # HPC
    elif torch.backends.mps.is_available():
        device = "mps"   # Apple Silicon (M1/M2/M3)
    else:
        device = "cpu"   # Fallback
        
    print(f"Embedding Device erkannt: {device}")

    print("Lade Modell für Vektorisierung...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
    embedding_model.max_seq_length = 256

    # Precompute embeddings (schneller & reproduzierbarer)
    print("Berechne Embeddings (dein Mac wird jetzt kurz warm)...")
    batch_size = 256 if device in ["cuda", "mps"] else 64
    embeddings = embedding_model.encode(
        docs,
        batch_size=256 if device == "cuda" else 64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # ---- Vectorizer / Representation ----
    print("Stoppwörter und N-Gramme anwenden...")
    vectorizer_model = CountVectorizer(
        stop_words=GERMAN_STOP_WORDS, # 3. NEU: Stoppwörter übergeben
        ngram_range=(1, 3),
        min_df=5,
        max_df=0.95,
        max_features=120_000,
        token_pattern=r"(?u)\b[\wäöüÄÖÜß]{3,}\b", # Mindestens 3 Buchstaben pro Wort
    )

    representation_model = KeyBERTInspired()

    print("Starte BERTopic Analyse...")
    topic_model = BERTopic(
        embedding_model=None,  # Embeddings sind schon fertig berechnet
        language="german",
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        calculate_probabilities=False,
        nr_topics="auto",
        min_topic_size=25,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)

    # ---- Save outputs ----
    print("Speichere Ergebnisse...")
    info = topic_model.get_topic_info()
    info.to_csv(OUT_DIR / "topic_info.csv", index=False)

    out = df.copy()
    out["topic"] = topics
    out.to_parquet(OUT_DIR / "segment_topics.parquet", index=False)

    topic_model.visualize_topics().write_html(str(OUT_DIR / "topic_map.html"))
    topic_model.visualize_barchart(top_n_topics=15).write_html(str(OUT_DIR / "topic_bar_chart.html"))

    print(f"🎉 Fertig! Die Visualisierungen und Datensätze liegen im Ordner: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()