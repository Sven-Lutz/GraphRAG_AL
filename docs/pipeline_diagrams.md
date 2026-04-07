# KIT KlimaCrawler — GraphRAG Pipeline Diagrams

Visual overview of the full index-time and query-time flows.

---

## 1. Index-Time Pipeline (Data Ingestion → Graph Construction)

```mermaid
flowchart TD
    subgraph CRAWL["🕷️ Stage 0 — Distributed Crawl"]
        S0A[seed_jobs table\nATOMIC municipality claim] --> S0B[Fetcher\nasync · polite delay · robots.txt]
        S0B --> S0C{Content-Type?}
        S0C -->|text/html| S0D[HTML Parser\nmeta tags · heading_context h1–h6\noutlinks → document_links]
        S0C -->|application/pdf| S0E[PDF Parser\nper-page sub-chunking\n≤3000 chars → paragraph split]
        S0D --> S0F[Scorer\nkeyword hit · heading boost\nis_negative filter]
        S0E --> S0F
        S0F --> S0G[(SQLite\nsegments · documents_raw\ndocument_links)]
    end

    subgraph CHUNK["📄 Stage 1 — Tiktoken Chunking"]
        S1A[(segments\nimpact_score ≥ threshold)] --> S1B[chunker.py\ncl100k_base\n600 tokens / 60 overlap]
        S1B --> S1C[(SQLite\nchunks table\nchunk_id · heading_context\nimpact_score · embedding BLOB)]
    end

    subgraph EMBED["🔢 Stage 2a — Vector Embeddings"]
        S2A[(chunks\nembedding IS NULL)] --> S2B[SentenceTransformers\nparaphrase-multilingual-MiniLM-L12-v2\n384-dim · CUDA/MPS/CPU]
        S2B --> S2C[float32 BLOB\n→ SQLite chunks.embedding]
        S2B --> S2D[Neo4j Chunk nodes\nchunk_embedding vector index\ncosine similarity]
    end

    subgraph GRAPH["🧠 Stage 2b — Graph Extraction (Ollama)"]
        S3A[(segments / chunks\nhigh impact_score)] --> S3B[extract_graph_ollama.py\nOllama llama3.1:8b\nformat=json]
        S3B --> S3C{Gleaning}
        S3C -->|Round 1| S3D[Initial extraction\nEntities + Relationships\nPydantic schema]
        S3D --> S3E[Self-reflection pass\n'Did you miss entities?']
        S3E --> S3F[Merge unique entities\nlowercase dedup]
        S3C -->|N rounds| S3E
        S3F --> S3G[(SQLite\ngraph_triplets)]
    end

    subgraph NEO4J["🗄️ Stage 2c — Neo4j Export"]
        S4A[(graph_triplets)] --> S4B[export_graph_to_neo4j.py\nentity_canonical resolution\nfuzzy name normalisation]
        S4B --> S4C[(Neo4j\nEntity nodes\nRelationship edges\nSegment · Document · Chunk)]
        S4B --> S4D[document_links\nLINKS_TO edges]
        S4D --> S4C
    end

    subgraph COMM["🔍 Stage 3 — Community Detection"]
        S5A[(Neo4j Entity graph)] --> S5B[GDS graph.project\nklimiaEntities\n9 rel-types UNDIRECTED]
        S5B --> S5C{Algorithm}
        S5C -->|default| S5D[gds.leiden.write\ngamma=1.0]
        S5C -->|alt| S5E[gds.louvain.write]
        S5D --> S5F[communityId on Entity nodes]
        S5E --> S5F
        S5F --> S5G[LLM Community Report\nCommunityReport Pydantic\ntitle · summary · importance\ndomain · key_claims · key_entities]
        S5G --> S5H[(Neo4j CommunityReport nodes\nBELONGS_TO_COMMUNITY edges)]
        S5G --> S5I[(SQLite\ncommunity_reports table)]
    end

    CRAWL --> CHUNK
    CHUNK --> EMBED
    CHUNK --> GRAPH
    EMBED --> NEO4J
    GRAPH --> NEO4J
    NEO4J --> COMM
```

---

## 2. Query-Time Pipeline (Multi-Agent Retrieval)

```mermaid
flowchart TD
    Q[User Query] --> R[Router LLM\nroute to tool]

    R -->|factual / local| LS[local_search\nbrute-force cosine on SQLite BLOBs\n→ top-k chunks\n→ Entity neighbourhood Neo4j\ncontext assembly]

    R -->|thematic / global| GS[global_search — LazyGraphRAG\nkeyword pre-filter on community reports\nmap: LLM summary per report\nreduce: final holistic answer]

    R -->|structured / count| TC[text2cypher\nLLM → Cypher query\nNeo4j execute\n→ markdown table]

    LS --> A[Draft Answer]
    GS --> A
    TC --> A

    A --> C{Answer Critic\ncompleteness check}
    C -->|complete| OUT[Final Answer]
    C -->|incomplete\nloop ≤ MAX_CRITIC_LOOPS=3| FU[follow_up_query\nback to Router]
    FU --> R
```

---

## 3. Evaluation Pipeline (Stage 5)

```mermaid
flowchart LR
    subgraph AutoQ["AutoQ — Synthetic Question Generation"]
        AQ1[community_reports\nSQLite] --> AQ2[generate_local_questions\nfactual, entity-specific]
        AQ1 --> AQ3[generate_global_questions\nthematic, cross-community]
        AQ1 --> AQ4[generate_aggregation_questions\ncounting, comparison]
        AQ2 & AQ3 & AQ4 --> AQ5[TestQuestion list\nJSON export]
    end

    subgraph QA["QA — Answer Generation"]
        AQ5 --> QA1[query_agent.answer\nfor each question]
        QA1 --> QA2[EvalResult\nquestion · answer · contexts\nroute_used · latency_s]
    end

    subgraph RAGAS["RAGAS Metrics"]
        QA2 --> R1[score_faithfulness\nLLM-as-judge]
        QA2 --> R2[score_context_recall\nLLM-as-judge]
        QA2 --> R3[claimify\natomic claim decomposition\nper-claim verification]
        R1 & R2 & R3 --> R4[per-question scores\nfaithfulness · context_recall\nclaim_accuracy]
    end

    subgraph AutoE["AutoE — Head-to-Head Comparison"]
        SYS_A[System A answers] --> AE1[autoe_compare\nLLM judge]
        SYS_B[System B answers] --> AE1
        AE1 --> AE2[winner per dimension\nComprehensiveness\nDiversity\nFaithfulness]
    end

    RAGAS --> RPT[Evaluation Report\nJSON + console summary\nmean scores per route type]
    AutoE --> RPT
```

---

## 4. Data Model (Neo4j Graph Schema)

```mermaid
graph LR
    D[Document] -->|HAS_SEGMENT| S[Segment\nimpact_score\nheading_context\norder_index]
    D -->|HAS_CHUNK| C[Chunk\nembedding float32\ntoken_count\nimpact_score]
    D -->|LINKS_TO| D2[Document]

    S -->|MENTIONS| E[Entity\ntype · category · status\nmetrics dict\ncommunityId]
    E -->|FÖRDERT\nBAUT\nBESCHLIESST\nIMPLEMENTIERT\nPLANT\nGEHÖRT_ZU\nKOOPERIERT_MIT\nFINANZIERT_DURCH\nBEZIEHT_SICH_AUF| E2[Entity]

    E -->|BELONGS_TO_COMMUNITY| CR[CommunityReport\ntitle · summary\nimportance 1–10\ndomain · key_claims]
```

---

## 5. Running the Full Pipeline

```bash
# 0. Crawl municipalities (distributed)
python -m crawler.scripts.init_seed_jobs
python -m crawler.scripts.run_worker

# 1. Tiktoken chunking
python -m crawler.scripts.chunk_documents --chunk-size 600 --overlap 60 --min-score 15

# 2a. Generate embeddings (local GPU)
python -m crawler.scripts.generate_embeddings --model paraphrase-multilingual-MiniLM-L12-v2 --neo4j

# 2b. Extract entity graph (Ollama, privacy-compliant)
python -m crawler.scripts.extract_graph_ollama --model llama3.1:8b --gleaning 1 --chunk-mode

# 2c. Export to Neo4j
python -m crawler.scripts.export_graph_to_neo4j

# 3. Community detection + LLM reports
python -m crawler.scripts.detect_communities --algorithm leiden --min-size 3

# 4. Interactive query (multi-agent)
python -m crawler.scripts.query_agent --interactive

# 5. Scientific evaluation
python -m crawler.scripts.evaluate --questions 30 --output eval_results.json

# Utilities
python -m crawler.scripts.resolve_entities          # Entity deduplication
python -m crawler.scripts.summarize_documents       # Per-document LLM summaries
python -m crawler.scripts.analyze_topics            # BERTopic topic modelling
```
