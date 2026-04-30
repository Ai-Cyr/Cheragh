# Chunking avancé et query routing

Cette note décrit les ajouts v0.4.1.

## Chunking sémantique

```python
from cheragh import HashingEmbedding
from cheragh.ingestion import SemanticChunker

chunker = SemanticChunker(
    embedding_model=HashingEmbedding(),
    breakpoint_threshold=0.75,
    max_chunk_size=1200,
)
chunks = chunker.split_documents(docs)
```

`SemanticChunker` calcule la similarité cosinus entre phrases adjacentes et coupe le texte lorsque la similarité descend sous le seuil.

## CodeChunker

```python
from cheragh.ingestion import CodeChunker

chunks = CodeChunker().split_documents(code_docs)
```

Le chunker détecte les symboles Python, JavaScript/TypeScript et SQL. Les métadonnées incluent `code_language`, `symbol_name`, `start_line` et `end_line`.

## TableChunker

```python
from cheragh.ingestion import TableChunker

chunks = TableChunker(rows_per_chunk=20).split_documents(docs)
```

Les métadonnées incluent `table_index`, `row_start`, `row_end`, `column_count` et `table_format`.

## PDFLayoutChunker

```python
from cheragh.ingestion import PDFLayoutChunker

chunks = PDFLayoutChunker().split_documents(pdf_page_docs)
```

Il découpe les pages PDF en blocs layout-inspired et conserve les métadonnées `page`, `source`, `bbox` si elles existent.

## HierarchicalChunker

```python
from cheragh.ingestion import HierarchicalChunker

chunks = HierarchicalChunker(include_parent_sections=True).split_documents(docs)
```

Il génère des chunks parent/child avec `section_path`, `hierarchy_level`, `chunk_role` et `parent_section_id`.

## QueryRouter applicatif

```python
from cheragh.routing import QueryRouter

router = QueryRouter(
    routes={
        "qa": qa_engine,
        "summary": summary_engine,
        "sql": sql_engine,
        "fallback": fallback_engine,
    },
    route_descriptions={
        "qa": "questions factuelles sur le corpus",
        "summary": "résumés et synthèses",
        "sql": "questions sur données structurées, ventes, revenus, tables",
        "fallback": "questions hors corpus",
    },
)

response = router.ask("Compare les ventes Q1 et Q2")
```

Routes supportées : objets avec `ask`, objets avec `run`, retrievers avec `retrieve`, ou callables. Une décision de routing est ajoutée dans `response.metadata["routing"]` lorsque la route retourne une réponse structurée, ou dans `response["routing"]` pour les dictionnaires.
