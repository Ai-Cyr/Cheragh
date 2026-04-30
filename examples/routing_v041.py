"""Minimal v0.4.1 query-routing example."""
from cheragh.routing import QueryRouter


def qa_engine(query: str, **kwargs):
    return {"answer": f"QA route: {query}"}


def summary_engine(query: str, **kwargs):
    return {"answer": f"Summary route: {query}"}


def sql_engine(query: str, **kwargs):
    return {"answer": f"SQL route: {query}"}


router = QueryRouter(
    routes={"qa": qa_engine, "summary": summary_engine, "sql": sql_engine},
    route_descriptions={
        "qa": "questions factuelles sur des documents",
        "summary": "résumés et synthèses",
        "sql": "données structurées, ventes, revenus, trimestres",
    },
)

if __name__ == "__main__":
    for query in [
        "Quelle est la politique de remboursement ?",
        "Résume le document",
        "Compare les ventes Q1 et Q2",
    ]:
        print(router.ask(query))
