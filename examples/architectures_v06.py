from cheragh import Document, HashingEmbedding, StaticLLMClient, RAGEngine
from cheragh.multihop import MultiHopRAGEngine
from cheragh.graph import GraphRAGEngine
from cheragh.raptor_engine import RAPTOREngine
from cheragh.federated import FederatedRAGEngine


docs = [
    Document("Client Alpha a généré 100 euros au Q1. Client Alpha utilise Produit A.", doc_id="sales-alpha"),
    Document("Client Beta a généré 180 euros au Q1. Client Beta utilise Produit B.", doc_id="sales-beta"),
    Document("Produit B a un risque contractuel élevé avec Fournisseur Delta.", doc_id="risk-beta"),
]

embedder = HashingEmbedding(dimension=128)
llm = StaticLLMClient("Réponse démonstrative avec sources. [source: sales-beta]")

base = RAGEngine.from_documents(docs, embedding_model=embedder, llm_client=llm)

multi = MultiHopRAGEngine(base.retriever, llm_client=llm)
print("Multi-hop:", multi.ask("Compare Alpha et Beta puis indique les risques").answer)

graph = GraphRAGEngine.from_documents(docs, embedding_model=embedder, llm_client=llm)
print("Graph:", graph.ask("Que sait-on sur Client Beta ?").metadata["query_entities"])

raptor = RAPTOREngine.from_documents(docs, embedding_model=embedder, llm_client=llm, levels=1)
print("RAPTOR:", raptor.ask("Résume les clients").metadata["raptor_index"])

federated = FederatedRAGEngine({"sales": base, "risk": graph}, llm_client=llm)
print("Federated:", federated.ask("Compare ventes et risques").metadata["sources_queried"])
