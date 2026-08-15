"""Cheragh toolkit for standard Python applications.

Imports from the package root are intentionally lazy for heavier integrations and
experimental techniques. Core abstractions are available immediately; everything
else is loaded on first access.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .base import (
    BaseRetriever,
    CallableLLMClient,
    Document,
    EmbeddingModel,
    ExtractiveLLMClient,
    HashingEmbedding,
    LLMClient,
    OpenAILLMClient,
    SentenceTransformerEmbedding,
    StaticLLMClient,
)

if TYPE_CHECKING:
    from .adaptive import AdaptiveRetriever as AdaptiveRetriever
    from .adaptive import GateDecision as GateDecision
    from .agentic import AgentAction as AgentAction
    from .agentic import AgenticRAGEngine as AgenticRAGEngine
    from .agentic import AgenticRAGResult as AgenticRAGResult
    from .agentic import AgentPlanner as AgentPlanner
    from .agentic import AgentStep as AgentStep
    from .agentic import AgentTool as AgentTool
    from .agentic import AgentTrace as AgentTrace
    from .agentic import LLMJSONPlanner as LLMJSONPlanner
    from .agentic import RetrievalToolAdapter as RetrievalToolAdapter
    from .agentic import ScriptedPlanner as ScriptedPlanner
    from .agentic import ToolObservation as ToolObservation
    from .agentic import ToolRegistry as ToolRegistry
    from .agentic import ToolSpec as ToolSpec
    from .cache import CacheBackend as CacheBackend
    from .cache import CacheEntry as CacheEntry
    from .cache import CacheSerializerError as CacheSerializerError
    from .cache import CacheStats as CacheStats
    from .cache import CachedEmbeddingModel as CachedEmbeddingModel
    from .cache import CachedLLMClient as CachedLLMClient
    from .cache import CachedReranker as CachedReranker
    from .cache import CachedRetriever as CachedRetriever
    from .cache import MemoryCache as MemoryCache
    from .cache import SQLiteCache as SQLiteCache
    from .cache import build_cache_backend as build_cache_backend
    from .cache import cache_embedding_model as cache_embedding_model
    from .cache import cache_llm_client as cache_llm_client
    from .cache import cache_method as cache_method
    from .cache import cache_reranker as cache_reranker
    from .cache import cache_retriever as cache_retriever
    from .cache import cached_call as cached_call
    from .cache import embedder_fingerprint as embedder_fingerprint
    from .cache import hash_documents as hash_documents
    from .cache import load_cache as load_cache
    from .cache import make_cache_key as make_cache_key
    from .cache import save_cache as save_cache
    from .cache.redis import RedisCache as RedisCache
    from .catalog import TECHNIQUES as TECHNIQUES
    from .catalog import TechniqueFamily as TechniqueFamily
    from .catalog import TechniqueSpec as TechniqueSpec
    from .catalog import TechniqueStatus as TechniqueStatus
    from .catalog import get_technique as get_technique
    from .catalog import list_techniques as list_techniques
    from .chain_of_note import ChainOfNoteRetriever as ChainOfNoteRetriever
    from .citations import CitationValidationResult as CitationValidationResult
    from .citations import citation_location as citation_location
    from .citations import extract_citations as extract_citations
    from .citations import validate_citations as validate_citations
    from .compression import CompressionPipeline as CompressionPipeline
    from .compression import ContextCompressor as ContextCompressor
    from .compression import ExtractiveContextCompressor as ExtractiveContextCompressor
    from .compression import RedundancyFilter as RedundancyFilter
    from .contextual_compression import ContextualCompressionRetriever as ContextualCompressionRetriever
    from .conversation import ConversationTurn as ConversationTurn
    from .conversation import ConversationalRAGEngine as ConversationalRAGEngine
    from .conversation import InMemoryConversationStore as InMemoryConversationStore
    from .corrective import CorrectiveRAGEngine as CorrectiveRAGEngine
    from .corrective import CorrectiveRAGResult as CorrectiveRAGResult
    from .corrective import LexicalRetrievalGrader as LexicalRetrievalGrader
    from .corrective import RetrievalGrade as RetrievalGrade
    from .corrective_rag import CorrectiveRAGRetriever as CorrectiveRAGRetriever
    from .corrective_rag import DocQuality as DocQuality
    from .embeddings import AzureOpenAIEmbedding as AzureOpenAIEmbedding
    from .embeddings import CohereEmbedding as CohereEmbedding
    from .embeddings import OpenAIEmbedding as OpenAIEmbedding
    from .embeddings import VoyageEmbedding as VoyageEmbedding
    from .engine import RAGEngine as RAGEngine
    from .engine import RAGStream as RAGStream
    from .evaluation import GenerationEvaluationResult as GenerationEvaluationResult
    from .evaluation import RetrievalEvaluationResult as RetrievalEvaluationResult
    from .evaluation import RetrievalExample as RetrievalExample
    from .evaluation import context_precision_at_k as context_precision_at_k
    from .evaluation import evaluate_generation as evaluate_generation
    from .evaluation import evaluate_pipeline as evaluate_pipeline
    from .evaluation import evaluate_retrieval as evaluate_retrieval
    from .evaluation import ndcg_at_k as ndcg_at_k
    from .evaluation import recall_at_k as recall_at_k
    from .federated import FederatedRAGEngine as FederatedRAGEngine
    from .federated import FederatedRAGResult as FederatedRAGResult
    from .federated import FederatedRetriever as FederatedRetriever
    from .federated import FederatedSourceResult as FederatedSourceResult
    from .feedback import FeedbackLoop as FeedbackLoop
    from .feedback import FeedbackRecord as FeedbackRecord
    from .feedback import FeedbackSummary as FeedbackSummary
    from .feedback import InMemoryFeedbackStore as InMemoryFeedbackStore
    from .feedback import JSONLFeedbackStore as JSONLFeedbackStore
    from .filters import metadata_matches as metadata_matches
    from .flare import FLAREPipeline as FLAREPipeline
    from .graph import GraphRAGEngine as GraphRAGEngine
    from .graph import GraphRAGRetriever as GraphRAGRetriever
    from .graph import KnowledgeGraph as KnowledgeGraph
    from .graph import KnowledgeTriple as KnowledgeTriple
    from .hybrid_search import HybridSearchRetriever as HybridSearchRetriever
    from .hyde import HyDERetriever as HyDERetriever
    from .hyqe import HyQERetriever as HyQERetriever
    from .indexing import IndexedFile as IndexedFile
    from .indexing import IndexManifest as IndexManifest
    from .indexing import IndexOptions as IndexOptions
    from .indexing import IndexPlan as IndexPlan
    from .indexing import index_from_config as index_from_config
    from .indexing import index_path as index_path
    from .indexing import inspect_index as inspect_index
    from .indexing import load_manifest as load_manifest
    from .indexing import plan_incremental_update as plan_incremental_update
    from .indexing import save_manifest as save_manifest
    from .indexing import scan_indexable_files as scan_indexable_files
    from .ingestion import CodeChunker as CodeChunker
    from .ingestion import HierarchicalChunker as HierarchicalChunker
    from .ingestion import HTMLSectionChunker as HTMLSectionChunker
    from .ingestion import MarkdownHeaderChunker as MarkdownHeaderChunker
    from .ingestion import PDFLayoutChunker as PDFLayoutChunker
    from .ingestion import RecursiveTextChunker as RecursiveTextChunker
    from .ingestion import SemanticChunker as SemanticChunker
    from .ingestion import SentenceWindowChunker as SentenceWindowChunker
    from .ingestion import TableChunker as TableChunker
    from .ingestion import TextChunk as TextChunk
    from .ingestion import TokenTextChunker as TokenTextChunker
    from .ingestion import chunk_documents as chunk_documents
    from .ingestion import ingest_path as ingest_path
    from .ingestion import load_documents as load_documents
    from .llms import AnthropicClient as AnthropicClient
    from .llms import AzureOpenAIChatClient as AzureOpenAIChatClient
    from .llms import LiteLLMClient as LiteLLMClient
    from .llms import OllamaClient as OllamaClient
    from .llms import OpenAIChatClient as OpenAIChatClient
    from .mmr import MMRRetriever as MMRRetriever
    from .multihop import EvidenceHop as EvidenceHop
    from .multihop import MultiHopRAGEngine as MultiHopRAGEngine
    from .multihop import MultiHopRAGResult as MultiHopRAGResult
    from .multihop import RuleBasedQueryDecomposer as RuleBasedQueryDecomposer
    from .multimodal import CallableMultimodalEmbedding as CallableMultimodalEmbedding
    from .multimodal import CLIPMultimodalEmbedding as CLIPMultimodalEmbedding
    from .multimodal import Modality as Modality
    from .multimodal import MultimodalDocument as MultimodalDocument
    from .multimodal import MultimodalEmbeddingModel as MultimodalEmbeddingModel
    from .multimodal import MultimodalQuery as MultimodalQuery
    from .multimodal import MultimodalRAGEngine as MultimodalRAGEngine
    from .multimodal import MultimodalRetriever as MultimodalRetriever
    from .parent_document import ParentDocumentRetriever as ParentDocumentRetriever
    from .pipeline import AdvancedRAGPipeline as AdvancedRAGPipeline
    from .presets import production_hybrid_rag as production_hybrid_rag
    from .presets import simple_rag as simple_rag
    from .presets import strict_rag as strict_rag
    from .presets import vector_rag as vector_rag
    from .propositional import PropositionalRetriever as PropositionalRetriever
    from .query import IdentityQueryTransformer as IdentityQueryTransformer
    from .query import MultiQueryTransformer as MultiQueryTransformer
    from .query import QueryTransformer as QueryTransformer
    from .query import StepBackQueryTransformer as StepBackQueryTransformer
    from .query_decomposition import QueryDecompositionRetriever as QueryDecompositionRetriever
    from .rag_fusion import RAGFusionRetriever as RAGFusionRetriever
    from .raptor import RAPTORRetriever as RAPTORRetriever
    from .raptor_engine import RAPTOREngine as RAPTOREngine
    from .raptor_engine import RAPTORIndex as RAPTORIndex
    from .raptor_engine import RAPTORNode as RAPTORNode
    from .raptor_engine import RAPTORRetrieverV2 as RAPTORRetrieverV2
    from .reranking import BaseReranker as BaseReranker
    from .reranking import CohereReranker as CohereReranker
    from .reranking import CrossEncoderReranker as CrossEncoderReranker
    from .reranking import KeywordOverlapReranker as KeywordOverlapReranker
    from .reranking import ReciprocalRankFusionReranker as ReciprocalRankFusionReranker
    from .reranking import RerankingConfig as RerankingConfig
    from .reranking import RerankingRetriever as RerankingRetriever
    from .reranking import build_reranker as build_reranker
    from .retrieval import ColBERTRetriever as ColBERTRetriever
    from .retrieval import LearnedSparseRetriever as LearnedSparseRetriever
    from .retrieval import ParentChildIndex as ParentChildIndex
    from .retrieval import ParentChildRetriever as ParentChildRetriever
    from .retrieval import SentenceTransformerTokenEncoder as SentenceTransformerTokenEncoder
    from .retrieval import SPLADERetriever as SPLADERetriever
    from .router import EnsembleRetriever as EnsembleRetriever
    from .router import QueryRouter as LegacyRetrieverQueryRouter  # noqa: F401
    from .routing import KeywordIntentClassifier as KeywordIntentClassifier
    from .routing import QueryRouter as QueryRouter
    from .routing import RouteDecision as RouteDecision
    from .routing import RoutedResponse as RoutedResponse
    from .routing import RouteRule as RouteRule
    from .routing import RuleBasedQueryClassifier as RuleBasedQueryClassifier
    from .schema import Chunk as Chunk
    from .schema import EmbeddingProtocol as EmbeddingProtocol
    from .schema import LLMProtocol as LLMProtocol
    from .schema import RAGResponse as RAGResponse
    from .schema import RerankerProtocol as RerankerProtocol
    from .schema import RetrieverProtocol as RetrieverProtocol
    from .schema import Source as Source
    from .security import AccessControlledRAGEngine as AccessControlledRAGEngine
    from .security import AccessControlledRetriever as AccessControlledRetriever
    from .security import AccessDecision as AccessDecision
    from .security import AccessPolicy as AccessPolicy
    from .security import Principal as Principal
    from .security import filter_documents_for_principal as filter_documents_for_principal
    from .self_query import SelfQueryRetriever as SelfQueryRetriever
    from .self_rag import AlwaysRetrieveGate as AlwaysRetrieveGate
    from .self_rag import EvidenceCritic as EvidenceCritic
    from .self_rag import EvidenceRelevance as EvidenceRelevance
    from .self_rag import LexicalEvidenceCritic as LexicalEvidenceCritic
    from .self_rag import RelevanceAssessment as RelevanceAssessment
    from .self_rag import RetrievalDecision as RetrievalDecision
    from .self_rag import RetrievalGate as RetrievalGate
    from .self_rag import ScriptedEvidenceCritic as ScriptedEvidenceCritic
    from .self_rag import SelfRAGEngine as SelfRAGEngine
    from .self_rag import SelfRAGIteration as SelfRAGIteration
    from .self_rag import SelfRAGResult as SelfRAGResult
    from .self_rag import SelfRAGTrace as SelfRAGTrace
    from .self_rag import StaticRetrievalGate as StaticRetrievalGate
    from .self_rag import SupportAssessment as SupportAssessment
    from .semantic_chunker import SemanticChunker as LegacySemanticChunker  # noqa: F401
    from .sentence_window import SentenceWindowRetriever as SentenceWindowRetriever
    from .sentence_window import split_sentences as split_sentences
    from .step_back import StepBackRetriever as StepBackRetriever
    from .structured import SQLExecutionResult as SQLExecutionResult
    from .structured import SQLGenerationResult as SQLGenerationResult
    from .structured import SQLRAGEngine as SQLRAGEngine
    from .structured import StructuredRAG as StructuredRAG
    from .structured import TableSchema as TableSchema
    from .tenancy import CollectionBinding as CollectionBinding
    from .tenancy import MultiTenantRAGEngine as MultiTenantRAGEngine
    from .tenancy import TenantConfig as TenantConfig
    from .tenancy import TenantRegistry as TenantRegistry
    from .tokenization import RetrievalTokenizer as RetrievalTokenizer
    from .tokenization import tokenize as tokenize
    from .tracing import RAGTrace as RAGTrace
    from .tracing import RAGTraceStep as RAGTraceStep
    from .tracing import append_trace_jsonl as append_trace_jsonl
    from .tracing import estimate_tokens as estimate_tokens
    from .vectorstores import ChromaRetriever as ChromaRetriever
    from .vectorstores import ChromaVectorStore as ChromaVectorStore
    from .vectorstores import FaissRetriever as FaissRetriever
    from .vectorstores import FaissVectorStore as FaissVectorStore
    from .vectorstores import MemoryVectorStore as MemoryVectorStore
    from .vectorstores import QdrantRetriever as QdrantRetriever
    from .vectorstores import QdrantVectorStore as QdrantVectorStore
    from .vectorstores import VectorStoreRetriever as VectorStoreRetriever
    from .workflow import CompressNode as CompressNode
    from .workflow import FunctionNode as FunctionNode
    from .workflow import GenerateNode as GenerateNode
    from .workflow import RAGWorkflow as RAGWorkflow
    from .workflow import RetrieveNode as RetrieveNode
    from .workflow import TransformQueryNode as TransformQueryNode
    from .workflow import WorkflowResult as WorkflowResult

__version__ = "1.1.0"

_LAZY_EXPORTS = {
    "embedder_fingerprint": (".cache", "embedder_fingerprint"),
    "hash_documents": (".cache", "hash_documents"),
    "load_cache": (".cache", "load_cache"),
    "save_cache": (".cache", "save_cache"),
    "CacheBackend": (".cache", "CacheBackend"),
    "CacheEntry": (".cache", "CacheEntry"),
    "CacheStats": (".cache", "CacheStats"),
    "CacheSerializerError": (".cache", "CacheSerializerError"),
    "MemoryCache": (".cache", "MemoryCache"),
    "SQLiteCache": (".cache", "SQLiteCache"),
    "RedisCache": (".cache.redis", "RedisCache"),
    "build_cache_backend": (".cache", "build_cache_backend"),
    "make_cache_key": (".cache", "make_cache_key"),
    "CachedEmbeddingModel": (".cache", "CachedEmbeddingModel"),
    "CachedRetriever": (".cache", "CachedRetriever"),
    "CachedReranker": (".cache", "CachedReranker"),
    "CachedLLMClient": (".cache", "CachedLLMClient"),
    "cached_call": (".cache", "cached_call"),
    "cache_method": (".cache", "cache_method"),
    "cache_embedding_model": (".cache", "cache_embedding_model"),
    "cache_retriever": (".cache", "cache_retriever"),
    "cache_reranker": (".cache", "cache_reranker"),
    "cache_llm_client": (".cache", "cache_llm_client"),
    "CitationValidationResult": (".citations", "CitationValidationResult"),
    "extract_citations": (".citations", "extract_citations"),
    "validate_citations": (".citations", "validate_citations"),
    "citation_location": (".citations", "citation_location"),
    "ContextCompressor": (".compression", "ContextCompressor"),
    "CompressionPipeline": (".compression", "CompressionPipeline"),
    "ExtractiveContextCompressor": (".compression", "ExtractiveContextCompressor"),
    "RedundancyFilter": (".compression", "RedundancyFilter"),
    "AzureOpenAIEmbedding": (".embeddings", "AzureOpenAIEmbedding"),
    "CohereEmbedding": (".embeddings", "CohereEmbedding"),
    "OpenAIEmbedding": (".embeddings", "OpenAIEmbedding"),
    "VoyageEmbedding": (".embeddings", "VoyageEmbedding"),
    "RAGEngine": (".engine", "RAGEngine"),
    "RAGStream": (".engine", "RAGStream"),
    "RAGResponse": (".schema", "RAGResponse"),
    "Source": (".schema", "Source"),
    "Chunk": (".schema", "Chunk"),
    "RetrieverProtocol": (".schema", "RetrieverProtocol"),
    "EmbeddingProtocol": (".schema", "EmbeddingProtocol"),
    "LLMProtocol": (".schema", "LLMProtocol"),
    "RerankerProtocol": (".schema", "RerankerProtocol"),
    "GenerationEvaluationResult": (".evaluation", "GenerationEvaluationResult"),
    "RetrievalEvaluationResult": (".evaluation", "RetrievalEvaluationResult"),
    "RetrievalExample": (".evaluation", "RetrievalExample"),
    "recall_at_k": (".evaluation", "recall_at_k"),
    "ndcg_at_k": (".evaluation", "ndcg_at_k"),
    "context_precision_at_k": (".evaluation", "context_precision_at_k"),
    "evaluate_generation": (".evaluation", "evaluate_generation"),
    "evaluate_pipeline": (".evaluation", "evaluate_pipeline"),
    "evaluate_retrieval": (".evaluation", "evaluate_retrieval"),
    "HybridSearchRetriever": (".hybrid_search", "HybridSearchRetriever"),
    "RetrievalTokenizer": (".tokenization", "RetrievalTokenizer"),
    "tokenize": (".tokenization", "tokenize"),
    "metadata_matches": (".filters", "metadata_matches"),
    "IndexManifest": (".indexing", "IndexManifest"),
    "IndexedFile": (".indexing", "IndexedFile"),
    "IndexOptions": (".indexing", "IndexOptions"),
    "IndexPlan": (".indexing", "IndexPlan"),
    "scan_indexable_files": (".indexing", "scan_indexable_files"),
    "plan_incremental_update": (".indexing", "plan_incremental_update"),
    "index_from_config": (".indexing", "index_from_config"),
    "index_path": (".indexing", "index_path"),
    "inspect_index": (".indexing", "inspect_index"),
    "load_manifest": (".indexing", "load_manifest"),
    "save_manifest": (".indexing", "save_manifest"),
    "HTMLSectionChunker": (".ingestion", "HTMLSectionChunker"),
    "MarkdownHeaderChunker": (".ingestion", "MarkdownHeaderChunker"),
    "RecursiveTextChunker": (".ingestion", "RecursiveTextChunker"),
    "TextChunk": (".ingestion", "TextChunk"),
    "SentenceWindowChunker": (".ingestion", "SentenceWindowChunker"),
    "SemanticChunker": (".ingestion", "SemanticChunker"),
    "CodeChunker": (".ingestion", "CodeChunker"),
    "TableChunker": (".ingestion", "TableChunker"),
    "PDFLayoutChunker": (".ingestion", "PDFLayoutChunker"),
    "HierarchicalChunker": (".ingestion", "HierarchicalChunker"),
    "TokenTextChunker": (".ingestion", "TokenTextChunker"),
    "chunk_documents": (".ingestion", "chunk_documents"),
    "ingest_path": (".ingestion", "ingest_path"),
    "load_documents": (".ingestion", "load_documents"),
    "AnthropicClient": (".llms", "AnthropicClient"),
    "AzureOpenAIChatClient": (".llms", "AzureOpenAIChatClient"),
    "LiteLLMClient": (".llms", "LiteLLMClient"),
    "OllamaClient": (".llms", "OllamaClient"),
    "OpenAIChatClient": (".llms", "OpenAIChatClient"),
    "AdvancedRAGPipeline": (".pipeline", "AdvancedRAGPipeline"),
    "IdentityQueryTransformer": (".query", "IdentityQueryTransformer"),
    "MultiQueryTransformer": (".query", "MultiQueryTransformer"),
    "QueryTransformer": (".query", "QueryTransformer"),
    "StepBackQueryTransformer": (".query", "StepBackQueryTransformer"),
    "BaseReranker": (".reranking", "BaseReranker"),
    "CohereReranker": (".reranking", "CohereReranker"),
    "CrossEncoderReranker": (".reranking", "CrossEncoderReranker"),
    "KeywordOverlapReranker": (".reranking", "KeywordOverlapReranker"),
    "ReciprocalRankFusionReranker": (".reranking", "ReciprocalRankFusionReranker"),
    "RerankingConfig": (".reranking", "RerankingConfig"),
    "RerankingRetriever": (".reranking", "RerankingRetriever"),
    "build_reranker": (".reranking", "build_reranker"),
    "RAGTrace": (".tracing", "RAGTrace"),
    "RAGTraceStep": (".tracing", "RAGTraceStep"),
    "estimate_tokens": (".tracing", "estimate_tokens"),
    "append_trace_jsonl": (".tracing", "append_trace_jsonl"),
    "ChromaRetriever": (".vectorstores", "ChromaRetriever"),
    "ChromaVectorStore": (".vectorstores", "ChromaVectorStore"),
    "FaissRetriever": (".vectorstores", "FaissRetriever"),
    "FaissVectorStore": (".vectorstores", "FaissVectorStore"),
    "MemoryVectorStore": (".vectorstores", "MemoryVectorStore"),
    "QdrantRetriever": (".vectorstores", "QdrantRetriever"),
    "QdrantVectorStore": (".vectorstores", "QdrantVectorStore"),
    "VectorStoreRetriever": (".vectorstores", "VectorStoreRetriever"),
    "LearnedSparseRetriever": (".retrieval", "LearnedSparseRetriever"),
    "SPLADERetriever": (".retrieval", "SPLADERetriever"),
    "SentenceTransformerTokenEncoder": (".retrieval", "SentenceTransformerTokenEncoder"),
    "ColBERTRetriever": (".retrieval", "ColBERTRetriever"),
    "Modality": (".multimodal", "Modality"),
    "MultimodalDocument": (".multimodal", "MultimodalDocument"),
    "MultimodalQuery": (".multimodal", "MultimodalQuery"),
    "MultimodalEmbeddingModel": (".multimodal", "MultimodalEmbeddingModel"),
    "CallableMultimodalEmbedding": (".multimodal", "CallableMultimodalEmbedding"),
    "CLIPMultimodalEmbedding": (".multimodal", "CLIPMultimodalEmbedding"),
    "MultimodalRetriever": (".multimodal", "MultimodalRetriever"),
    "MultimodalRAGEngine": (".multimodal", "MultimodalRAGEngine"),
    "TechniqueFamily": (".catalog", "TechniqueFamily"),
    "TechniqueStatus": (".catalog", "TechniqueStatus"),
    "TechniqueSpec": (".catalog", "TechniqueSpec"),
    "TECHNIQUES": (".catalog", "TECHNIQUES"),
    "get_technique": (".catalog", "get_technique"),
    "list_techniques": (".catalog", "list_techniques"),
    "SelfRAGEngine": (".self_rag", "SelfRAGEngine"),
    "SelfRAGIteration": (".self_rag", "SelfRAGIteration"),
    "SelfRAGResult": (".self_rag", "SelfRAGResult"),
    "SelfRAGTrace": (".self_rag", "SelfRAGTrace"),
    "RetrievalDecision": (".self_rag", "RetrievalDecision"),
    "RetrievalGate": (".self_rag", "RetrievalGate"),
    "StaticRetrievalGate": (".self_rag", "StaticRetrievalGate"),
    "AlwaysRetrieveGate": (".self_rag", "AlwaysRetrieveGate"),
    "EvidenceCritic": (".self_rag", "EvidenceCritic"),
    "EvidenceRelevance": (".self_rag", "EvidenceRelevance"),
    "LexicalEvidenceCritic": (".self_rag", "LexicalEvidenceCritic"),
    "ScriptedEvidenceCritic": (".self_rag", "ScriptedEvidenceCritic"),
    "RelevanceAssessment": (".self_rag", "RelevanceAssessment"),
    "SupportAssessment": (".self_rag", "SupportAssessment"),
    "AgenticRAGEngine": (".agentic", "AgenticRAGEngine"),
    "AgenticRAGResult": (".agentic", "AgenticRAGResult"),
    "AgentTrace": (".agentic", "AgentTrace"),
    "AgentStep": (".agentic", "AgentStep"),
    "AgentAction": (".agentic", "AgentAction"),
    "AgentPlanner": (".agentic", "AgentPlanner"),
    "AgentTool": (".agentic", "AgentTool"),
    "ToolSpec": (".agentic", "ToolSpec"),
    "ToolObservation": (".agentic", "ToolObservation"),
    "ToolRegistry": (".agentic", "ToolRegistry"),
    "RetrievalToolAdapter": (".agentic", "RetrievalToolAdapter"),
    "ScriptedPlanner": (".agentic", "ScriptedPlanner"),
    "LLMJSONPlanner": (".agentic", "LLMJSONPlanner"),

    "ParentChildIndex": (".retrieval", "ParentChildIndex"),
    "ParentChildRetriever": (".retrieval", "ParentChildRetriever"),
    "CorrectiveRAGEngine": (".corrective", "CorrectiveRAGEngine"),
    "CorrectiveRAGResult": (".corrective", "CorrectiveRAGResult"),
    "RetrievalGrade": (".corrective", "RetrievalGrade"),
    "LexicalRetrievalGrader": (".corrective", "LexicalRetrievalGrader"),
    "ConversationalRAGEngine": (".conversation", "ConversationalRAGEngine"),
    "ConversationTurn": (".conversation", "ConversationTurn"),
    "InMemoryConversationStore": (".conversation", "InMemoryConversationStore"),
    "RAGWorkflow": (".workflow", "RAGWorkflow"),
    "WorkflowResult": (".workflow", "WorkflowResult"),
    "FunctionNode": (".workflow", "FunctionNode"),
    "RetrieveNode": (".workflow", "RetrieveNode"),
    "GenerateNode": (".workflow", "GenerateNode"),
    "TransformQueryNode": (".workflow", "TransformQueryNode"),
    "CompressNode": (".workflow", "CompressNode"),
    "HyDERetriever": (".hyde", "HyDERetriever"),
    "RAGFusionRetriever": (".rag_fusion", "RAGFusionRetriever"),
    "ParentDocumentRetriever": (".parent_document", "ParentDocumentRetriever"),
    "SelfQueryRetriever": (".self_query", "SelfQueryRetriever"),
    "ContextualCompressionRetriever": (".contextual_compression", "ContextualCompressionRetriever"),
    "QueryDecompositionRetriever": (".query_decomposition", "QueryDecompositionRetriever"),
    "StepBackRetriever": (".step_back", "StepBackRetriever"),
    "MMRRetriever": (".mmr", "MMRRetriever"),
    "CorrectiveRAGRetriever": (".corrective_rag", "CorrectiveRAGRetriever"),
    "DocQuality": (".corrective_rag", "DocQuality"),
    "SentenceWindowRetriever": (".sentence_window", "SentenceWindowRetriever"),
    "split_sentences": (".sentence_window", "split_sentences"),
    "HyQERetriever": (".hyqe", "HyQERetriever"),
    "LegacySemanticChunker": (".semantic_chunker", "SemanticChunker"),
    "QueryRouter": (".routing", "QueryRouter"),
    "RoutedResponse": (".routing", "RoutedResponse"),
    "RouteDecision": (".routing", "RouteDecision"),
    "RuleBasedQueryClassifier": (".routing", "RuleBasedQueryClassifier"),
    "KeywordIntentClassifier": (".routing", "KeywordIntentClassifier"),
    "RouteRule": (".routing", "RouteRule"),
    "LegacyRetrieverQueryRouter": (".router", "QueryRouter"),
    "EnsembleRetriever": (".router", "EnsembleRetriever"),
    "MultiHopRAGEngine": (".multihop", "MultiHopRAGEngine"),
    "MultiHopRAGResult": (".multihop", "MultiHopRAGResult"),
    "EvidenceHop": (".multihop", "EvidenceHop"),
    "RuleBasedQueryDecomposer": (".multihop", "RuleBasedQueryDecomposer"),
    "GraphRAGEngine": (".graph", "GraphRAGEngine"),
    "GraphRAGRetriever": (".graph", "GraphRAGRetriever"),
    "KnowledgeGraph": (".graph", "KnowledgeGraph"),
    "KnowledgeTriple": (".graph", "KnowledgeTriple"),
    "RAPTOREngine": (".raptor_engine", "RAPTOREngine"),
    "RAPTORIndex": (".raptor_engine", "RAPTORIndex"),
    "RAPTORNode": (".raptor_engine", "RAPTORNode"),
    "RAPTORRetrieverV2": (".raptor_engine", "RAPTORRetrieverV2"),
    "FederatedRAGEngine": (".federated", "FederatedRAGEngine"),
    "FederatedRAGResult": (".federated", "FederatedRAGResult"),
    "FederatedRetriever": (".federated", "FederatedRetriever"),
    "FederatedSourceResult": (".federated", "FederatedSourceResult"),
    "SQLRAGEngine": (".structured", "SQLRAGEngine"),
    "StructuredRAG": (".structured", "StructuredRAG"),
    "TableSchema": (".structured", "TableSchema"),
    "SQLGenerationResult": (".structured", "SQLGenerationResult"),
    "SQLExecutionResult": (".structured", "SQLExecutionResult"),
    "Principal": (".security", "Principal"),
    "AccessDecision": (".security", "AccessDecision"),
    "AccessPolicy": (".security", "AccessPolicy"),
    "AccessControlledRetriever": (".security", "AccessControlledRetriever"),
    "AccessControlledRAGEngine": (".security", "AccessControlledRAGEngine"),
    "filter_documents_for_principal": (".security", "filter_documents_for_principal"),
    "TenantConfig": (".tenancy", "TenantConfig"),
    "CollectionBinding": (".tenancy", "CollectionBinding"),
    "TenantRegistry": (".tenancy", "TenantRegistry"),
    "MultiTenantRAGEngine": (".tenancy", "MultiTenantRAGEngine"),
    "FeedbackRecord": (".feedback", "FeedbackRecord"),
    "FeedbackSummary": (".feedback", "FeedbackSummary"),
    "InMemoryFeedbackStore": (".feedback", "InMemoryFeedbackStore"),
    "JSONLFeedbackStore": (".feedback", "JSONLFeedbackStore"),
    "FeedbackLoop": (".feedback", "FeedbackLoop"),
    "RAPTORRetriever": (".raptor", "RAPTORRetriever"),
    "FLAREPipeline": (".flare", "FLAREPipeline"),
    "PropositionalRetriever": (".propositional", "PropositionalRetriever"),
    "ChainOfNoteRetriever": (".chain_of_note", "ChainOfNoteRetriever"),
    "AdaptiveRetriever": (".adaptive", "AdaptiveRetriever"),
    "production_hybrid_rag": (".presets", "production_hybrid_rag"),
    "simple_rag": (".presets", "simple_rag"),
    "vector_rag": (".presets", "vector_rag"),
    "strict_rag": (".presets", "strict_rag"),
    "GateDecision": (".adaptive", "GateDecision"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


__all__ = [
    "Document",
    "BaseRetriever",
    "LLMClient",
    "EmbeddingModel",
    "HashingEmbedding",
    "SentenceTransformerEmbedding",
    "CallableLLMClient",
    "OpenAILLMClient",
    "StaticLLMClient",
    "ExtractiveLLMClient",
    *_LAZY_EXPORTS.keys(),
]


def __dir__() -> list[str]:
    """Expose lazy public names to introspection and completion tools."""

    return sorted(set(globals()) | set(__all__))
