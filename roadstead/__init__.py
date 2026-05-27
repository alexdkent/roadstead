"""Centralized LLM scheduler proxy.

All LLM traffic in the Collective routes through this service.  It
implements deficit-round-robin scheduling with priority bands,
duration-weighted cost accounting, and concurrency-aware admission.
"""
