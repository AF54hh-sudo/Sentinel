"""Deterministic Markdown extraction and heading/paragraph chunking."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import yaml

from sentinel.rag.schemas import DocumentChunk, DocumentMetadata, SourceDocument

FRONT_MATTER_PATTERN = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
SUPPORTED_SUFFIXES = {".md", ".markdown"}
DEFAULT_TARGET_TOKENS = 550
DEFAULT_MAX_TOKENS = 700


class DocumentIngestionError(ValueError):
    """Raised when a source document violates the corpus contract."""


def estimate_tokens(text: str) -> int:
    """Return a deterministic approximation suitable for small English documents."""
    return max(1, math.ceil(len(text) / 4))


def clean_document_text(text: str) -> str:
    """Normalize line endings and horizontal whitespace while preserving paragraphs."""
    lines = [WHITESPACE_PATTERN.sub(" ", line).strip() for line in text.splitlines()]
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        cleaned.append(line)
        previous_blank = is_blank
    return "\n".join(cleaned).strip()


def _parse_front_matter(path: Path, raw_text: str) -> tuple[dict[str, Any], str]:
    match = FRONT_MATTER_PATTERN.match(raw_text.replace("\r\n", "\n"))
    if match is None:
        raise DocumentIngestionError(f"{path.name} must begin with YAML front matter")
    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError as error:
        raise DocumentIngestionError(f"{path.name} has invalid YAML front matter") from error
    if not isinstance(metadata, dict):
        raise DocumentIngestionError(f"{path.name} front matter must be a mapping")
    return metadata, clean_document_text(match.group(2))


def load_document(path: Path, *, corpus_root: Path | None = None) -> SourceDocument:
    """Load one UTF-8 Markdown document and validate its required metadata."""
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise DocumentIngestionError(f"Unsupported document format: {path.suffix}")
    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise DocumentIngestionError(f"{path.name} must be UTF-8 encoded") from error
    raw_metadata, content = _parse_front_matter(path, raw_text)
    root = corpus_root or path.parent
    try:
        source_path = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        source_path = path.name
    try:
        metadata = DocumentMetadata.model_validate(
            {**raw_metadata, "source_path": source_path}
        )
    except ValueError as error:
        raise DocumentIngestionError(f"{path.name} metadata is invalid: {error}") from error
    if not content:
        raise DocumentIngestionError(f"{path.name} contains no document text")
    return SourceDocument(metadata=metadata, content=content)


def load_corpus(documents_dir: Path) -> list[SourceDocument]:
    """Load the allowlisted Markdown corpus in a stable order."""
    if not documents_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {documents_dir}")
    paths = sorted(
        path
        for path in documents_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and path.name.lower() != "readme.md"
    )
    if not paths:
        raise DocumentIngestionError(f"No supported documents found in {documents_dir}")
    documents = [load_document(path, corpus_root=documents_dir) for path in paths]
    ids = [document.metadata.document_id for document in documents]
    if len(ids) != len(set(ids)):
        raise DocumentIngestionError("Corpus document IDs must be unique")
    return documents


def _sections(content: str, fallback_title: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_heading = fallback_title
    current_lines: list[str] = []
    for line in content.splitlines():
        heading = HEADING_PATTERN.match(line)
        if heading:
            body = clean_document_text("\n".join(current_lines))
            if body:
                sections.append((current_heading, body.split("\n\n")))
            current_heading = heading.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
    body = clean_document_text("\n".join(current_lines))
    if body:
        sections.append((current_heading, body.split("\n\n")))
    return sections


def _section_chunks(
    title: str,
    section: str,
    paragraphs: list[str],
    *,
    target_tokens: int,
    max_tokens: int,
) -> list[str]:
    prefix = f"Document: {title}\nSection: {section}\n\n"
    results: list[str] = []
    current: list[str] = []
    for paragraph in paragraphs:
        candidate = prefix + "\n\n".join([*current, paragraph])
        if current and estimate_tokens(candidate) > max_tokens:
            results.append(prefix + "\n\n".join(current))
            current = [paragraph]
        else:
            current.append(paragraph)
        if estimate_tokens(prefix + "\n\n".join(current)) >= target_tokens:
            results.append(prefix + "\n\n".join(current))
            current = []
    if current:
        remainder = prefix + "\n\n".join(current)
        if results and estimate_tokens(remainder) < target_tokens // 2:
            merged = results[-1] + "\n\n" + "\n\n".join(current)
            if estimate_tokens(merged) <= max_tokens:
                results[-1] = merged
            else:
                results.append(remainder)
        else:
            results.append(remainder)
    return results


def chunk_document(
    document: SourceDocument,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[DocumentChunk]:
    """Split a document on headings and paragraph boundaries with stable IDs."""
    if target_tokens < 1 or max_tokens < target_tokens:
        raise ValueError("chunk token bounds are invalid")
    pieces: list[tuple[str, str]] = []
    for section, paragraphs in _sections(document.content, document.metadata.title):
        for content in _section_chunks(
            document.metadata.title,
            section,
            paragraphs,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
        ):
            pieces.append((section, content))
    chunks: list[DocumentChunk] = []
    for index, (section, content) in enumerate(pieces, start=1):
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.metadata.document_id}-C{index:03d}",
                document_id=document.metadata.document_id,
                document_title=document.metadata.title,
                document_date=document.metadata.document_date,
                category=document.metadata.category,
                regions=document.metadata.regions,
                topics=document.metadata.topics,
                source_path=document.metadata.source_path,
                section=section,
                page=1,
                chunk_index=index,
                content=content,
                content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                estimated_tokens=estimate_tokens(content),
            )
        )
    if not chunks:
        raise DocumentIngestionError(
            f"{document.metadata.source_path} produced no retrievable chunks"
        )
    return chunks


def chunk_corpus(
    documents: list[SourceDocument],
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[DocumentChunk]:
    """Chunk all documents and enforce corpus-wide stable-ID uniqueness."""
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(
            document,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
        )
    ]
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise DocumentIngestionError("Corpus chunk IDs must be unique")
    return chunks
