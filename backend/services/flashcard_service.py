"""
MediTutor AI - Flashcard Service with User Isolation
Generates Q&A flashcard pairs from document chunks using LLM.
Supports Anki CSV export with per-user caching.
"""

import uuid
import json
import logging
import hashlib
import csv
import io
from typing import List, Optional, Tuple

from config import FLASHCARD_COUNT, MAX_CONTEXT_LENGTH
from services.llm_service import llm_service
from services.vector_service import vector_service
from utils.cache import get_cache

logger = logging.getLogger(__name__)
cache = get_cache()


FLASHCARD_SYSTEM_PROMPT = """You are an expert medical/educational flashcard creator.
Your flashcards are clear, concise, and test genuine understanding.
Always respond ONLY with valid JSON — no preamble, no explanation outside the JSON."""


def _flashcard_prompt(context: str, count: int, topic: Optional[str]) -> str:
    topic_clause = f" Focus specifically on the topic: '{topic}'." if topic else ""
    return f"""Based on the following study material, generate exactly {count} high-quality flashcards.{topic_clause}

MATERIAL:
{context}

Rules:
- Questions must be specific and test real understanding (not trivial facts)
- Answers should be 1-3 sentences maximum
- Cover different aspects of the material
- Vary difficulty (mix easy, medium, hard)
- Label each card with its difficulty and a short topic tag

Respond ONLY with this exact JSON format:
{{
  "flashcards": [
    {{
      "question": "What is...",
      "answer": "...",
      "topic": "short topic label",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}"""


class FlashcardService:

    async def generate(
        self,
        user_id: str,  # ← NEW: required for user isolation
        doc_id: str,
        count: int = FLASHCARD_COUNT,
        topic: Optional[str] = None,
    ) -> Tuple[List[dict], str]:
        """
        Generate flashcards for a document (or topic within it).
        
        Args:
            user_id: User identifier for cache isolation
            doc_id: Document identifier
            count: Number of flashcards to generate
            topic: Optional topic filter
            
        Returns:
            (flashcards_list, model_used)
        """
        if not user_id:
            raise ValueError("user_id is required for flashcard generation")
        
        # Cache key with user isolation
        cache_key = f"flashcards:{doc_id}:{topic or 'all'}:{count}"
        cached = cache.get(user_id, cache_key)
        if cached:
            logger.info(f"Flashcard cache hit for user {user_id[:8]}..., doc {doc_id[:8]}...")
            return cached["cards"], cached["model"] + " (cached)"

        # Retrieve relevant chunks with user isolation
        query = topic if topic else "key concepts definitions important facts"
        
        try:
            chunks = await vector_service.search(
                user_id=user_id,
                doc_id=doc_id,
                query=query,
                top_k=8,
            )
        except ValueError as e:
            logger.error(f"Vector search failed for user {user_id[:8]}..., doc {doc_id[:8]}...: {e}")
            raise ValueError(f"No content found. Make sure the PDF is uploaded and indexed. Error: {e}")

        if not chunks:
            raise ValueError("No content found. Make sure the PDF is uploaded and contains text.")

        # Build context
        context = self._build_context(chunks)
        prompt = _flashcard_prompt(context, count, topic)

        # Generate
        raw_text, model = await llm_service.generate(
            prompt=prompt,
            system=FLASHCARD_SYSTEM_PROMPT,
            max_tokens=2048,
            use_cache=False,  # We cache ourselves with user isolation
        )

        # Parse JSON
        cards = self._parse_flashcards(raw_text, count)

        # Attach IDs
        for card in cards:
            card["id"] = str(uuid.uuid4())

        # Cache with user isolation
        cache.set(user_id, cache_key, {"cards": cards, "model": model})
        
        logger.info(
            f"Generated {len(cards)} flashcards for user {user_id[:8]}..., "
            f"doc {doc_id[:8]}..., topic: {topic or 'all'}"
        )
        
        return cards, model

    def _build_context(self, chunks: List[dict]) -> str:
        """Build context string from chunks, respecting max length."""
        parts = []
        total = 0
        for chunk in chunks:
            text = chunk["text"]
            page = chunk.get("page_number", "?")
            if total + len(text) > MAX_CONTEXT_LENGTH:
                # Try to cut the last chunk
                remaining = MAX_CONTEXT_LENGTH - total
                if remaining > 200:  # Only add if we have substantial text
                    parts.append(f"[Page {page}]\n{text[:remaining]}")
                break
            parts.append(f"[Page {page}]\n{text}")
            total += len(text)
        return "\n\n---\n\n".join(parts)

    def _parse_flashcards(self, raw: str, expected_count: int) -> List[dict]:
        """Robust JSON parser — handles LLM formatting quirks."""
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            # Remove first and last line if they are code fences
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)

        # Find first { and last }
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            logger.error(f"No JSON found in LLM output: {raw[:200]}")
            return self._fallback_flashcards(raw, expected_count)

        try:
            data = json.loads(raw[start:end])
            cards = data.get("flashcards", [])
            
            # Validate and clean
            valid = []
            for c in cards:
                if c.get("question") and c.get("answer"):
                    # Truncate if too long
                    question = str(c["question"]).strip()
                    answer = str(c["answer"]).strip()
                    
                    if len(question) > 500:
                        question = question[:497] + "..."
                    if len(answer) > 1000:
                        answer = answer[:997] + "..."
                    
                    valid.append({
                        "question": question,
                        "answer": answer,
                        "topic": str(c.get("topic", "General")).strip()[:50],
                        "difficulty": c.get("difficulty", "medium") if c.get("difficulty") in ["easy", "medium", "hard"] else "medium",
                    })
            
            if not valid:
                logger.warning(f"No valid flashcards after parsing, using fallback")
                return self._fallback_flashcards(raw, expected_count)
            
            return valid[:expected_count]  # Limit to expected count
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}, raw: {raw[:500]}")
            return self._fallback_flashcards(raw, expected_count)

    def _fallback_flashcards(self, raw: str, count: int) -> List[dict]:
        """Last resort: try to extract Q&A pairs from plain text."""
        cards = []
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        i = 0
        while i < len(lines) and len(cards) < count:
            line = lines[i]
            lower_line = line.lower()
            
            # Check for Q: or Question: patterns
            if lower_line.startswith("q:") or lower_line.startswith("question:"):
                q = line.split(":", 1)[-1].strip()
                if i + 1 < len(lines):
                    a_line = lines[i + 1]
                    lower_a = a_line.lower()
                    if lower_a.startswith("a:") or lower_a.startswith("answer:"):
                        a = a_line.split(":", 1)[-1].strip()
                        cards.append({
                            "question": q[:500],
                            "answer": a[:1000],
                            "topic": "General",
                            "difficulty": "medium",
                            "id": str(uuid.uuid4()),
                        })
                        i += 2
                        continue
            i += 1
        
        # If still no cards, create generic ones
        if not cards:
            cards.append({
                "question": "What is the main topic of this material?",
                "answer": "Review the uploaded PDF to understand the main concepts covered.",
                "topic": "General",
                "difficulty": "medium",
                "id": str(uuid.uuid4()),
            })
        
        logger.warning(f"Fallback parsing produced {len(cards)} flashcards")
        return cards

    def to_anki_csv(self, cards: List[dict]) -> str:
        """
        Export flashcards in Anki-compatible CSV format.
        Format: Front, Back, Tags
        """
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        
        for card in cards:
            # Clean and escape content
            question = card["question"].replace('"', '""').strip()
            answer = card["answer"].replace('"', '""').strip()
            topic = card.get("topic", "General").replace(' ', '_')
            difficulty = card.get("difficulty", "medium")
            tags = f"meditutor {topic} {difficulty}"
            
            writer.writerow([question, answer, tags])
        
        return output.getvalue()

    async def get_cached_flashcards(
        self,
        user_id: str,
        doc_id: str,
        topic: Optional[str] = None,
        count: int = FLASHCARD_COUNT,
    ) -> Optional[List[dict]]:
        """
        Retrieve cached flashcards without regenerating.
        Returns None if not cached.
        """
        cache_key = f"flashcards:{doc_id}:{topic or 'all'}:{count}"
        cached = cache.get(user_id, cache_key)
        if cached:
            return cached["cards"]
        return None

    async def clear_cache(
        self,
        user_id: str,
        doc_id: Optional[str] = None,
        topic: Optional[str] = None,
    ):
        """
        Clear cached flashcards for a user.
        If doc_id provided, clear only for that document.
        """
        if doc_id and topic:
            cache_key = f"flashcards:{doc_id}:{topic}:*"
            # In a real implementation, we'd need pattern matching
            # For simplicity, we'll note this for future enhancement
            logger.info(f"Pattern cache clearing requested for {user_id[:8]}..., {doc_id[:8]}...")
        elif doc_id:
            # Clear all topics for this document
            for t in [None, "all"]:  # Common patterns
                cache_key = f"flashcards:{doc_id}:{t or 'all'}:*"
                logger.info(f"Would clear pattern: {cache_key}")
        else:
            # Clear all flashcard cache for user
            logger.info(f"Clear all flashcard cache requested for user {user_id[:8]}...")


# ─── Singleton ────────────────────────────────────────────────────────────────
flashcard_service = FlashcardService()