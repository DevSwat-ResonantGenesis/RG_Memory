"""
Memory Summarization Service
=============================

Compresses and summarizes older memories to:
1. Reduce storage costs
2. Improve retrieval efficiency
3. Maintain long-term context without overwhelming context windows

Author: Resonant Genesis Team
Date: December 29, 2025
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SummarizationResult:
    """Result of memory summarization."""
    success: bool
    original_length: int
    summary_length: int
    compression_ratio: float
    summary: str
    key_points: List[str]
    error: Optional[str] = None


@dataclass
class SummarizationConfig:
    """Configuration for memory summarization."""
    enabled: bool = True
    min_content_length: int = 500          # Only summarize content > 500 chars
    target_summary_length: int = 200       # Target summary length
    max_memories_per_batch: int = 10       # Max memories to summarize at once
    summarize_after_days: int = 7          # Summarize memories older than 7 days
    preserve_key_points: bool = True       # Extract and preserve key points
    llm_provider: str = "groq"             # Default LLM for summarization


class MemorySummarizationService:
    """
    Summarizes and compresses memories for efficient storage and retrieval.
    
    Features:
    - Automatic summarization of old memories
    - Key point extraction
    - Batch summarization for efficiency
    - Preserves semantic meaning while reducing size
    """
    
    def __init__(self):
        self.config = SummarizationConfig(
            enabled=os.getenv("MEMORY_SUMMARIZATION_ENABLED", "true").lower() == "true",
            min_content_length=int(os.getenv("MEMORY_MIN_SUMMARIZE_LENGTH", "500")),
            target_summary_length=int(os.getenv("MEMORY_TARGET_SUMMARY_LENGTH", "200")),
            summarize_after_days=int(os.getenv("MEMORY_SUMMARIZE_AFTER_DAYS", "7")),
        )
        
        self._stats = {
            "total_summarized": 0,
            "total_chars_saved": 0,
            "avg_compression_ratio": 0.0,
        }
        
        # LLM service URL
        self.llm_service_url = os.getenv("LLM_SERVICE_URL", "http://llm_service:8000")
    
    def should_summarize(self, content: str, created_at: datetime = None) -> bool:
        """Check if a memory should be summarized."""
        if not self.config.enabled:
            return False
        
        # Check content length
        if len(content) < self.config.min_content_length:
            return False
        
        # Check age if timestamp provided
        if created_at:
            age = datetime.utcnow() - created_at
            if age.days < self.config.summarize_after_days:
                return False
        
        return True
    
    async def summarize_content(
        self,
        content: str,
        context: Optional[str] = None,
        extract_key_points: bool = True,
    ) -> SummarizationResult:
        """
        Summarize a single piece of content.
        
        Args:
            content: Content to summarize
            context: Optional context about the content
            extract_key_points: Whether to extract key points
            
        Returns:
            SummarizationResult
        """
        original_length = len(content)
        
        if original_length < self.config.min_content_length:
            return SummarizationResult(
                success=False,
                original_length=original_length,
                summary_length=original_length,
                compression_ratio=1.0,
                summary=content,
                key_points=[],
                error="Content too short to summarize",
            )
        
        try:
            # Build summarization prompt
            prompt = self._build_summarization_prompt(content, context, extract_key_points)
            
            # Call LLM service
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.llm_service_url}/generate",
                    json={
                        "prompt": prompt,
                        "max_tokens": self.config.target_summary_length * 2,
                        "temperature": 0.3,  # Low temperature for consistent summaries
                    }
                )
                
                if response.status_code != 200:
                    # Fallback to simple truncation
                    return self._fallback_summarize(content)
                
                result = response.json()
                summary_text = result.get("text", "")
            
            # Parse summary and key points
            summary, key_points = self._parse_summary_response(summary_text, extract_key_points)
            
            summary_length = len(summary)
            compression_ratio = summary_length / original_length if original_length > 0 else 1.0
            
            # Update stats
            self._stats["total_summarized"] += 1
            self._stats["total_chars_saved"] += original_length - summary_length
            self._update_avg_compression(compression_ratio)
            
            return SummarizationResult(
                success=True,
                original_length=original_length,
                summary_length=summary_length,
                compression_ratio=compression_ratio,
                summary=summary,
                key_points=key_points,
            )
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return self._fallback_summarize(content)
    
    def _build_summarization_prompt(
        self,
        content: str,
        context: Optional[str],
        extract_key_points: bool,
    ) -> str:
        """Build the summarization prompt."""
        prompt_parts = [
            "Summarize the following memory content concisely while preserving the key information.",
            f"Target length: approximately {self.config.target_summary_length} characters.",
        ]
        
        if context:
            prompt_parts.append(f"Context: {context}")
        
        if extract_key_points:
            prompt_parts.append("Also extract 3-5 key points as bullet points.")
        
        prompt_parts.append(f"\nContent to summarize:\n{content}")
        prompt_parts.append("\nProvide the summary followed by key points (if requested):")
        
        return "\n".join(prompt_parts)
    
    def _parse_summary_response(
        self,
        response: str,
        extract_key_points: bool,
    ) -> Tuple[str, List[str]]:
        """Parse the LLM response to extract summary and key points."""
        key_points = []
        summary = response
        
        if extract_key_points and "key point" in response.lower():
            # Try to split summary and key points
            parts = response.split("Key point", 1)
            if len(parts) > 1:
                summary = parts[0].strip()
                key_points_text = "Key point" + parts[1]
                
                # Extract bullet points
                for line in key_points_text.split("\n"):
                    line = line.strip()
                    if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                        key_points.append(line.lstrip("-•* "))
                    elif line and len(line) < 200:  # Short lines might be key points
                        key_points.append(line)
        
        return summary[:self.config.target_summary_length * 2], key_points[:5]
    
    def _fallback_summarize(self, content: str) -> SummarizationResult:
        """Fallback summarization using simple extraction."""
        original_length = len(content)
        
        # Extract first and last sentences
        sentences = content.replace("\n", " ").split(". ")
        
        if len(sentences) <= 3:
            summary = content[:self.config.target_summary_length]
        else:
            # Take first 2 and last sentence
            summary = ". ".join(sentences[:2]) + "... " + sentences[-1]
            if len(summary) > self.config.target_summary_length:
                summary = summary[:self.config.target_summary_length] + "..."
        
        # Extract potential key points (sentences with important keywords)
        key_points = []
        important_keywords = ["important", "key", "main", "critical", "essential", "must", "should"]
        
        for sentence in sentences[:10]:
            if any(kw in sentence.lower() for kw in important_keywords):
                key_points.append(sentence.strip())
                if len(key_points) >= 3:
                    break
        
        summary_length = len(summary)
        compression_ratio = summary_length / original_length if original_length > 0 else 1.0
        
        return SummarizationResult(
            success=True,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio,
            summary=summary,
            key_points=key_points,
            error="Used fallback summarization",
        )
    
    def _update_avg_compression(self, new_ratio: float):
        """Update running average compression ratio."""
        n = self._stats["total_summarized"]
        old_avg = self._stats["avg_compression_ratio"]
        self._stats["avg_compression_ratio"] = old_avg + (new_ratio - old_avg) / n
    
    async def summarize_batch(
        self,
        memories: List[Dict],
    ) -> List[Tuple[str, SummarizationResult]]:
        """
        Summarize a batch of memories.
        
        Args:
            memories: List of memory dicts with 'id' and 'content'
            
        Returns:
            List of (memory_id, SummarizationResult) tuples
        """
        results = []
        
        for memory in memories[:self.config.max_memories_per_batch]:
            memory_id = memory.get("id", "unknown")
            content = memory.get("content", "")
            context = memory.get("context")
            
            result = await self.summarize_content(content, context)
            results.append((memory_id, result))
        
        return results
    
    def get_stats(self) -> Dict:
        """Get summarization statistics."""
        return {
            "enabled": self.config.enabled,
            "total_summarized": self._stats["total_summarized"],
            "total_chars_saved": self._stats["total_chars_saved"],
            "avg_compression_ratio": f"{self._stats['avg_compression_ratio']:.2%}",
            "config": {
                "min_content_length": self.config.min_content_length,
                "target_summary_length": self.config.target_summary_length,
                "summarize_after_days": self.config.summarize_after_days,
            }
        }


# Global instance
memory_summarization = MemorySummarizationService()


async def summarize_memory(content: str, context: str = None) -> SummarizationResult:
    """Convenience function to summarize memory content."""
    return await memory_summarization.summarize_content(content, context)
