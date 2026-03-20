"""
Short-Term Memory Service
Provides working memory window for immediate context recall.
Similar to Cursor, Devin, Replit Agent working memory.

PATCH #7: Episodic immediate context recall.
"""

from __future__ import annotations

from typing import List, Dict


def extract_short_term_window(messages: List[Dict], window: int = 5) -> List[Dict]:
    """
    Extract the short-term memory window.
    Returns the last N user + assistant messages (cleaned).
    
    This is the "working memory" - what just happened in the conversation.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        window: Number of recent messages to include (default: 5)
    
    Returns:
        List of cleaned messages (role + content only)
    """
    if not messages:
        return []
    
    # Get the last N messages
    recent = messages[-window:] if len(messages) > window else messages
    
    # Clean messages: keep only role + content
    cleaned = []
    for msg in recent:
        if isinstance(msg, dict) and msg.get("content"):
            cleaned.append({
                "role": msg.get("role", "user"),
                "content": str(msg.get("content", ""))
            })
    
    return cleaned


def summarize_episode(messages: List[Dict]) -> str:
    """
    Summarize short-term memory into 1–2 sentences.
    This is the 'episodic memory' summary.
    
    Args:
        messages: List of recent messages
    
    Returns:
        Episodic summary string
    """
    if not messages:
        return ""
    
    # Build conversation text
    conversation_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))[:150]  # Limit each message
        conversation_parts.append(f"{role}: {content}")
    
    text = " | ".join(conversation_parts)
    
    # Create summary
    if len(text) > 350:
        text = text[:350] + "..."
    
    summary = f"Episodic summary: The recent conversation is about {text}"
    
    return summary

