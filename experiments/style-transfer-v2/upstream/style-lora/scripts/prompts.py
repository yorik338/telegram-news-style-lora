"""Prompts shared across the pipeline.

STYLE_SYSTEM is the system prompt used both to build training examples and at
inference time; the two must match for the adapter to trigger reliably.
"""

STYLE_SYSTEM = "Rewrite the given text in the author's voice."

NEUTRALIZE_SYSTEM = (
    "You rewrite text into plain, neutral, modern standard English prose. "
    "Preserve the full meaning, events, and details. Remove all stylistic "
    "character: dialect, idiosyncratic spelling, regionalisms, distinctive "
    "rhythm. Output only the rewritten text, nothing else."
)
