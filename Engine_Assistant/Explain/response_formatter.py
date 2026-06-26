"""
response_formatter.py - Cleans and structures LLM output for UI
================================================================
YOUR MODULE: engine/explain/response_formatter.py
 
WHAT THIS FILE DOES:
--------------------
LLM responses are messy raw text. This file cleans, structures,
and formats them into consistent output for dashboard.
 
Think of it as the FINAL STEP before output reaches the user:
    Raw LLM text -> ResponseFormatter -> Clean structured output
 
WHAT THIS FILE CONTAINS:
    1. FormattedResponse  - structured output object
    2. ResponseFormatter  - main class, cleans all LLM output
    3. format_for_ui()    - converts to dashboard-ready format
"""
 
import re
import json
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict
 
from engine.explain.prompt_builder import MoveData, _score_to_words, _score_delta_to_severity
 
 
# =============================================================================
# OUTPUT DATA STRUCTURE
# =============================================================================
 
@dataclass
class FormattedResponse:
    """
    Clean structured output ready for Piyush's dashboard.
 
    Attributes:
        raw_text        : Original LLM response
        clean_text      : Cleaned version (no markdown artifacts)
        headline        : One line summary e.g. "Critical blunder on move 14"
        explanation     : Main explanation paragraph
        coaching_tip    : Actionable advice
        severity_label  : "BLUNDER" / "MISTAKE" / "INACCURACY" / "GOOD"
        severity_color  : "red" / "orange" / "yellow" / "green"
        move_number     : Which move
        player_color    : "White" or "Black"
        category        : blunder/mistake/inaccuracy/good/excellent
        score_drop      : Pawn drop (float)
        display_format  : "markdown" / "plain" / "json"
    """
    raw_text:       str
    clean_text:     str
    headline:       str
    explanation:    str
    coaching_tip:   str
    severity_label: str
    severity_color: str
    move_number:    int   = 0
    player_color:   str   = "White"
    category:       str   = "good"
    score_drop:     float = 0.0
    display_format: str   = "markdown"
 
    def to_dict(self) -> dict:
        return {
            "headline":       self.headline,
            "explanation":    self.explanation,
            "coaching_tip":   self.coaching_tip,
            "severity_label": self.severity_label,
            "severity_color": self.severity_color,
            "move_number":    self.move_number,
            "player_color":   self.player_color,
            "category":       self.category,
            "score_drop":     self.score_drop,
        }
 
    def to_markdown(self) -> str:
        icons = {
            "red":    "🔴",
            "orange": "🟠",
            "yellow": "🟡",
            "green":  "🟢",
        }
        icon = icons.get(self.severity_color, "⬜")
        return (
            f"{icon} **{self.headline}**\n\n"
            f"{self.explanation}\n\n"
            f"💡 **Tip:** {self.coaching_tip}"
        )
 
    def to_plain(self) -> str:
        return (
            f"{self.headline}\n"
            f"{self.explanation}\n"
            f"Tip: {self.coaching_tip}"
        )
 
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
 
 
# =============================================================================
# MAIN CLASS
# =============================================================================
 
class ResponseFormatter:
    """
    Cleans and structures LLM output for the dashboard.
 
    USAGE:
        formatter = ResponseFormatter()
 
        # From MoveExplainer output:
        raw = explainer.explain_move(data)
        formatted = formatter.format(raw.explanation, data)
        print(formatted.to_markdown())
 
    Args:
        max_length : Max characters for explanation. Truncates if longer.
        style      : "markdown" / "plain" / "json"
    """
 
    # Severity config — maps category to display properties
    SEVERITY_CONFIG = {
        "blunder":    ("BLUNDER",    "red",    "🔴"),
        "mistake":    ("MISTAKE",    "orange", "🟠"),
        "inaccuracy": ("INACCURACY", "yellow", "🟡"),
        "good":       ("GOOD MOVE",  "green",  "🟢"),
        "excellent":  ("EXCELLENT",  "green",  "⭐"),
    }
 
    def __init__(self, max_length: int = 500, style: str = "markdown"):
        self.max_length = max_length
        self.style      = style
 
    def format(self, raw_text: str, data: MoveData,
               coaching_tip: str = "") -> FormattedResponse:
        """
        Main method — takes raw LLM text and MoveData,
        returns clean FormattedResponse.
 
        Args:
            raw_text     : Raw text from LLM
            data         : MoveData object for this move
            coaching_tip : Optional coaching tip to include
 
        Returns:
            FormattedResponse: Clean structured output
        """
        # Clean the raw text
        clean = self._clean_text(raw_text)
 
        # Truncate if too long
        clean = self._truncate(clean)
 
        # Generate headline
        headline = self._make_headline(data)
 
        # Get severity display properties
        config = self.SEVERITY_CONFIG.get(
            data.category.lower(),
            ("MOVE", "green", "⬜")
        )
        severity_label = config[0]
        severity_color = config[1]
 
        # Calculate score drop
        score_drop = abs(data.score_after - data.score_before)
 
        return FormattedResponse(
            raw_text       = raw_text,
            clean_text     = clean,
            headline       = headline,
            explanation    = clean,
            coaching_tip   = coaching_tip or self._default_tip(data),
            severity_label = severity_label,
            severity_color = severity_color,
            move_number    = data.move_number,
            player_color   = data.player_color,
            category       = data.category,
            score_drop     = round(score_drop, 2),
            display_format = self.style,
        )
 
    def format_batch(
        self,
        explanations: List[str],
        move_data_list: List[MoveData],
        coaching_tips: List[str] = None
    ) -> List[FormattedResponse]:
        """
        Formats multiple explanations at once.
        Used for post-game analysis of all moves.
 
        Args:
            explanations    : List of raw LLM responses
            move_data_list  : Corresponding MoveData objects
            coaching_tips   : Optional tips per move
 
        Returns:
            List[FormattedResponse]: One formatted response per move
        """
        tips = coaching_tips or [""] * len(explanations)
        results = []
 
        for i, (text, data) in enumerate(zip(explanations, move_data_list)):
            try:
                tip = tips[i] if i < len(tips) else ""
                results.append(self.format(text, data, tip))
            except Exception as e:
                print(f"[ResponseFormatter] Warning: skipping move {i}: {e}")
 
        return results
 
    def format_for_streamlit(self, response: FormattedResponse) -> Dict:
        """
        Formats response specifically for  Streamlit dashboard.
        Returns a dict that maps directly to Streamlit components.
 
        Args:
            response : FormattedResponse object
 
        Returns:
            dict: Ready for Streamlit display
        """
        return {
            "title":     response.headline,
            "body":      response.explanation,
            "tip":       response.coaching_tip,
            "color":     response.severity_color,
            "badge":     response.severity_label,
            "move":      f"Move {response.move_number}",
            "player":    response.player_color,
            "score_drop": f"{response.score_drop:.2f} pawns",
            "markdown":  response.to_markdown(),
        }
 
    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------
 
    def _clean_text(self, text: str) -> str:
        """
        Cleans raw LLM output.
        Removes markdown artifacts, extra spaces, repeated punctuation.
        """
        if not text:
            return "No explanation available."
 
        # Remove markdown bold/italic
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*",   r"\1", text)
        text = re.sub(r"__(.+?)__",   r"\1", text)
        text = re.sub(r"_(.+?)_",     r"\1", text)
 
        # Remove markdown headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
 
        # Remove bullet points
        text = re.sub(r"^[-*•]\s+", "", text, flags=re.MULTILINE)
 
        # Remove numbered lists
        text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
 
        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)
 
        # Collapse multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
 
        # Remove leading/trailing whitespace
        text = text.strip()
 
        # Ensure ends with period
        if text and not text[-1] in ".!?":
            text += "."
 
        return text
 
    def _truncate(self, text: str) -> str:
        """
        Truncates text to max_length, cutting at sentence boundary.
        """
        if len(text) <= self.max_length:
            return text
 
        # Find last sentence end within limit
        truncated = text[:self.max_length]
        last_period = max(
            truncated.rfind("."),
            truncated.rfind("!"),
            truncated.rfind("?")
        )
 
        if last_period > self.max_length * 0.5:
            return truncated[:last_period + 1]
 
        return truncated + "..."
 
    def _make_headline(self, data: MoveData) -> str:
        """
        Creates a one-line headline for the move.
 
        Examples:
            "Critical blunder on move 14 by White"
            "Excellent move on move 5 by White"
            "Mistake in the endgame by Black"
        """
        category  = data.category.lower()
        drop      = abs(data.score_after - data.score_before)
        severity  = _score_delta_to_severity(drop)
 
        prefixes = {
            "blunder":    f"Critical {severity} blunder",
            "mistake":    f"Significant mistake",
            "inaccuracy": f"Minor inaccuracy",
            "good":       f"Solid move",
            "excellent":  f"Excellent move",
        }
        prefix = prefixes.get(category, "Move")
        return f"{prefix} on move {data.move_number} by {data.player_color}"
 
    def _default_tip(self, data: MoveData) -> str:
        """Returns a default tip when none is provided."""
        tips = {
            "blunder":    "Review this position carefully — big mistakes often have a pattern.",
            "mistake":    "Look for the key difference between your move and the best move.",
            "inaccuracy": "Small improvements add up — try to find the subtle difference.",
            "good":       "Good move! Understanding why helps you repeat it.",
            "excellent":  "This is exactly the kind of move to remember and repeat.",
        }
        return tips.get(data.category.lower(),
                       "Review this position with the engine to understand better.")
 
 
# =============================================================================
# QUICK TEST
# =============================================================================
 
if __name__ == "__main__":
    print("=" * 65)
    print("response_formatter.py — Quick Test")
    print("=" * 65)
 
    formatter = ResponseFormatter(max_length=500, style="markdown")
 
    # Test data matching 
    data = MoveData(
        move_played  = "d6",
        score_before = 0.5,
        score_after  = 1.8,
        best_move    = "Nf6",
        best_score   = 0.4,
        category     = "blunder",
        piece        = "pawn",
        phase        = "opening",
        move_number  = 6,
        player_color = "Black",
        tactic_type  = "opening_error",
        reason       = "Poor opening move"
    )
 
    raw_text = """This was a critical blunder that immediately shifted 
    the position. **The move** overlooked the opponent's response.
    The correct approach was Nf6, which maintains development."""
 
    print("\n[TEST 1] Single format")
    result = formatter.format(raw_text, data, "Focus on development in the opening.")
    print(result.to_markdown())
 
    print("\n[TEST 2] Streamlit format")
    streamlit_data = formatter.format_for_streamlit(result)
    for key, val in streamlit_data.items():
        print(f"  {key}: {val[:60] if isinstance(val, str) else val}")
 
    print("\n[TEST 3] Batch format")
    data2 = MoveData("Nf3", 0.1, 0.4, "Nf3", 0.4, "excellent",
                     "knight", "opening", 3, "White")
    results = formatter.format_batch(
        [raw_text, "Excellent move that develops the knight."],
        [data, data2],
        ["Work on openings.", "Keep playing like this!"]
    )
    print(f"  Formatted {len(results)} moves")
    for r in results:
        print(f"  Move {r.move_number}: [{r.severity_label}] {r.headline}")
 
    print("\n" + "=" * 65)
    print("ALL TESTS PASSED!")
    print("=" * 65)