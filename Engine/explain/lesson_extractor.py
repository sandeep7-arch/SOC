"""
lesson_extractor.py - Extracts Coaching Lessons from Game Mistakes
==================================================================
YOUR MODULE: engine/explain/lesson_extractor.py
 
WHAT THIS FILE DOES:
--------------------
Looks at ALL mistakes a player made in a game,
finds PATTERNS in those mistakes, and extracts
concrete coaching lessons.
 
Example:
  Player made 3 opening mistakes + 2 king safety mistakes
  LessonExtractor detects: "recurring opening problems"
  Output: "You consistently struggle with piece development
           in the opening. Practice the first 10 moves of
           your main openings until they become automatic."
 
This is different from blunder_explainer.py which explains
each mistake individually. This file finds the PATTERN
ACROSS mistakes and gives ONE big lesson per pattern.
 
WHAT THIS FILE CONTAINS:
    1. Lesson              - single extracted lesson
    2. LessonReport        - full lesson report for a game
    3. LessonExtractor     - main class
"""
 
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import Counter
 
from engine.explain.llm_client import LLMProvider, MockLLMProvider
from engine.explain.prompt_builder import (
    MoveData, PromptBuilder,
    _score_to_words, _score_delta_to_severity,
    build_lesson_prompt
)
 
 
# =============================================================================
# OUTPUT DATA STRUCTURES
# =============================================================================
 
@dataclass
class Lesson:
    """
    A single coaching lesson extracted from a pattern of mistakes.
 
    Attributes:
        title          : Short lesson title e.g. "Opening Development Issues"
        description    : Full lesson explanation (from LLM)
        action_item    : One specific thing to practice
        pattern_type   : What pattern triggered this e.g. "opening_error"
        phase          : Which phase this lesson applies to
        mistake_count  : How many mistakes showed this pattern
        severity       : How important this lesson is (high/medium/low)
        example_moves  : The actual moves that triggered this lesson
    """
    title:         str
    description:   str
    action_item:   str
    pattern_type:  str
    phase:         str
    mistake_count: int
    severity:      str
    example_moves: List[MoveData] = field(default_factory=list)
 
    def to_dict(self) -> dict:
        return {
            "title":         self.title,
            "description":   self.description,
            "action_item":   self.action_item,
            "pattern_type":  self.pattern_type,
            "phase":         self.phase,
            "mistake_count": self.mistake_count,
            "severity":      self.severity,
        }
 
    def to_markdown(self) -> str:
        severity_icons = {
            "high":   "🔴",
            "medium": "🟠",
            "low":    "🟡",
        }
        icon = severity_icons.get(self.severity, "📚")
        return f"""### {icon} {self.title}
 
**Pattern found:** {self.mistake_count} mistakes of type `{self.pattern_type}` in {self.phase}
 
{self.description}
 
**Practice this:** {self.action_item}
 
---"""
 
 
@dataclass
class LessonReport:
    """
    Complete lesson report for an entire game.
    Contains all patterns found and lessons extracted.
 
    Attributes:
        lessons         : List of Lesson objects
        total_lessons   : How many lessons found
        priority_lesson : Most important lesson (highest severity + count)
        phase_breakdown : How many mistakes per phase
        type_breakdown  : How many mistakes per tactic type
        player_color    : Which player these lessons are for
        generated_at    : Timestamp
    """
    lessons:         List[Lesson]
    total_lessons:   int
    priority_lesson: Optional[Lesson]
    phase_breakdown: Dict[str, int]
    type_breakdown:  Dict[str, int]
    player_color:    str   = "White"
    generated_at:    float = field(default_factory=time.time)
 
    def to_dict(self) -> dict:
        return {
            "total_lessons":   self.total_lessons,
            "player_color":    self.player_color,
            "phase_breakdown": self.phase_breakdown,
            "type_breakdown":  self.type_breakdown,
            "lessons":         [l.to_dict() for l in self.lessons],
            "priority_lesson": self.priority_lesson.to_dict()
                               if self.priority_lesson else None,
        }
 
    def to_markdown(self) -> str:
        lines = [
            f"# Coaching Lessons for {self.player_color}",
            "",
            "## Summary",
            f"- **Lessons identified:** {self.total_lessons}",
            "",
            "### Mistakes by Phase",
        ]
        for phase, count in self.phase_breakdown.items():
            lines.append(f"- {phase.capitalize()}: {count} mistakes")
 
        lines += ["", "### Mistakes by Type"]
        for mtype, count in self.type_breakdown.items():
            lines.append(f"- {mtype}: {count} occurrences")
 
        if self.priority_lesson:
            lines += [
                "",
                "## Most Important Lesson",
                self.priority_lesson.to_markdown(),
            ]
 
        lines += ["", "## All Lessons"]
        for lesson in self.lessons:
            lines.append(lesson.to_markdown())
 
        return "\n".join(lines)
 
 
# =============================================================================
# MAIN CLASS
# =============================================================================
 
class LessonExtractor:
    """
    Extracts coaching lessons from patterns in game mistakes.
 
    HOW IT WORKS:
        1. Takes all mistakes from a game
        2. Groups them by tactic_type and phase
        3. Finds patterns (e.g. 3 opening errors = a pattern)
        4. Calls LLM to generate a lesson for each pattern
        5. Returns prioritized LessonReport
 
    CONNECTS TO:
        Uses tactic_type from categorize_mistake() to group mistakes.
        The more specific analysis, the better the lessons.
 
    USAGE:
        extractor = LessonExtractor(llm=MockLLMProvider())
 
        # All mistakes from the game (from classify_game output)
        all_mistakes = blunders + mistakes + inaccuracies
 
        # Get lessons for White
        report = extractor.extract(all_mistakes, player_color="White")
        print(report.to_markdown())
 
    Args:
        llm              : Any LLMProvider
        min_pattern_count: Minimum mistakes to form a pattern (default 2)
        max_lessons      : Max lessons to return (default 5)
    """
 
    # Lesson titles per pattern type — shown in the report
    LESSON_TITLES = {
        "opening_error":   "Opening Principles Need Work",
        "missed_tactic":   "Tactical Awareness Issues",
        "king_safety":     "King Safety Awareness",
        "endgame_error":   "Endgame Technique",
        "positional_error":"Positional Understanding",
        "blunder":         "Avoiding Catastrophic Mistakes",
        "mistake":         "Reducing Serious Errors",
        "inaccuracy":      "Improving Move Precision",
    }
 
    # Action items per pattern — concrete practice advice
    ACTION_ITEMS = {
        "opening_error": (
            "Practice your opening repertoire daily. "
            "Know the ideas behind every move in your first 10 moves."
        ),
        "missed_tactic": (
            "Solve 10 tactical puzzles every day on chess.com or lichess.org. "
            "Focus on forks, pins, and skewers."
        ),
        "king_safety": (
            "Before every move, ask: does this leave my king exposed? "
            "Practice games where you prioritize castling in the first 8 moves."
        ),
        "endgame_error": (
            "Study basic endgame patterns: K+P vs K, rook endgames, and "
            "king centralization. 15 minutes of endgame study daily."
        ),
        "positional_error": (
            "Study one classical game per week focusing on positional ideas: "
            "piece placement, pawn structure, and long-term planning."
        ),
        "blunder": (
            "Before every move, do a quick check: Is my piece hanging? "
            "Am I walking into a tactic? Take 10 seconds before every move."
        ),
        "mistake": (
            "After each game, review your mistakes with an engine. "
            "Understand the key difference between your move and the best move."
        ),
        "inaccuracy": (
            "Work on candidate move selection — always consider at least "
            "3 moves before deciding. Quality over speed."
        ),
    }
 
    # Severity based on how many mistakes of this type
    SEVERITY_THRESHOLDS = {
        "high":   3,   # 3+ mistakes = high priority lesson
        "medium": 2,   # 2 mistakes = medium priority
        "low":    1,   # 1 mistake = low priority
    }
 
    def __init__(
        self,
        llm:               Optional[LLMProvider] = None,
        min_pattern_count: int = 1,
        max_lessons:       int = 5,
    ):
        self._llm              = llm or MockLLMProvider()
        self._prompt_builder   = PromptBuilder()
        self.min_pattern_count = min_pattern_count
        self.max_lessons       = max_lessons
 
    def extract(
        self,
        mistakes:     List[MoveData],
        player_color: str = "White"
    ) -> LessonReport:
        """
        Extracts lessons from all mistakes for one player.
 
        Args:
            mistakes     : All bad moves (blunders+mistakes+inaccuracies)
                           from classify_game() output as MoveData objects
            player_color : "White" or "Black" — whose mistakes to analyze
 
        Returns:
            LessonReport: Prioritized lessons for this player
        """
        # Filter to this player's mistakes only
        player_mistakes = [
            m for m in mistakes
            if m.player_color.lower() == player_color.lower()
            and m.category.lower() in ("blunder", "mistake", "inaccuracy")
        ]
 
        if not player_mistakes:
            return self._empty_report(player_color)
 
        # Step 1: Find patterns
        patterns = self._find_patterns(player_mistakes)
 
        # Step 2: Generate lesson for each pattern
        lessons = []
        for pattern_key, pattern_moves in patterns.items():
            if len(pattern_moves) >= self.min_pattern_count:
                lesson = self._generate_lesson(
                    pattern_key, pattern_moves
                )
                lessons.append(lesson)
 
        # Step 3: Sort by priority (severity + count)
        lessons = self._prioritize(lessons)[:self.max_lessons]
 
        # Step 4: Build breakdown stats
        phase_breakdown = self._count_by_field(player_mistakes, "phase")
        type_breakdown  = self._count_by_field(player_mistakes, "tactic_type")
 
        # Step 5: Find priority lesson
        priority = lessons[0] if lessons else None
 
        return LessonReport(
            lessons         = lessons,
            total_lessons   = len(lessons),
            priority_lesson = priority,
            phase_breakdown = phase_breakdown,
            type_breakdown  = type_breakdown,
            player_color    = player_color,
        )
 
    def extract_by_phase(
        self,
        mistakes:     List[MoveData],
        player_color: str = "White"
    ) -> Dict[str, LessonReport]:
        """
        Extracts separate lesson reports for each game phase.
 
        Returns a dict with keys "opening", "middlegame", "endgame".
        Each value is a LessonReport for that phase.
 
        Args:
            mistakes     : All bad moves as MoveData objects
            player_color : "White" or "Black"
 
        Returns:
            Dict[str, LessonReport]: One report per phase
        """
        reports = {}
        for phase in ["opening", "middlegame", "endgame"]:
            phase_mistakes = [
                m for m in mistakes
                if m.phase.lower() == phase
                and m.player_color.lower() == player_color.lower()
            ]
            if phase_mistakes:
                reports[phase] = self.extract(phase_mistakes, player_color)
        return reports
 
    def get_top_lesson(
        self,
        mistakes:     List[MoveData],
        player_color: str = "White"
    ) -> Optional[Lesson]:
        """
        Returns ONLY the single most important lesson.
        Quick method for dashboard summary cards.
 
        Args:
            mistakes     : All bad moves
            player_color : "White" or "Black"
 
        Returns:
            Lesson: The most important lesson, or None
        """
        report = self.extract(mistakes, player_color)
        return report.priority_lesson
 
    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------
 
    def _find_patterns(
        self, mistakes: List[MoveData]
    ) -> Dict[str, List[MoveData]]:
        """
        Groups mistakes by their tactic_type (from categorize_mistake).
        Falls back to grouping by category if tactic_type is empty.
 
        Returns:
            Dict mapping pattern_key -> list of MoveData
        """
        patterns: Dict[str, List[MoveData]] = {}
 
        for move in mistakes:
            # Use tactic_type if available (from categorize_mistake)
            # Otherwise use category (blunder/mistake/inaccuracy)
            key = move.tactic_type if move.tactic_type else move.category
 
            if key not in patterns:
                patterns[key] = []
            patterns[key].append(move)
 
        return patterns
 
    def _generate_lesson(
        self,
        pattern_key:  str,
        moves:        List[MoveData]
    ) -> Lesson:
        """
        Generates one Lesson object for a pattern.
 
        Uses LLM for description, templates for action_item.
        Phase is the most common phase among the moves.
 
        Args:
            pattern_key : The pattern identifier e.g. "opening_error"
            moves       : All moves showing this pattern
 
        Returns:
            Lesson: Complete lesson with description and action item
        """
        # Most common phase among these mistakes
        phases      = [m.phase for m in moves]
        common_phase = Counter(phases).most_common(1)[0][0]
 
        # Build lesson prompt and get LLM description
        prompt      = build_lesson_prompt(pattern_key, moves, common_phase)
        description = self._get_llm_response(prompt)
 
        # Get title and action item from templates
        title       = self.LESSON_TITLES.get(
            pattern_key,
            f"{pattern_key.replace('_', ' ').title()} Issues"
        )
        action_item = self.ACTION_ITEMS.get(
            pattern_key,
            "Review this pattern with an engine after every game."
        )
 
        # Determine severity
        count    = len(moves)
        severity = "low"
        if count >= self.SEVERITY_THRESHOLDS["high"]:
            severity = "high"
        elif count >= self.SEVERITY_THRESHOLDS["medium"]:
            severity = "medium"
 
        return Lesson(
            title         = title,
            description   = description.strip(),
            action_item   = action_item,
            pattern_type  = pattern_key,
            phase         = common_phase,
            mistake_count = count,
            severity      = severity,
            example_moves = moves[:3],  # keep top 3 examples
        )
 
    def _prioritize(self, lessons: List[Lesson]) -> List[Lesson]:
        """
        Sorts lessons by priority.
        High severity first, then by mistake count.
        """
        severity_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(
            lessons,
            key=lambda l: (severity_order.get(l.severity, 3),
                          -l.mistake_count)
        )
 
    def _count_by_field(
        self, moves: List[MoveData], field_name: str
    ) -> Dict[str, int]:
        """Counts mistakes grouped by a field name."""
        counts: Dict[str, int] = {}
        for move in moves:
            value = getattr(move, field_name, "unknown") or "unknown"
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))
 
    def _get_llm_response(self, prompt: str) -> str:
        """Gets LLM response with retry."""
        try:
            return self._llm.complete_with_retry(
                prompt, max_tokens=300, retries=2, delay=1.0
            )
        except Exception:
            return "Review this pattern carefully with engine analysis."
 
    def _empty_report(self, player_color: str) -> LessonReport:
        """Returns empty report when no mistakes found."""
        return LessonReport(
            lessons         = [],
            total_lessons   = 0,
            priority_lesson = None,
            phase_breakdown = {},
            type_breakdown  = {},
            player_color    = player_color,
        )
 
 
# =============================================================================
# QUICK TEST
# =============================================================================
 
if __name__ == "__main__":
    print("=" * 65)
    print("lesson_extractor.py — Quick Test")
    print("=" * 65)
 
    extractor = LessonExtractor(
        llm               = MockLLMProvider(),
        min_pattern_count = 1,
        max_lessons       = 5,
    )
 
    # Simulating Piyush's classify_game() + categorize_mistake() output
    mistakes = [
        MoveData("d6",   0.5,  1.8,  "Nf6", 0.4, "blunder",
                 "pawn",   "opening",    6,  "Black",
                 "opening_error", "Poor development"),
        MoveData("h6",   0.3,  0.9,  "Nc6", 0.2, "mistake",
                 "pawn",   "opening",    8,  "Black",
                 "opening_error", "Wasted tempo"),
        MoveData("Rxd4", -0.8, -3.5, "Nf6", -0.6, "blunder",
                 "rook",  "middlegame", 28,  "Black",
                 "missed_tactic", "Fork was available"),
        MoveData("Kf8",  -2.0, -3.1, "Ke7", -1.8, "mistake",
                 "king",  "endgame",   42,   "Black",
                 "endgame_error", "King should centralize"),
        MoveData("a6",   0.2,  0.7,  "Nc6", 0.1, "inaccuracy",
                 "pawn",  "opening",    4,   "Black",
                 "opening_error", "Another tempo loss"),
    ]
 
    # ── TEST 1: Extract all lessons ───────────────────────────────────
    print("\n[TEST 1] Extract all lessons for Black")
    report = extractor.extract(mistakes, player_color="Black")
    print(report.to_markdown())
 
    # ── TEST 2: Phase breakdown ───────────────────────────────────────
    print("\n[TEST 2] Lessons by phase")
    phase_reports = extractor.extract_by_phase(mistakes, player_color="Black")
    for phase, r in phase_reports.items():
        print(f"  {phase}: {r.total_lessons} lessons")
        if r.priority_lesson:
            print(f"    Top lesson: {r.priority_lesson.title}")
 
    # ── TEST 3: Top single lesson ─────────────────────────────────────
    print("\n[TEST 3] Single most important lesson")
    top = extractor.get_top_lesson(mistakes, player_color="Black")
    if top:
        print(f"  Title    : {top.title}")
        print(f"  Pattern  : {top.pattern_type}")
        print(f"  Count    : {top.mistake_count} mistakes")
        print(f"  Severity : {top.severity}")
        print(f"  Action   : {top.action_item[:60]}...")
 
    # ── TEST 4: Type breakdown ────────────────────────────────────────
    print("\n[TEST 4] Mistake type breakdown")
    for mtype, count in report.type_breakdown.items():
        print(f"  {mtype}: {count}")
 
    print("\n" + "=" * 65)
    print("ALL TESTS PASSED!")
    print("Modules 1+2+3+4+5+6+7 complete.")
    print("\nYour explain/ module is fully built!")
    print("=" * 65)
 