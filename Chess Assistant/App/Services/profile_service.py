import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

sys.path.append(os.path.join(ROOT, 'Engine_Assistant', 'Analysis'))
sys.path.append(os.path.join(ROOT, 'Engine_Assistant', 'Player model'))
sys.path.append(os.path.join(ROOT, 'Database'))          
sys.path.append(ROOT)


from Database.session import get_db_session
from Database.models import Player, Game, Mistake, Recommendation, VulnerabilityVector
from Engine_Assistant.Player_model.player_profile import PlayerProfile
from Engine_Assistant.Player_model.game_index import GameIndex, GameRecord
from Engine_Assistant.Player_model.vulnerability_vector import build_vulnerability_vector
from Engine_Assistant.Player_model.progress_tracker import get_full_progress_report
from Engine_Assistant.Player_model.personalization import build_personalization_context, build_llm_context_string


def get_or_create_player(name: str, rating: int = 1200) -> Player:
    """
    Loads existing player from database or creates a new one.
    """
    with get_db_session() as session:
        player = session.query(Player).filter_by(name=name).first()
        if not player:
            player = Player(name=name, rating=rating)
            session.add(player)
            session.commit()
            session.refresh(player)
        session.expunge(player)  # detach from session so it can be used outside
        return player


def save_game_to_db(player_id: int, game_analysis: dict, result: str, opponent_rating: int = 1200):
    """
    Saves a completed game and all its mistakes to the database.
    game_analysis: output from analysis_service.analyze_full_game()
    """
    with get_db_session() as session:
        player = session.query(Player).filter_by(id=player_id).first()
        if not player:
            raise ValueError(f"Player id {player_id} does not exist.")

        # Save game record
        game = Game(
            player_id=player_id,
            result=result,
            opponent_rating=opponent_rating,
            phase_reached=game_analysis["moves"][-1]["phase"] if game_analysis["moves"] else "opening",
            total_moves=game_analysis["total_moves"],
            blunder_count=game_analysis["total_blunders"],
            mistake_count=game_analysis["total_mistakes"],
            inaccuracy_count=game_analysis["total_inaccuracies"]
        )
        session.add(game)
        session.flush()  # get game.id without full commit

        player.total_games = (player.total_games or 0) + 1
        if result == "win":
            player.wins = (player.wins or 0) + 1
        elif result == "loss":
            player.losses = (player.losses or 0) + 1
        else:
            player.draws = (player.draws or 0) + 1

        # Save individual mistakes
        for move in game_analysis["moves"]:
            if not move.get("is_player_move", True):
                continue

            if move["classification"] in ["blunder", "mistake", "inaccuracy"]:
                mistake = Mistake(
                    game_id=game.id,
                    player_id=player_id,
                    move_number=move["move_number"],
                    move=move["move"],
                    classification=move["classification"],
                    category=move.get("category"),
                    eval_before=move["eval_before"],
                    eval_after=move["eval_after"],
                    drop=move["drop"],
                    phase=move["phase"]
                )
                session.add(mistake)

        session.commit()
        return game.id


def save_recommendations(player_id: int, recommendations: list[dict]):
    """
    Saves coaching recommendations shown to the player after a game.
    """
    with get_db_session() as session:
        for rec in recommendations:
            recommendation = Recommendation(
                player_id=player_id,
                weakness_category=rec.get("weakness"),
                recommendation_text=(
                    f"{rec.get('lesson_title', 'Training')}: "
                    f"{rec.get('lesson_description', '')}"
                ),
                priority=rec.get("score", 0.5)
            )
            session.add(recommendation)


def save_vulnerability_vector(player_id: int, vector: dict):
    """
    Saves current vulnerability vector snapshot to database.
    Called after every game so progress_tracker has history.
    """
    with get_db_session() as session:
        vv = VulnerabilityVector(
            player_id=player_id,
            missed_tactic=vector.get("missed_tactic", 0.0),
            king_safety=vector.get("king_safety", 0.0),
            positional_error=vector.get("positional_error", 0.0),
            endgame_error=vector.get("endgame_error", 0.0),
            opening_error=vector.get("opening_error", 0.0),
            doubled_pawns=vector.get("doubled_pawns", 0.0),
            isolated_pawns=vector.get("isolated_pawns", 0.0),
            low_piece_activity=vector.get("low_piece_activity", 0.0),
            poor_center_control=vector.get("poor_center_control", 0.0)
        )
        session.add(vv)
        session.commit()


def load_player_profile(player_id: int) -> dict:
    """
    Loads full player profile from database.
    Returns everything needed for the profile page and LLM context.
    """
    with get_db_session() as session:
        # Load player
        player = session.query(Player).filter_by(id=player_id).first()
        if not player:
            return None

        # Load vulnerability history
        vectors = session.query(VulnerabilityVector).filter_by(
            player_id=player_id
        ).order_by(VulnerabilityVector.id).all()

        vector_history = [v.to_dict() for v in vectors]

        # Latest vector
        latest_vector = vector_history[-1] if vector_history else {}

        # Progress trends
        progress = get_full_progress_report(vector_history) if len(vector_history) >= 2 else {}

        # Recent mistakes
        recent_mistakes = session.query(Mistake).filter_by(
            player_id=player_id
        ).order_by(Mistake.id.desc()).limit(20).all()

        mistake_categories = [m.category for m in recent_mistakes if m.category]

        return {
            "player_id": player_id,
            "name": player.name,
            "rating": player.rating,
            "total_games": player.total_games,
            "wins": player.wins,
            "losses": player.losses,
            "draws": player.draws,
            "win_rate": player.win_rate(),
            "playing_style": player.playing_style,
            "vulnerability_vector": latest_vector,
            "vector_history": vector_history,
            "progress": progress,
            "recent_mistake_categories": mistake_categories
        }


if __name__ == "__main__":
    # Test the full pipeline
    print("Testing profile service:")

    # Get or create player
    player = get_or_create_player("Jonathan", rating=1200)
    print(f"Player: {player.name} (id={player.id})")

    # Simulate saving a game
    mock_game_analysis = {
        "moves": [
            {"move_number": 1, "move": "e2e4", "phase": "opening",
             "eval_before": 0.0, "eval_after": 0.3,
             "drop": -0.3, "classification": "good"},
            {"move_number": 3, "move": "g1f3", "phase": "opening",
             "eval_before": 0.3, "eval_after": 0.4,
             "drop": -0.1, "classification": "good"},
            {"move_number": 5, "move": "d2d4", "phase": "opening",
             "eval_before": 0.5, "eval_after": -1.2,
             "drop": 1.7, "classification": "blunder"},
        ],
        "total_blunders": 1,
        "total_mistakes": 0,
        "total_inaccuracies": 0,
        "total_moves": 3,
        "final_weaknesses": {"weaknesses": []}
    }

    game_id = save_game_to_db(player.id, mock_game_analysis, "loss", 1400)
    print(f"Game saved with id: {game_id}")

    # Save vulnerability vector
    mock_vector = {
        "missed_tactic": 0.5, "king_safety": 0.3,
        "positional_error": 0.2, "endgame_error": 0.1,
        "opening_error": 0.3, "doubled_pawns": 0.0,
        "isolated_pawns": 0.0, "low_piece_activity": 0.0,
        "poor_center_control": 0.0
    }
    save_vulnerability_vector(player.id, mock_vector)
    print("Vulnerability vector saved")

    # Load full profile
    profile = load_player_profile(player.id)
    print(f"\nLoaded profile for: {profile['name']}")
    print(f"Total games: {profile['total_games']}")
    print(f"Vulnerability vector: {profile['vulnerability_vector']}")
    print(f"Progress: {profile['progress']}")
