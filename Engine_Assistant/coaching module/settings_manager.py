# settings_manager.py

DEFAULT_SETTINGS = {
    "feedback_enabled": True,
    "difficulty": "beginner",
    "show_recommendations": True,
    "show_postgame_report": True,
    "adaptive_training": True
}


def get_default_settings():
    """
    Returns default coaching settings.
    """

    return DEFAULT_SETTINGS.copy()


def update_setting(settings,
                   key,
                   value):
    """
    Updates a specific setting.
    """

    if key not in settings:
        raise KeyError(
            f"Unknown setting: {key}"
        )

    settings[key] = value

    return settings


def reset_settings():
    """
    Resets all settings to defaults.
    """

    return DEFAULT_SETTINGS.copy()


def is_feedback_enabled(settings):
    """
    Checks if realtime feedback is enabled.
    """

    return settings.get(
        "feedback_enabled",
        True
    )


def is_adaptive_training_enabled(
    settings
):
    """
    Checks if adaptive training is enabled.
    """

    return settings.get(
        "adaptive_training",
        True
    )


if __name__ == "__main__":

    settings = get_default_settings()

    print("Default Settings:")
    print(settings)

    print()

    update_setting(
        settings,
        "difficulty",
        "intermediate"
    )

    update_setting(
        settings,
        "feedback_enabled",
        False
    )

    print("Updated Settings:")
    print(settings)

    print()

    settings = reset_settings()

    print("Reset Settings:")
    print(settings)