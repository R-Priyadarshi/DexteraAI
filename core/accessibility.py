"""
DexteraAI Accessibility & Internationalization Utilities

- Locale-aware onboarding
- Multi-language support
- Accessibility features (motor disability, sign language)
- Privacy compliance (GDPR, CCPA)
"""

from __future__ import annotations

import locale

SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "zh", "hi", "ar", "ru"]

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "welcome": "Welcome to DexteraAI!",
        "onboarding": "Get started with real-time gesture recognition.",
        "privacy": "All inference is on-device. No data leaves your device.",
    },
    "es": {
        "welcome": "¡Bienvenido a DexteraAI!",
        "onboarding": "Comienza con el reconocimiento de gestos en tiempo real.",
        "privacy": "Toda la inferencia es en el dispositivo. Ningún dato sale de tu dispositivo.",
    },
    # ...add more translations...
}


def get_locale() -> str:
    lang, _ = locale.getdefaultlocale()
    return lang.split("_")[0] if lang else "en"


def translate(key: str, lang: str | None = None) -> str:
    lang = lang or get_locale()
    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


# Accessibility features
class Accessibility:
    @staticmethod
    def motor_disability_support() -> str:
        return "DexteraAI supports customizable gestures for motor disabilities."

    @staticmethod
    def sign_language_support() -> str:
        return "DexteraAI can recognize basic sign language gestures."

    @staticmethod
    def privacy_notice() -> str:
        return translate("privacy")


# Example usage
if __name__ == "__main__":
    print(translate("welcome"))
    print(Accessibility.motor_disability_support())
    print(Accessibility.privacy_notice())
