"""Test configuration: tests must never send real Telegram messages, even when a live bot token is in .env."""
import pytest


@pytest.fixture(autouse=True)
def _no_live_telegram(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
