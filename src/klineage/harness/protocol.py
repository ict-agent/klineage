"""Model wire protocols supported by the provider bridge."""

from enum import StrEnum


class WireAPI(StrEnum):
    RESPONSES = "responses"
    MESSAGES = "messages"
