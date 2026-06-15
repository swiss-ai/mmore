"""TUI-only exceptions."""

from __future__ import annotations


class UserCancelledError(Exception):
    """Raised when the user cancels a sub-flow (Ctrl-C or Esc inside a prompt).

    Caught by the top-level menu loop so cancellation returns to the main menu
    instead of exiting the whole TUI.
    """
