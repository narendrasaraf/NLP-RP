"""
game/chat.py
------------
Self-contained chat system for the Adaptive Shooter.

Design Goals:
  - Zero blocking: all state lives in plain Python variables.
  - Press ENTER  → toggle chat-input mode on/off.
  - While in chat mode:
      ENTER  → sends message (adds to history, stores latest)
      ESC    → cancel without sending
      BACKSPACE → delete last character
      Any printable key → append to buffer
  - Blinking cursor rendered via a simple timer flag.
  - Message length capped at MAX_MSG_LEN characters.
  - Keeps the last HISTORY_SIZE messages for on-screen display.
"""

import time
import pygame

# ── Tunable Constants ────────────────────────────────────────────────────────
MAX_MSG_LEN   = 100   # hard cap on characters per message
HISTORY_SIZE  = 8     # how many messages to display on screen
CURSOR_BLINK  = 0.5   # seconds per cursor phase (on / off)

# Chat panel geometry (bottom-left of screen)
CHAT_X        = 10
CHAT_BOTTOM_Y = None  # resolved at first draw() call using screen height
LINE_HEIGHT   = 22
PANEL_W       = 460
INPUT_H       = 28


class ChatSystem:
    """Encapsulates all chat state.  No pygame calls in __init__; safe to create early."""

    def __init__(self):
        # Whether the player is actively typing
        self.active: bool = False

        # Current in-progress text (shown while typing)
        self.buffer: str = ""

        # List of {"sender": str, "text": str} dicts — newest last
        self.history: list[dict] = []

        # The most recently *sent* message (consumed by telemetry dispatcher)
        self.latest_message: str = ""

        # Blinking cursor state
        self._cursor_visible: bool = True
        self._cursor_last_flip: float = time.time()

        # Font — initialised lazily on first draw()
        self._font: pygame.font.Font | None = None

    # ── Input Handling ────────────────────────────────────────────────────────

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Process a single pygame event that may be chat-related.

        Returns True if the event was consumed by chat (prevents game from also
        acting on it), False otherwise.
        """
        if event.type != pygame.KEYDOWN:
            return False

        # ENTER toggles chat mode / sends message
        if event.key == pygame.K_RETURN:
            if not self.active:
                # Open chat input mode
                self.active = True
                self.buffer = ""
                self._reset_cursor()
            else:
                # Send if buffer is non-empty
                self._send_message()
                self.active = False
            return True

        # Only react to remaining keys if chat is active
        if not self.active:
            return False

        if event.key == pygame.K_ESCAPE:
            # Cancel without sending
            self.buffer = ""
            self.active = False
            return True

        if event.key == pygame.K_BACKSPACE:
            self.buffer = self.buffer[:-1]
            return True

        # Printable character
        ch = event.unicode
        if ch and ch.isprintable() and len(self.buffer) < MAX_MSG_LEN:
            self.buffer += ch
            return True

        return False

    # ── Telemetry Integration ─────────────────────────────────────────────────

    def pop_latest(self) -> str:
        """
        Returns the latest chat message string and resets it.
        Safe to call even if no message was sent (returns "").
        Used by the telemetry dispatcher before building the API payload.
        """
        msg = self.latest_message or ""
        self.latest_message = ""
        return msg

    # ── Rendering ────────────────────────────────────────────────────────────

    def draw(self, screen: pygame.Surface) -> None:
        """
        Render the chat history panel and (if active) the input box.
        Lightweight: semi-transparent surfaces, no per-frame allocation of fonts.
        """
        # Lazy font init (must be inside a pygame.init() context)
        if self._font is None:
            self._font = pygame.font.SysFont("monospace", 18)

        sw, sh = screen.get_size()
        panel_top = sh - (HISTORY_SIZE * LINE_HEIGHT) - INPUT_H - 20

        # ── History Panel ────────────────────────────────────────────────────
        if self.history:
            # Semi-transparent dark backing
            visible = self.history[-HISTORY_SIZE:]
            panel_h = len(visible) * LINE_HEIGHT + 6
            bg = pygame.Surface((PANEL_W, panel_h), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 110))
            screen.blit(bg, (CHAT_X, panel_top))

            for i, msg in enumerate(visible):
                sender = msg.get("sender", "player")
                text   = msg.get("text",   "")
                label  = f"[{sender.upper()}]: {text}"
                # Truncate if it would overflow the panel width
                label = self._truncate(label, PANEL_W - 10)
                surf = self._font.render(label, True, (220, 220, 220))
                screen.blit(surf, (CHAT_X + 5, panel_top + 3 + i * LINE_HEIGHT))

        # ── Input Box (only when active) ─────────────────────────────────────
        if self.active:
            self._update_cursor()

            input_y = sh - INPUT_H - 8

            # Backing rect
            input_bg = pygame.Surface((PANEL_W, INPUT_H), pygame.SRCALPHA)
            input_bg.fill((0, 0, 0, 170))
            screen.blit(input_bg, (CHAT_X, input_y))
            pygame.draw.rect(screen, (80, 200, 120), (CHAT_X, input_y, PANEL_W, INPUT_H), 1)

            # Prompt + buffer + blinking cursor
            cursor_char = "|" if self._cursor_visible else " "
            display_text = f"▶ {self.buffer}{cursor_char}"
            display_text = self._truncate(display_text, PANEL_W - 10)
            surf = self._font.render(display_text, True, (80, 255, 130))
            screen.blit(surf, (CHAT_X + 5, input_y + 5))

            # "ENTER to send / ESC to cancel" hint
            hint_surf = self._font.render(
                "ENTER=send  ESC=cancel", True, (130, 130, 130)
            )
            screen.blit(hint_surf, (CHAT_X + PANEL_W + 6, input_y + 5))

        elif not self.history:
            # Show a subtle hint so players know chat exists
            hint = self._font.render(
                "Press ENTER to chat", True, (80, 80, 80)
            )
            screen.blit(hint, (CHAT_X, sh - 24))

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _send_message(self) -> None:
        """Commit the current buffer as a sent message."""
        text = self.buffer.strip()
        if not text:
            return
        record = {"sender": "player", "text": text}
        self.history.append(record)
        # Keep history bounded in memory (double the display limit is plenty)
        if len(self.history) > HISTORY_SIZE * 2:
            self.history = self.history[-HISTORY_SIZE:]
        # Expose for telemetry
        self.latest_message = text
        self.buffer = ""

    def _reset_cursor(self) -> None:
        self._cursor_visible = True
        self._cursor_last_flip = time.time()

    def _update_cursor(self) -> None:
        """Flip cursor visibility every CURSOR_BLINK seconds."""
        now = time.time()
        if now - self._cursor_last_flip >= CURSOR_BLINK:
            self._cursor_visible = not self._cursor_visible
            self._cursor_last_flip = now

    def _truncate(self, text: str, max_px: int) -> str:
        """Truncate text so it fits within max_px pixels at the current font size."""
        if self._font is None:
            return text
        while text and self._font.size(text)[0] > max_px:
            text = text[:-1]
        return text
