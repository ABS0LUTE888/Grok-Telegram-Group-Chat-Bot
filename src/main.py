"""
This bot is being added in Telegram group chat and, when mentioned, forwards the prompt
(along with the recent chat history and an optional replied-to message)
to xAI's Grok and replies with the model's text.

Environment variables
---------------------
- BOT_TOKEN:    Telegram bot token (required).
- XAI_API_KEY:  xAI API key for Grok (required).
- MAX_SNIPPET_LEN:  Max chars to keep from each message for history  (default 160).
- MESSAGE_LIMIT:    Max number of recent messages to store in the 'memory' (default 30).
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import httpx
from aiogram import Bot, Dispatcher, F, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatType, ParseMode
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN missing - set it in .env")
elif not XAI_API_KEY:
    raise RuntimeError("XAI_API_KEY missing - set it in .env")


bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

BOT_USERNAME: str | None = None
BOT_DISPLAY: str | None = None
BOT_ID: int | None = None

MAX_SNIPPET_LEN = int(os.getenv("MAX_SNIPPET_LEN"))
MESSAGE_HISTORY_LIMIT = int(os.getenv("MESSAGE_LIMIT"))

# Rolling history dictionary (recent messages in the chat):
_HISTORY: Dict[int, Deque[Tuple[bool, str]]] = defaultdict(lambda: deque(maxlen=MESSAGE_HISTORY_LIMIT))


async def ensure_identity() -> None:
    """
    Populate global bot identity values (username, display, id) once
    """
    global BOT_USERNAME, BOT_DISPLAY, BOT_ID
    if BOT_USERNAME is None:
        me = await bot.get_me()
        BOT_USERNAME = f"@{me.username.lower()}"
        BOT_DISPLAY = me.first_name or "Bot"
        BOT_ID = me.id


def format_user(u: types.User) -> str:
    """
    Build a human-readable user label like "Alice Jones (@alice)".

    Falls back to just a name if the user has no public @username.
    """
    name = " ".join(filter(None, [u.first_name, u.last_name]))
    return f"{name} (@{u.username})" if u.username else name


def format_line(msg: types.Message, is_bot_msg: bool) -> str:
    """
    Convert a Telegram Message to a single-line snippet for context history.

    Parameters
    ----------
    msg : types.Message
        Telegram message to summarize.
    is_bot_msg : bool
        Whether the message was sent by the bot.

    Returns
    -------
    str
        One line summary ready to append to `_HISTORY`.
    """
    text = msg.text or msg.caption or f"[{msg.content_type} message]"
    text = text.replace("\n", " ")
    if len(text) > MAX_SNIPPET_LEN:
        text = text[: MAX_SNIPPET_LEN - 1] + "…"
    if is_bot_msg:
        return f"> {BOT_DISPLAY} (you): {text}"
    return f"> {format_user(msg.from_user)}: {text}"


async def ask_grok(chat_id: int, prompt: str, replied_line: Optional[str]) -> str:
    """
    Send a contextualized request to Grok and return text.

    A minimal prompt is constructed that includes:
      - A short rolling history for the current chat.
      - The one-line text of the message the user replied to (optional).
      - The current user prompt.



    The function handles HTTP errors and malformed JSON, returning a readable
    error marker string (so the bot can safely send it back to Telegram).

    Parameters
    ----------
    chat_id : int
        Telegram chat identifier used to fetch the local history buffer.
    prompt : str
        The user-provided text after mentioning the bot.
    replied_line : Optional[str]
        One-line summary of the replied-to message, or None if not replying.

    Returns
    -------
    str
        The Grok response text, or a bracketed error message placeholder.
    """
    if not XAI_API_KEY:
        return "<XAI_API_KEY missing>"

    history_lines = [line for _, line in list(_HISTORY[chat_id])]

    blocks: List[str] = []
    if history_lines:
        blocks.append("Chat history:\n" + "\n".join(history_lines))
    if replied_line and replied_line not in history_lines:
        blocks.append("Replied message:\n" + replied_line)
    blocks.append("User prompt:\n" + prompt)

    payload = {
        "model": "grok-4",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Grok integrated into a Telegram group chat. "
                    "Respond concisely and helpfully using the context."
                ),
            },
            {"role": "user", "content": "\n\n".join(blocks)},
        ],
    }

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            r = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_API_KEY}"},
                json=payload,
            )
            r.raise_for_status()

            try:
                data = r.json()
            except ValueError:
                return "<Grok response malformed – expected JSON>"

            return r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            return f"<Grok API error – {exc}>"


@dp.message(F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))
async def handle_group(msg: types.Message) -> None:
    """
    Group chat handler: triggers when the bot is *mentioned* in a group/supergroup.

    Behavior:
    - Ensures we know the bot's identity (username/id) so we can detect mentions.
    - Appends the incoming message to the rolling history.
    - If the bot is not actually mentioned in the text, bail out early.
    - Extract the user's prompt (text after the mention).
    - If the message is a reply to the message of another user, include a one-line summary of the replied-to
      message for additional context.
    - Send a typing action, ask Grok, and reply with the result.
    - Append the bot's own reply to the history as well.

    Caveats:
    - Only plain text content is used for context; other media is not uploaded.
    """
    await ensure_identity()

    chat_id = msg.chat.id
    is_bot_msg = msg.from_user.id == BOT_ID

    _HISTORY[chat_id].append((is_bot_msg, format_line(msg, is_bot_msg)))

    if BOT_USERNAME not in (msg.text or "").lower():
        return

    prompt = (msg.text or "").replace(BOT_USERNAME, "").strip()
    if not prompt:
        await msg.reply(f"Add a prompt after mentioning me, e.g. {BOT_USERNAME} what’s up?")
        return

    reply = (
        format_line(msg.reply_to_message, msg.reply_to_message.from_user.id == BOT_ID)
        if msg.reply_to_message
        else None
    )

    await msg.chat.do("typing")
    answer = await ask_grok(chat_id, prompt, reply)

    sent = await msg.answer(answer)

    _HISTORY[chat_id].append((True, format_line(sent, is_bot_msg=True)))


async def main() -> None:
    await ensure_identity()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
