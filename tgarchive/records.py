"""Преобразование telethon-сообщений в записи формата архива (совместимо со старыми дампами)."""

from telethon import utils
from telethon.tl import types


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


def classify_media(msg):
    """Тип файла для скачивания; None — нечего скачивать (текст/вебстраница/опрос)."""
    if msg.media is None:
        return None
    if msg.photo:
        return "photo"
    if msg.voice:
        return "voice"
    if msg.video_note:
        return "video_note"
    if msg.sticker:
        return "sticker"
    if msg.gif:
        return "gif"
    if msg.audio:
        return "audio"
    if msg.video:
        return "video"
    if msg.document:
        return "document"
    return None


def _reactions(msg):
    rx = getattr(msg, "reactions", None)
    if not rx or not rx.results:
        return None
    out = []
    for r in rx.results:
        reaction = r.reaction
        if isinstance(reaction, types.ReactionEmoji):
            e = reaction.emoticon
        elif isinstance(reaction, types.ReactionCustomEmoji):
            e = f"custom:{reaction.document_id}"
        else:
            e = type(reaction).__name__
        out.append({"emoji": e, "count": r.count})
    return out


def _poll(msg):
    media = msg.media
    if not isinstance(media, types.MessageMediaPoll):
        return None
    q = getattr(media.poll.question, "text", media.poll.question)
    answers = [getattr(a.text, "text", a.text) for a in media.poll.answers]
    return {"question": q, "answers": answers}


def _forward(msg):
    f = msg.fwd_from
    if not f:
        return None
    return {
        "from_id": utils.get_peer_id(f.from_id) if f.from_id else None,
        "from_name": f.from_name,
        "date": iso(f.date),
        "channel_post": f.channel_post,
    }


def _links(msg):
    out = []
    try:
        for ent, txt in msg.get_entities_text() or []:
            if isinstance(ent, types.MessageEntityUrl):
                out.append(txt)
            elif isinstance(ent, types.MessageEntityTextUrl):
                out.append(ent.url)
    except Exception:
        pass
    if isinstance(msg.media, types.MessageMediaWebPage):
        url = getattr(msg.media.webpage, "url", None)
        if url and url not in out:
            out.append(url)
    return out


def message_to_record(msg, chat_id: int) -> dict:
    topic_id = reply_to = None
    rt = msg.reply_to
    if rt is not None and getattr(rt, "reply_to_msg_id", None) is not None:
        if getattr(rt, "forum_topic", False):
            top = getattr(rt, "reply_to_top_id", None)
            if top:
                topic_id, reply_to = top, rt.reply_to_msg_id
            else:
                # пост прямо в топик: reply_to_msg_id указывает на корень топика
                topic_id, reply_to = rt.reply_to_msg_id, None
        else:
            reply_to = rt.reply_to_msg_id

    sender = msg.sender
    sender_name = utils.get_display_name(sender) if sender else (getattr(msg, "post_author", None) or "")

    rec = {
        "chat_id": chat_id,
        "message_id": msg.id,
        "sender_id": msg.sender_id,
        "sender_name": sender_name or None,
        "text": msg.message or "",
        "date": iso(msg.date),
        "edit_date": iso(msg.edit_date),
        "topic_id": topic_id,
        "reply_to_message_id": reply_to,
        "forward": _forward(msg),
        "links": _links(msg),
        "media_type": type(msg.media).__name__ if msg.media else None,
        "raw": {
            "post": bool(msg.post),
            "silent": bool(msg.silent),
            "views": msg.views,
            "forwards": msg.forwards,
            "grouped_id": msg.grouped_id,
        },
    }
    rx = _reactions(msg)
    if rx:
        rec["reactions"] = rx
    p = _poll(msg)
    if p:
        rec["poll"] = p
    action = getattr(msg, "action", None)
    if action is not None:
        rec["service"] = {"action": type(action).__name__}
    return rec
