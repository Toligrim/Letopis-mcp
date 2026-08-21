"""TUI-просмотрщик архива (textual). Запуск: ./tg tui"""

import subprocess
import webbrowser

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView

from . import db as dbm
from . import viewdata as VD
from .config import load_config, project_root

PAGE = 80
PALETTE = ["bright_cyan", "bright_magenta", "bright_green", "bright_yellow",
           "bright_blue", "bright_red", "cyan", "magenta", "green", "yellow"]


def _color(name):
    h = 0
    for c in name or "?":
        h = (h * 31 + ord(c)) % len(PALETTE)
    return PALETTE[h]


def _msg_text(m, show_chat=False, topic_filtered=False):
    t = Text(overflow="fold")
    t.append((m["date"] or "")[:16].replace("T", " ") + " ", style="dim")
    if show_chat:
        label = m["chat"] + (f"/{m['topic']}" if m.get("topic") else "")
        t.append(f"[{label}] ", style="cyan")
    elif not topic_filtered and m.get("topic"):
        t.append(f"[{m['topic']}] ", style="dim cyan")
    t.append(m["sender"] or "—", style=f"bold {_color(m['sender'])}")
    if m.get("reply_to"):
        rp = m.get("reply_preview")
        frag = f" ↩{(rp['sender'] or '')}: {rp['text'][:40]}" if rp else f" ↩#{m['reply_to']}"
        t.append(frag, style="dim")
    t.append(": ")
    if m.get("service"):
        t.append(f"⚙ {m['service']['action']} ", style="italic dim")
    if m.get("media_type"):
        kind = m.get("media_kind") or (m["media_type"] or "").replace("MessageMedia", "").lower()
        name = f" «{m['media_name']}»" if m.get("media_name") else ""
        dl = "" if m.get("media_file") else " ✗"
        t.append(f"〈{kind}{name}{dl}〉 ", style="magenta")
    if m.get("poll"):
        t.append(f"📊 {m['poll']['question']} ", style="yellow")
    t.append((m.get("text") or "").replace("\n", " ⏎ "))
    if m.get("transcript"):
        t.append(f" 📝{m['transcript']}", style="italic green")
    if m.get("reactions"):
        t.append("  " + " ".join(f"{r['emoji']}{r['count']}" for r in m["reactions"]),
                 style="dim")
    return t


class MsgItem(ListItem):
    def __init__(self, m, show_chat=False, topic_filtered=False):
        super().__init__(Label(_msg_text(m, show_chat, topic_filtered), markup=False))
        self.m = m


class TgTui(App):
    TITLE = "tg архив"
    CSS = """
    #side { width: 44; }
    #chats { height: 12; border: round $primary 30%; }
    #chats:focus { border: round $primary; }
    #topics { height: 1fr; border: round $primary 30%; }
    #topics:focus { border: round $primary; }
    #feed { border: round $primary 30%; }
    #feed:focus { border: round $primary; }
    #prompt { dock: top; display: none; }
    ListItem { padding: 0 1; }
    """
    BINDINGS = [
        Binding("q", "quit", "выход"),
        Binding("slash", "prompt('search')", "поиск", key_display="/"),
        Binding("g", "prompt('date')", "дата"),
        Binding("s", "prompt('sender')", "автор"),
        Binding("o", "older", "↑старее"),
        Binding("n", "newer", "↓новее"),
        Binding("c", "context", "контекст"),
        Binding("m", "tme", "t.me"),
        Binding("f", "open_file", "файл"),
        Binding("escape", "back", "назад"),
    ]

    def __init__(self):
        super().__init__()
        self.root = project_root()
        self.cfg = load_config(self.root)
        self.conn = dbm.connect(self.root / self.cfg["general"].get("db", "data/index.db"))
        self.chat = None
        self.topic = None
        self.sender = None
        self.view_mode = "chat"
        self.msgs = []
        self._prompt_kind = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(id="prompt")
        with Horizontal():
            with Vertical(id="side"):
                yield ListView(id="chats")
                yield ListView(id="topics")
            yield ListView(id="feed")
        yield Footer()

    # ---------- данные ----------

    async def on_mount(self):
        chats = self.query_one("#chats", ListView)
        items = []
        for c in VD.list_chats(self.conn):
            label = Text()
            label.append(c["title"] + "\n", style="bold")
            label.append(f"{c['n']:,} сообщ · до {(c['d1'] or '')[:10]}".replace(",", " "),
                         style="dim")
            item = ListItem(Label(label, markup=False))
            item.chat = c
            items.append(item)
        if items:
            await chats.extend(items)
            chats.index = 0
        chats.focus()

    async def open_chat(self, chat_id, around=None, date=None):
        self.chat = chat_id
        self.view_mode = "chat"
        topics = self.query_one("#topics", ListView)
        await topics.clear()
        t_items = []
        all_item = ListItem(Label("Весь чат"))
        all_item.topic = None
        t_items.append(all_item)
        for t in VD.list_topics(self.conn, chat_id)[:60]:
            label = Text()
            label.append(t["title"], style="bold")
            label.append(f"  {t['n']:,}".replace(",", " "), style="dim")
            item = ListItem(Label(label, markup=False))
            item.topic = t["topic_id"]
            t_items.append(item)
        await topics.extend(t_items)
        topics.index = 0
        await self.load_page(around=around, date=date)

    async def load_page(self, around=None, date=None):
        rows = VD.page_messages(self.conn, self.chat, self.topic, self.sender,
                                around_id=around, date=date, limit=PAGE)
        self.msgs = VD.serialize(self.conn, rows)
        self.view_mode = "chat"
        await self._render(select_mid=around)

    async def _render(self, select_mid=None, show_chat=False):
        feed = self.query_one("#feed", ListView)
        await feed.clear()
        items = [MsgItem(m, show_chat, topic_filtered=self.topic is not None)
                 for m in self.msgs]
        if items:
            await feed.extend(items)
            idx = len(items) - 1
            if select_mid is not None:
                for i, m in enumerate(self.msgs):
                    if m["message_id"] == select_mid:
                        idx = i
                        break
            feed.index = idx
        feed.focus()

    # ---------- события ----------

    async def on_list_view_selected(self, event: ListView.Selected):
        if event.list_view.id == "chats":
            self.topic = None
            self.sender = None
            await self.open_chat(event.item.chat["chat_id"])
        elif event.list_view.id == "topics":
            self.topic = event.item.topic
            await self.load_page()
        elif event.list_view.id == "feed":
            if self.view_mode == "search":
                await self.action_context()
            else:
                self.action_open_file()

    def action_prompt(self, kind):
        ph = {"search": "поиск: слова (пробел = AND, OR поддерживается)…",
              "date": "дата: 2026-05 или 2026-05-15…",
              "sender": "автор (пусто — сбросить фильтр)…"}[kind]
        self._prompt_kind = kind
        inp = self.query_one("#prompt", Input)
        inp.placeholder = ph
        inp.value = ""
        inp.styles.display = "block"
        inp.focus()

    async def on_input_submitted(self, event: Input.Submitted):
        inp = self.query_one("#prompt", Input)
        inp.styles.display = "none"
        val = event.value.strip()
        kind, self._prompt_kind = self._prompt_kind, None
        if kind == "search" and val:
            try:
                total, rows = VD.search_messages(
                    self.conn, val, chat_id=self.chat, topic_id=self.topic,
                    sender=self.sender, limit=300)
            except ValueError as e:
                self.notify(str(e), severity="error")
                return
            self.msgs = VD.serialize(self.conn, rows)
            self.view_mode = "search"
            await self._render(show_chat=True)
            scope = "по чату" if self.chat else "по всем чатам"
            self.notify(f"найдено {total} ({scope}; показаны {len(rows)}); Enter — контекст")
        elif kind == "date" and val and self.chat:
            await self.load_page(date=val)
        elif kind == "sender":
            self.sender = val or None
            self.notify(f"фильтр по автору: {val or 'снят'}")
            if self.chat:
                await self.load_page()
        else:
            self.query_one("#feed", ListView).focus()

    def _current(self):
        feed = self.query_one("#feed", ListView)
        if feed.index is not None and 0 <= feed.index < len(self.msgs):
            return self.msgs[feed.index]
        return None

    async def action_older(self):
        if self.view_mode != "chat" or not self.msgs:
            return
        first = self.msgs[0]["message_id"]
        rows = VD.page_messages(self.conn, self.chat, self.topic, self.sender,
                                before_id=first, limit=PAGE)
        if not rows:
            self.notify("это начало")
            return
        self.msgs = VD.serialize(self.conn, rows) + self.msgs
        await self._render(select_mid=first)

    async def action_newer(self):
        if self.view_mode != "chat" or not self.msgs:
            return
        last = self.msgs[-1]["message_id"]
        rows = VD.page_messages(self.conn, self.chat, self.topic, self.sender,
                                after_id=last, limit=PAGE)
        if not rows:
            self.notify("это конец")
            return
        anchor = rows[0]["message_id"]
        self.msgs = self.msgs + VD.serialize(self.conn, rows)
        await self._render(select_mid=anchor)

    async def action_context(self):
        m = self._current()
        if not m:
            return
        self.topic = None
        self.sender = None
        await self.open_chat(m["chat_id"], around=m["message_id"])

    def action_tme(self):
        m = self._current()
        if m and m.get("tme"):
            webbrowser.open(m["tme"])
            self.notify("открываю в Telegram…")

    def action_open_file(self):
        m = self._current()
        if not m or not m.get("media_file"):
            return
        path = self.root / "archive" / str(m["chat_id"]) / m["media_file"]
        if path.exists():
            subprocess.Popen(["open", str(path)])
        else:
            self.notify("файл не найден на диске", severity="warning")

    async def action_back(self):
        if self.view_mode == "search" and self.chat:
            await self.load_page()
        else:
            self.query_one("#chats", ListView).focus()


def run():
    TgTui().run()
