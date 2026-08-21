"""Транскрипция голосовых и кружков. Провайдеры сменные (config [transcription])."""

import asyncio
import os
import sys

from . import manifest as MF

PROVIDERS = ("whisper-local", "openai", "telegram", "edge")


def pick_rows(conn, chat_id=None, limit=50):
    cond = ["media_kind IN ('voice','video_note')", "transcript IS NULL", "media_file IS NOT NULL"]
    params = []
    if chat_id:
        cond.insert(0, "chat_id=?")
        params.append(chat_id)
    sql = (f"SELECT chat_id, message_id, media_file FROM messages "
           f"WHERE {' AND '.join(cond)} ORDER BY date DESC")
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, params).fetchall()


class WhisperLocal:
    def __init__(self, cfg, model=None):
        t = cfg.get("transcription", {})
        self.model_name = model or t.get("whisper_model", "small")
        self.language = t.get("language") or None
        self._impl = None

    def _load(self):
        if self._impl:
            return self._impl
        try:
            from faster_whisper import WhisperModel

            print(f"  загружаю модель whisper {self.model_name} (faster-whisper)...", file=sys.stderr)
            self._impl = ("fw", WhisperModel(self.model_name, device="auto", compute_type="int8"))
        except ImportError:
            try:
                import mlx_whisper

                self._impl = ("mlx", mlx_whisper)
            except ImportError:
                raise SystemExit(
                    "Для whisper-local установи движок: .venv/bin/pip install faster-whisper "
                    "(или mlx-whisper на Apple Silicon)"
                )
        return self._impl

    def transcribe(self, path) -> str:
        kind, impl = self._load()
        if kind == "fw":
            segments, _ = impl.transcribe(str(path), language=self.language)
            return " ".join(s.text.strip() for s in segments).strip()
        res = impl.transcribe(str(path), path_or_hf_repo=f"mlx-community/whisper-{self.model_name}-mlx")
        return (res.get("text") or "").strip()


class OpenAIProvider:
    def __init__(self, root, cfg):
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Нет OPENAI_API_KEY в .env")
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("Установи клиент: .venv/bin/pip install openai")
        self.client = OpenAI()
        t = cfg.get("transcription", {})
        self.model = t.get("openai_model", "whisper-1")
        self.language = t.get("language") or None

    def transcribe(self, path) -> str:
        kwargs = {"model": self.model}
        if self.language:
            kwargs["language"] = self.language
        with open(path, "rb") as f:
            r = self.client.audio.transcriptions.create(file=f, **kwargs)
        return (r.text or "").strip()


def run_transcribe(root, cfg, conn, provider_name=None, chat_id=None, limit=50,
                   account="default", model=None):
    provider_name = provider_name or cfg.get("transcription", {}).get("provider", "whisper-local")
    if provider_name == "none":
        raise SystemExit("В config.toml [transcription] provider = \"none\" — укажи --provider")
    if provider_name == "edge":
        raise SystemExit(
            "У Microsoft Edge нет публичного STT-эндпоинта (edge-tts умеет только синтез речи).\n"
            "Рабочие провайдеры: whisper-local (бесплатно, локально), telegram (нужен Premium), openai.\n"
            "Могу добавить azure (движок Edge, бесплатные 5 ч/мес, нужен ключ) — скажи, если надо."
        )
    if provider_name not in PROVIDERS:
        raise SystemExit(f"Неизвестный провайдер «{provider_name}»; доступны: {', '.join(PROVIDERS)}")

    rows = pick_rows(conn, chat_id, limit)
    if not rows:
        print("Нет нерасшифрованных голосовых/кружков (сначала скачай: tg media --media voice)",
              file=sys.stderr)
        return 0

    from .downloader import ChatWriter

    writers = {}

    def writer(cid):
        w = writers.get(cid)
        if w is None:
            w = writers[cid] = ChatWriter(root, cid)
        return w

    def store(cid, message_id, text):
        writer(cid).write_transcript({
            "message_id": message_id,
            "text": text,
            "provider": provider_name,
            "created": MF.now_iso(),
        })
        print(f"  #{message_id}: {text[:90]}", file=sys.stderr)

    done = 0
    if provider_name == "telegram":
        from .meta import make_client

        client = make_client(root, cfg, account)

        async def run():
            nonlocal done
            from telethon.tl.functions.messages import TranscribeAudioRequest

            async with client:
                if not await client.is_user_authorized():
                    raise SystemExit(f"Сессия «{account}» не авторизована")
                for r in rows:
                    try:
                        res = await client(TranscribeAudioRequest(peer=r["chat_id"], msg_id=r["message_id"]))
                        for _ in range(30):
                            if not res.pending:
                                break
                            await asyncio.sleep(2)
                            res = await client(TranscribeAudioRequest(peer=r["chat_id"], msg_id=r["message_id"]))
                        text = (res.text or "").strip()
                    except Exception as e:
                        if "PREMIUM" in str(e).upper():
                            raise SystemExit("Транскрипция Telegram доступна только Premium-аккаунтам — "
                                             "используй whisper-local или openai")
                        print(f"  !! #{r['message_id']}: {e}", file=sys.stderr)
                        continue
                    if text:
                        store(r["chat_id"], r["message_id"], text)
                        done += 1

        asyncio.run(run())
    else:
        prov = WhisperLocal(cfg, model) if provider_name == "whisper-local" else OpenAIProvider(root, cfg)
        for r in rows:
            path = root / "archive" / str(r["chat_id"]) / r["media_file"]
            if not path.exists():
                print(f"  !! нет файла: {path}", file=sys.stderr)
                continue
            try:
                text = prov.transcribe(path)
            except SystemExit:
                raise
            except Exception as e:
                print(f"  !! #{r['message_id']}: {e}", file=sys.stderr)
                continue
            if text:
                store(r["chat_id"], r["message_id"], text)
                done += 1

    for w in writers.values():
        w.close()
    from .indexer import index_archive

    index_archive(conn, root / cfg["general"].get("archive_dir", "archive"))
    return done
