import os
import re
import html
import time
import sqlite3
import asyncio
import difflib
import logging
import traceback
from io import BytesIO
from urllib.parse import quote_plus, urljoin, urlparse

import httpx
import feedparser
from dotenv import load_dotenv
from openai import OpenAI

from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.request import HTTPXRequest

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%G-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SwedishWallBot")

# ============================================================
# ENV
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "config", ".env")
load_dotenv(ENV_PATH)

BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
EDITOR_CHAT_ID = int((os.getenv("EDITOR_CHAT_ID") or "0").strip())
PUBLIC_CHANNEL_ID = int((os.getenv("PUBLIC_CHANNEL_ID") or "0").strip())
TELEGRAM_HANDLE = (os.getenv("TELEGRAM_HANDLE") or "@SWEzhach0k").strip()

JOB_TICK_SECONDS = int((os.getenv("JOB_TICK_SECONDS") or "360").strip())
RUN_COOLDOWN_SECONDS = int((os.getenv("RUN_COOLDOWN_SECONDS") or "300").strip())

PER_FEED_CAP = int((os.getenv("PER_FEED_CAP") or "10").strip())
MAX_PER_RUN = int((os.getenv("MAX_PER_RUN") or "1").strip())
MIN_SCORE = int((os.getenv("MIN_SCORE") or "1").strip())

OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
OPENAI_MAX_TOKENS = int((os.getenv("OPENAI_MAX_TOKENS") or "650").strip())
OPENAI_TEMPERATURE = float((os.getenv("OPENAI_TEMPERATURE") or "0.2").strip())

DB_PATH = os.path.join(BASE_DIR, "posted_items.sqlite")

# Turn OFF Telegram web previews
DISABLE_PREVIEWS = True
AUTO_POST = (os.getenv("AUTO_POST", "false").lower().strip() == "true")

if not BOT_TOKEN or not OPENAI_API_KEY or not EDITOR_CHAT_ID or not PUBLIC_CHANNEL_ID:
    raise RuntimeError(
        f"Missing env vars. Check {ENV_PATH}\n"
        f"Required: BOT_TOKEN, OPENAI_API_KEY, EDITOR_CHAT_ID, PUBLIC_CHANNEL_ID"
    )

# ============================================================
# SINGLE INSTANCE LOCK
# ============================================================

LOCK_PATH = os.path.join(BASE_DIR, ".bot.lock")


def acquire_lock_or_exit() -> None:
    if os.path.exists(LOCK_PATH):
        try:
            with open(LOCK_PATH, "r", encoding="utf-8") as f:
                pid_str = f.read().strip()
            if pid_str.isdigit():
                pid = int(pid_str)
                os.kill(pid, 0)
                raise SystemExit(f"[LOCK] Another bot instance is running (PID={pid}). Stop it first.")
        except ProcessLookupError:
            try:
                os.remove(LOCK_PATH)
            except Exception:
                pass
        except Exception:
            try:
                os.remove(LOCK_PATH)
            except Exception:
                pass

    with open(LOCK_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def release_lock() -> None:
    try:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
    except Exception:
        pass


# ============================================================
# RSS
# ============================================================

feedparser.USER_AGENT = "SwedishWallBot/10.0 (+https://t.me/SWEzhach0k)"


def google_news_rss(q: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=sv&gl=SE&ceid=SE:sv"


RSS_FEEDS = [
    ("SVT Nyheter", "https://www.svt.se/nyheter/rss.xml"),
    ("SR Ekot", "https://api.sr.se/api/rss/pod/3795"),
    ("8 Sidor", "https://8sidor.se/feed/"),
    ("Expressen Nyheter", "https://feeds.expressen.se/nyheter/"),
    ("Government.se via Google", google_news_rss("site:government.se")),
    ("TV4.se via Google", google_news_rss("site:tv4.se")),
    ("Expressen Debatt", "https://feeds.expressen.se/debatt/"),
    ("Aftonbladet", "https://rss.aftonbladet.se/rss2/small/pages/sections/senastenytt/"),
    # Optional DN coverage:
    # ("DN via Google", google_news_rss("site:dn.se")),
]

KEYWORDS = [
    "ukraina", "ryssland", "kriget", "putin", "zelensky",
    "nato", "försvar", "säkerhet", "sverige", "svensk",
    "regeringen", "riksdagen", "saab", "gripen"
]

HOT_TERMS = [
    "akut", "varnar", "skandal", "explosion", "attack", "ryssland", "ukraina", "krig",
    "olja", "gas", "nato", "ekonomi", "bistånd", "sverige", "regering", "putin",
    "zelenskyj", "brott", "migration", "drönare", "kultur", "musik",
    "melodifestivalen", "eurovision", "esc", "brand", "våldtäkt", "utvisning",
    "lån", "ränta", "riksbank", "euro", "dollar", "eu"
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


def score_entry(title: str, summary: str) -> int:
    text = normalize((title or "") + " " + (summary or ""))
    score = 0
    for kw in KEYWORDS:
        if kw in text:
            score += 3
    for ht in HOT_TERMS:
        if ht in text:
            score += 2
    return score


def detect_article_type(source_name: str, title: str, link: str) -> str:
    t = (source_name + " " + (title or "") + " " + (link or "")).lower()
    if any(x in t for x in ["debatt", "ledare", "opinion"]):
        return "debate"
    return "news"


async def safe_get(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    """Robust GET with retries for transient errors."""
    retries = 3
    backoff = 1.5
    for i in range(retries):
        try:
            resp = await client.get(url, **kwargs)
            resp.raise_for_status()
            return resp
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
            if i == retries - 1:
                raise
            wait = backoff * (2 ** i)
            logger.warning(f"[RETRY] {url} failed ({e}), waiting {wait:.1f}s...")
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500 and i < retries - 1:
                wait = backoff * (2 ** i)
                logger.warning(f"[RETRY] {url} status {e.response.status_code}, waiting {wait:.1f}s...")
                await asyncio.sleep(wait)
                continue
            raise
    raise httpx.RequestError("Max retries exceeded", request=None)


async def fetch_feed(client: httpx.AsyncClient, url: str) -> feedparser.FeedParserDict:
    resp = await safe_get(client, url, timeout=20.0)
    return feedparser.parse(resp.content)


def extract_item_id(entry) -> str:
    link = (entry.get("link") or "").strip()
    eid = (entry.get("id") or entry.get("guid") or link or "").strip()
    return eid


async def verify_link(client: httpx.AsyncClient, url: str) -> bool:
    """Check if the source URL actually works (200 OK) before processing."""
    try:
        resp = await client.head(url, timeout=10.0, follow_redirects=True)
        if resp.status_code == 405: # HEAD not allowed
            resp = await client.get(url, timeout=10.0, follow_redirects=True)
        return resp.status_code < 400
    except Exception as e:
        logger.debug(f"Link verification failed for {url}: {e}")
        return False


def strip_html_text(s: str) -> str:
    s = s or ""
    s = html.unescape(s)
    s = re.sub(r"<\s*br\s*/?\s*>", "\n", s, flags=re.I)
    s = re.sub(r"</p\s*>", "\n\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# PARSE PHOTO FROM ORIGINAL ARTICLE
# ============================================================

META_OG_IMAGE_RE = re.compile(
    r'<meta[^>]+(?:property|name)\s*=\s*["\'](?:og:image|twitter:image|twitter:image:src)["\'][^>]+content\s*=\s*["\']([^"\']+)["\']',
    re.I
)
META_OG_IMAGE_ALT_RE = re.compile(
    r'<meta[^>]+content\s*=\s*["\']([^"\']+)["\'][^>]+(?:property|name)\s*=\s*["\'](?:og:image|twitter:image|twitter:image:src)["\']',
    re.I
)
IMG_SRC_RE = re.compile(r"<img[^>]+src\s*=\s*['\"]([^'\"]+)['\"]", re.I)


async def fetch_article_body(client: httpx.AsyncClient, url: str) -> str:
    """Fetches the actual HTML article and extracts paragraph text for better AI context."""
    try:
        resp = await safe_get(client, url, timeout=15.0)
        html_content = resp.text
        # Extract content from <p> tags using regex to avoid heavy dependencies like BeautifulSoup
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html_content, re.S)
        text_parts = []
        for p in paragraphs:
            # Clean up tags inside the paragraph
            clean_p = re.sub(r'<[^>]+>', '', p)
            clean_p = html.unescape(clean_p).strip()
            # Basic heuristic: skip nav/small fragments
            if len(clean_p) > 40:
                text_parts.append(clean_p)
        
        # Join first few relevant paragraphs
        body = "\n".join(text_parts[:15])
        return body or ""
    except Exception as e:
        logger.debug(f"Failed to fetch article body for {url}: {e}")
        return ""


async def fetch_article_image(client: httpx.AsyncClient, article_url: str) -> str:
    u = (article_url or "").strip()
    if not u:
        return ""

    try:
        resp = await safe_get(
            client,
            u,
            timeout=20.0,
            headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},
        )
        html_text = resp.text or ""

        m = META_OG_IMAGE_RE.search(html_text) or META_OG_IMAGE_ALT_RE.search(html_text)
        if m:
            img = (m.group(1) or "").strip()
            if img:
                return urljoin(u, img)

        m2 = IMG_SRC_RE.search(html_text)
        if m2:
            img = (m2.group(1) or "").strip()
            if img:
                return urljoin(u, img)
    except Exception as e:
        logger.debug(f"Failed to fetch image from {u}: {e}")

    return ""


# ============================================================
# IMAGE FILTERING (Mode A: if not usable -> no image)
# ============================================================

BLOCKED_IMAGE_KEYWORDS = [
    "logo", "logos", "logotyp", "logotype", "icon", "icons", "favicon",
    "sprite", "badge", "default", "placeholder", "fallback", "share-default",
    "app-icon", "apple-touch-icon", "msapplication", "siteicon", "brand",
    "sverigesradio", "ekot_logo", "googlelogo", "google_logo"
]

BLOCKED_IMAGE_EXTENSIONS = [".svg"]

BLOCKED_IMAGE_HOSTS = {
    "news.google.com",
    "www.google.com",
}


def _looks_like_logo_url(image_url: str) -> bool:
    u = (image_url or "").strip()
    if not u:
        return True
    try:
        p = urlparse(u)
        host = (p.netloc or "").lower()
        path = (p.path or "").lower()
        query = (p.query or "").lower()
    except Exception:
        return True

    if host in BLOCKED_IMAGE_HOSTS:
        return True

    for ext in BLOCKED_IMAGE_EXTENSIONS:
        if path.endswith(ext):
            return True

    hay = f"{host} {path} {query}"
    return any(k in hay for k in BLOCKED_IMAGE_KEYWORDS)


def _image_dimensions_from_bytes(data: bytes) -> tuple[int, int]:
    if not data or len(data) < 16:
        return (0, 0)

    # PNG
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        w = int.from_bytes(data[16:20], "big", signed=False)
        h = int.from_bytes(data[20:24], "big", signed=False)
        return (w, h)

    # GIF
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        w = int.from_bytes(data[6:8], "little", signed=False)
        h = int.from_bytes(data[8:10], "little", signed=False)
        return (w, h)

    # WEBP
    if data[:4] == b"RIFF" and len(data) >= 30 and data[8:12] == b"WEBP":
        if data[12:16] == b"VP8X" and len(data) >= 30:
            w = 1 + int.from_bytes(data[24:27], "little", signed=False)
            h = 1 + int.from_bytes(data[27:30], "little", signed=False)
            return (w, h)
        return (0, 0)

    # JPEG
    if data.startswith(b"\xff\xd8"):
        i = 2
        n = len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            i += 2
            while marker == 0xFF and i < n:
                marker = data[i]
                i += 1
            if marker in (0xD8, 0xD9):
                continue
            if i + 2 > n:
                break
            seglen = int.from_bytes(data[i:i + 2], "big", signed=False)
            if seglen < 2 or i + seglen > n:
                break
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                if i + 7 <= n:
                    h = int.from_bytes(data[i + 3:i + 5], "big", signed=False)
                    w = int.from_bytes(data[i + 5:i + 7], "big", signed=False)
                    return (w, h)
                break
            i += seglen
        return (0, 0)

    return (0, 0)


async def is_usable_image(client: httpx.AsyncClient, image_url: str) -> bool:
    u = (image_url or "").strip()
    if not u:
        return False
    if _looks_like_logo_url(u):
        return False

    try:
        # Use stream to only probe the header
        async with client.stream(
            "GET",
            u,
            timeout=20.0,
            headers={
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Referer": u,
            },
        ) as r:
            r.raise_for_status()
            buf = b""
            max_probe = 256 * 1024
            async for chunk in r.aiter_bytes():
                buf += chunk
                if len(buf) >= max_probe:
                    break
                w, h = _image_dimensions_from_bytes(buf)
                if w and h:
                    break

            w, h = _image_dimensions_from_bytes(buf)
            if w == 0 or h == 0:
                return True

            if w < 420 or h < 240:
                return False

            aspect = w / h if h else 0.0
            if 0.85 <= aspect <= 1.15 and max(w, h) < 900:
                return False

            return True
    except Exception as e:
        logger.debug(f"is_usable_image failed for {u}: {e}")
        return False


async def download_image_bytes(client: httpx.AsyncClient, image_url: str, max_bytes: int = 12_000_000) -> tuple[bytes, str]:
    u = (image_url or "").strip()
    if not u:
        raise ValueError("empty image url")

    async with client.stream(
        "GET",
        u,
        timeout=25.0,
        headers={
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": u,
        },
    ) as r:
        r.raise_for_status()
        ct = (r.headers.get("Content-Type") or "").lower()
        chunks: list[bytes] = []
        total = 0
        async for chunk in r.aiter_bytes():
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("image too large")

    data = b"".join(chunks)

    if "image" not in ct:
        if not (data.startswith(b"\xff\xd8\xff") or data.startswith(b"\x89PNG") or data[:4] == b"RIFF" or data[:4] == b"GIF8"):
            raise ValueError(f"not an image (content-type={ct or 'unknown'})")

    ext = ".jpg"
    if "png" in ct or data.startswith(b"\x89PNG"):
        ext = ".png"
    elif "webp" in ct or (data[:4] == b"RIFF" and data[8:12] == b"WEBP"):
        ext = ".webp"
    elif "gif" in ct or data[:4] == b"GIF8":
        ext = ".gif"
    elif "jpeg" in ct or "jpg" in ct or data.startswith(b"\xff\xd8\xff"):
        ext = ".jpg"

    return data, f"photo{ext}"


# ============================================================
# HASHTAGS
# ============================================================

HASHTAG_CORE = ["#Швеция", "#Новости"]

HASHTAG_TOPICS = {
    "ukraine_russia": ["#Украина", "#Россия", "#Война", "#Безопасность", "#Санкции"],
    "nato_defense": ["#НАТО", "#Оборона", "#Армия", "#Безопасность", "#СевернаяЕвропа"],
    "politics": ["#Политика", "#Правительство", "#Риксдаг", "#Партии"],
    "crime": ["#Криминал", "#Полиция", "#Суд", "#Преступность", "#Безопасность"],
    "migration": ["#Миграция", "#Беженцы", "#Границы", "#Интеграция"],
    "economy": ["#Экономика", "#Инфляция", "#Цены", "#Налоги", "#Бюджет"],
    "rates": ["#Риксбанк", "#Ставки", "#Ипотека", "#Кредиты", "#Финансы"],
    "energy": ["#Энергетика", "#Газ", "#Нефть", "#Электричество", "#Климат"],
    "eu": ["#ЕС", "#Европа", "#Брюссель"],
    "foreign": ["#Дипломатия", "#МеждународныеОтношения", "#Геополитика"],
    "tech": ["#Технологии", "#Кибербезопасность", "#ИИ", "#Интернет"],
    "health": ["#Здравоохранение", "#Медицина", "#Больницы", "#Врачи"],
    "education": ["#Образование", "#Школа", "#Университет"],
    "society": ["#Общество", "#СоциальнаяПолитика", "#Права"],
    "environment": ["#Экология", "#Климат", "#Погода"],
    "transport": ["#Транспорт", "#ЖД", "#Авиасообщение", "#Дороги"],
    "culture": ["#Культура", "#Музыка", "#Кино", "#Телевидение", "#Мелодифестивален", "#Евровидение"],
    "sports": ["#Спорт", "#Футбол", "#Хоккей"],
}

TOPIC_TRIGGERS = {
    "ukraine_russia": ["украин", "киев", "зеленск", "росси", "путин", "войн", "ukraina", "ryssland", "zelensky", "putin", "krig"],
    "nato_defense": ["nato", "нато", "обор", "арм", "saab", "gripen", "försvar", "säkerhet", "försvaret", "försvars"],
    "politics": ["правитель", "министр", "парламент", "риксдаг", "выбор", "regering", "riksdag", "val", "minister"],
    "crime": ["убийств", "взрыв", "стрельб", "насил", "суд", "полици", "brott", "polis", "skjut", "explosion", "våldtäkt", "domstol"],
    "migration": ["миграц", "беженц", "убежищ", "депортац", "migration", "asyl", "utvis", "flykting"],
    "economy": ["эконом", "инфляц", "цены", "налог", "бюджет", "ekonomi", "inflation", "pris", "skatt", "budget"],
    "rates": ["риксбанк", "ставк", "ипотек", "кредит", "процент", "riksbank", "ränta", "lån", "bolån"],
    "energy": ["газ", "нефть", "электр", "энерг", "gas", "olja", "elpris", "energi"],
    "eu": ["ес", "евросоюз", "eu", "europarlament"],
    "foreign": ["диплом", "переговор", "посол", "геополит", "utrikes", "diplomati"],
    "tech": ["кибер", "хакер", "утечк", "ии", "искусственн", "cyber", "hack", "läcka", "ai"],
    "health": ["медицин", "больниц", "врач", "covid", "sjukhus", "vård"],
    "education": ["школ", "универс", "student", "skola", "universitet"],
    "society": ["обще", "социал", "пособи", "välfärd", "social"],
    "environment": ["климат", "эколог", "погод", "klimat", "miljö", "väder"],
    "transport": ["транспорт", "поезд", "самолет", "trafik", "tåg", "flyg"],
    "culture": ["культур", "кино", "музык", "евровиден", "melodifestivalen", "eurovision", "esc"],
    "sports": ["спорт", "футбол", "хокке", "sport", "fotboll", "hockey"],
}


def pick_hashtags(rss_title: str, rss_summary: str, source: str) -> list[str]:
    t = normalize((rss_title or "") + " " + (rss_summary or "") + " " + (source or ""))
    tags: list[str] = []
    tags.extend(HASHTAG_CORE)

    matched_topics = []
    for topic, triggers in TOPIC_TRIGGERS.items():
        if any(tr in t for tr in triggers):
            matched_topics.append(topic)

    for topic in matched_topics[:3]:
        tags.extend(HASHTAG_TOPICS.get(topic, []))

    seen = set()
    uniq = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            uniq.append(tag)

    return uniq[:12]


# ============================================================
# DB
# ============================================================

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, ddl_type: str) -> None:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl_type}")
        conn.commit()


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS drafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            text TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS posted (
            item_id TEXT PRIMARY KEY,
            posted_at TEXT NOT NULL,
            title TEXT,
            headline TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source TEXT,
            item_id TEXT,
            stage TEXT NOT NULL,
            error TEXT NOT NULL
        )
    """)
    conn.commit()

    _ensure_column(conn, "drafts", "image_url", "TEXT")
    _ensure_column(conn, "posted", "title", "TEXT")
    _ensure_column(conn, "posted", "headline", "TEXT")
    return conn


# Tracking sequential failures for Health Alerts
FEED_HEALTH = {} # {source_name: sequential_failure_count}


def backup_db():
    """Simple daily backup of the database."""
    import shutil
    if not os.path.exists(DB_PATH):
        return
    backup_name = f"{DB_PATH}.{time.strftime('%Y%m%d')}.bak"
    if not os.path.exists(backup_name):
        try:
            shutil.copy2(DB_PATH, backup_name)
            logger.info(f"[BACKUP] Created {backup_name}")
        except Exception as e:
            logger.error(f"[BACKUP] Failed: {e}")


def already_posted(conn: sqlite3.Connection, item_id: str) -> bool:
    c = conn.cursor()
    c.execute("SELECT 1 FROM posted WHERE item_id=?", (item_id,))
    return c.fetchone() is not None


def get_recent_titles(conn: sqlite3.Connection, limit: int = 100) -> list[str]:
    c = conn.cursor()
    c.execute("SELECT title FROM posted WHERE title IS NOT NULL AND title != '' ORDER BY posted_at DESC LIMIT ?",
              (limit,))
    return [row[0] for row in c.fetchall()]


def is_similar(text1: str, text2: str, threshold: float = 0.7) -> bool:
    if not text1 or not text2:
        return False

    def clean(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"^(expressen|aftonbladet|svt|sr|dn|svd|8 sidor|sverige|nyheter|radio|ekot)[:\-\s]+", "", s, flags=re.I)
        s = re.sub(r"[^\w\s]", "", s)
        return s.strip()

    t1, t2 = clean(text1), clean(text2)
    if not t1 or not t2:
        return False
    if t1 == t2:
        return True
    return difflib.SequenceMatcher(None, t1, t2).ratio() >= threshold


async def is_semantically_similar(client: OpenAI, new_headline: str, recent_headlines: list[str]) -> bool:
    """Uses AI to check if a new headline covers the same story as recent ones."""
    if not recent_headlines:
        return False

    recent_str = "\n".join([f"- {h}" for h in recent_headlines[:10]])
    prompt = f"""
Compare the new headline with the list of recent news headlines below.
Decide if the new headline is about the same story/event.

Recent headlines:
{recent_str}

New headline:
{new_headline}

Respond ONLY with 'YES' if it's the same story, or 'NO' if it's a different story.
""".strip()

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        answer = (resp.choices[0].message.content or "").strip().upper()
        return "YES" in answer
    except Exception as e:
        logger.error(f"Semantic deduplication failed: {e}")
        return False


def get_recent_headlines(conn: sqlite3.Connection, limit: int = 50) -> list[str]:
    c = conn.cursor()
    c.execute("SELECT headline FROM posted WHERE headline IS NOT NULL AND headline != '' ORDER BY posted_at DESC LIMIT ?",
              (limit,))
    return [row[0] for row in c.fetchall()]


def mark_posted(conn: sqlite3.Connection, item_id: str, title: str = "", headline: str = "") -> None:
    conn.execute("INSERT OR IGNORE INTO posted (item_id, posted_at, title, headline) VALUES (?, ?, ?, ?)",
                 (item_id, utc_now_iso(), title, headline))
    conn.commit()


def save_failure(conn: sqlite3.Connection, source: str, item_id: str, stage: str, error: str) -> None:
    conn.execute(
        "INSERT INTO failures (created_at, source, item_id, stage, error) VALUES (?, ?, ?, ?, ?)",
        (utc_now_iso(), source, item_id, stage, (error or "")[:2000])
    )
    conn.commit()


def save_draft(conn: sqlite3.Connection, msg_html: str, status: str = "pending", error: str | None = None,
               image_url: str = "") -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO drafts (created_at, text, status, error, image_url) VALUES (?, ?, ?, ?, ?)",
        (utc_now_iso(), msg_html, status, error, (image_url or "").strip())
    )
    conn.commit()
    return int(cur.lastrowid)


# ============================================================
# TELEGRAM HELPERS
# ============================================================

def hard_clip(text: str, max_len: int = 3800) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 20] + "\n\n…(truncated)"


def strip_html_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")


async def send_html_safe(bot, chat_id: int, html_text: str) -> None:
    html_text = hard_clip(html_text, 3800)
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=html_text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=DISABLE_PREVIEWS,
        )
    except Exception:
        plain = hard_clip(strip_html_tags(html_text), 3800)
        await bot.send_message(chat_id=chat_id, text=plain)


async def publish_to_channel(bot, chat_id: int, text: str, image_url: str = "", client: httpx.AsyncClient = None) -> None:
    image_url = (image_url or "").strip()

    if image_url and client:
        try:
            await bot.send_photo(chat_id=chat_id, photo=image_url)
        except Exception as e_url:
            msg = str(e_url)
            if "failed to get http url content" in msg.lower():
                try:
                    data, fname = await download_image_bytes(client, image_url)
                    bio = BytesIO(data)
                    bio.name = fname
                    await bot.send_photo(chat_id=chat_id, photo=InputFile(bio, filename=fname))
                except Exception as e2:
                    logger.warning(f"Fallback photo upload failed: {e2}")
    elif image_url:
        # Fallback if client is missing (should not happen in main flow)
        try:
            await bot.send_photo(chat_id=chat_id, photo=image_url)
        except Exception:
            pass

    await bot.send_message(
        chat_id=chat_id,
        text=hard_clip(text, 3900),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=DISABLE_PREVIEWS
    )


def make_clickable_read_link(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    u_esc = html.escape(u, quote=True)
    return f'<a href="{u_esc}">Читать</a>'


def build_message_html(headline: str, summary: str, details: str, source: str, link: str, rss_title: str,
                       rss_summary: str, ai_hashtags: list[str] = None) -> str:
    h = html.escape((headline or "").strip())
    s = html.escape((summary or "").strip())
    d = html.escape((details or "").strip())
    src = html.escape((source or "").strip())
    read = make_clickable_read_link(link)

    if ai_hashtags:
        final_tags = []
        for t in ai_hashtags:
            if not t.startswith("#"):
                t = "#" + t
            final_tags.append(t)

        for c in HASHTAG_CORE:
            if c not in final_tags:
                final_tags.append(c)

        hashtags = " ".join(final_tags[:12])
    else:
        hashtags = " ".join(pick_hashtags(rss_title, rss_summary, source))

    return (
        f"{h}\n\n"
        f"{s}\n\n"
        f"<blockquote expandable>{d}</blockquote>\n\n"
        f"<b>Source:</b> {src}\n"
        f"<b>Link:</b> {read}\n\n"
        f"{hashtags}\n"
        f"{html.escape(TELEGRAM_HANDLE)}"
    )


# ============================================================
# OPENAI + BULLETPROOF RUSSIAN-ONLY
# ============================================================

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

# Allowlist of Latin tokens that are acceptable (kept tiny; everything else gets scrubbed)
LATIN_ALLOWLIST = {
    "EU", "NATO", "USA", "UN", "WHO", "IMF", "ECB", "SVT", "SR", "DN", "TV4",
    "G7", "G20", "COVID", "COVID-19", "SEK", "OECD", "OSCE"
}

# Prefer Cyrillic replacements for common acronyms (prettier than transliteration)
ACRONYM_CYR_MAP = {
    "EU": "ЕС",
    "NATO": "НАТО",
    "USA": "США",
    "UN": "ООН",
    "WHO": "ВОЗ",
    "SVT": "СВТ",
    "SR": "СР",
    "DN": "ДН",
    "TV4": "ТВ4",
    "G7": "Г7",
    "G20": "Г20",
    "COVID": "КОВИД",
    "COVID-19": "КОВИД-19",
    "SEK": "СЕК",
    "OECD": "ОЭСР",
    "OSCE": "ОБСЕ",
}

# Minimal deterministic transliteration to eliminate Latin (not perfect Russian spelling, but no A-Za-z remains)
_LAT2CYR = str.maketrans({
    "A": "А", "B": "Б", "C": "К", "D": "Д", "E": "Е", "F": "Ф", "G": "Г", "H": "Х", "I": "И", "J": "Й",
    "K": "К", "L": "Л", "M": "М", "N": "Н", "O": "О", "P": "П", "Q": "К", "R": "Р", "S": "С", "T": "Т",
    "U": "У", "V": "В", "W": "В", "X": "Кс", "Y": "Й", "Z": "З",
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и", "j": "й",
    "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т",
    "u": "у", "v": "в", "w": "в", "x": "кс", "y": "й", "z": "з",
    "å": "о", "Å": "О", "ä": "е", "Ä": "Е", "ö": "е", "Ö": "Е",
})


def is_russian_enough(text: str) -> bool:
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text or "")
    if not letters:
        return False
    cyr = sum(1 for ch in letters if CYRILLIC_RE.match(ch))
    return (cyr / len(letters)) >= 0.80


def find_latin_words(text: str) -> list[str]:
    return re.findall(r"\b[A-Za-z][A-Za-z\-]{1,}\b", text or "")


def has_unwanted_latin(text: str) -> bool:
    words = find_latin_words(text)
    if not words:
        return False
    for w in words:
        if w.upper() in LATIN_ALLOWLIST:
            continue
        return True
    return False


def unwanted_latin_words(text: str) -> list[str]:
    words = find_latin_words(text)
    bad = []
    for w in words:
        if w.upper() in LATIN_ALLOWLIST:
            continue
        bad.append(w)
    return bad


def scrub_to_russian_only(text: str) -> str:
    """
    Deterministically remove Latin:
    1) replace known acronyms with Cyrillic equivalents
    2) transliterate remaining Latin letters (incl. åäö) to Cyrillic lookalikes
    """
    if not text:
        return text

    # Replace acronyms first (word boundaries, case-insensitive)
    if ACRONYM_CYR_MAP:
        pattern = r"\b(" + "|".join(re.escape(k) for k in sorted(ACRONYM_CYR_MAP.keys(), key=len, reverse=True)) + r")\b"

        def repl(m: re.Match) -> str:
            token = m.group(0)
            return ACRONYM_CYR_MAP.get(token.upper(), token)

        text = re.sub(pattern, repl, text, flags=re.I)

    # Transliterate any remaining latin letters
    text = text.translate(_LAT2CYR)
    return text


def extract_block(raw: str, label: str) -> str:
    m = re.search(rf"{label}:\s*\n(.*?)(?=\n[A-Z]+:\s*\n|\Z)", raw, flags=re.S)
    return (m.group(1).strip() if m else "")


RATE_WAIT_RE = re.compile(r"try again in\s+(\d+)m(\d+(?:\.\d+)?)s", re.I)


def parse_rate_limit_wait_seconds(msg: str) -> int:
    m = RATE_WAIT_RE.search(msg or "")
    if not m:
        return 60
    minutes = int(m.group(1))
    seconds = float(m.group(2))
    return max(10, int(minutes * 60 + seconds) + 3)


def openai_strict_blocks(client: OpenAI, source: str, title: str, rss_summary: str, link: str, article_type: str) -> str:
    if article_type == "news":
        format_rules = (
            "- HEADLINE: 1 line, starts with exactly 1 emoji, 6–14 words.\n"
            "- SUMMARY: 2–4 sentences, 70–110 words.\n"
            "- DETAILS: 8–12 short lines, 900–1600 characters.\n"
            "- HASHTAGS: 3-6 space-separated tags.\n"
        )
    else:
        format_rules = (
            "- HEADLINE: 1 line, starts with exactly 1 emoji, 6–14 words.\n"
            "- SUMMARY: 2 short paragraphs, 80–150 words total.\n"
            "- DETAILS: 10–14 short lines, 1000–1800 characters.\n"
            "- HASHTAGS: 3-6 space-separated tags.\n"
        )

    prompt = f"""
Ты пишешь для русскоязычного Telegram-канала о Швеции. Твоя задача — сделать максимально точный, интересный и фактически верный пост.

СТРОГО:
- Пиши ТОЛЬКО по-русски.
- Используй ТОЛЬКО факты из предоставленного текста.
- Если информации мало, не выдумывай "дипломатическую вежливость" или общие фразы.
- Пиши КОНКРЕТНО: имена, цифры, даты, планы.
- Никакого Markdown/HTML.
- Верни РОВНО 5 блоков с метками: THINKING / HEADLINE / SUMMARY / DETAILS / HASHTAGS.

Требования:
{format_rules}

Источник: {source}
Заголовок: {title}
Контент для анализа:
{rss_summary}

Ссылка: {link}
""".strip()

    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system",
             "content": "Return Russian only. Exactly 5 blocks: THINKING, HEADLINE, SUMMARY, DETAILS, HASHTAGS."},
            {"role": "user", "content": prompt},
        ],
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
    )
    return (r.choices[0].message.content or "").strip()


def openai_translate_compose(client: OpenAI, title: str, rss_summary: str, article_type: str) -> tuple[str, str, str, list[str]]:
    t = (title or "").strip()
    s = (rss_summary or "").strip()

    prompt = f"""
Сделай короткий пост на русском (только кириллица), без английских/шведских слов.

Дано:
- Заголовок: {t}
- Описание RSS: {s}

Сделай:
1) Заголовок (1 строка) — начни с одного эмодзи.
2) Краткое резюме (2–4 предложения, 70–110 слов).
3) Детали (8–12 коротких строк, без ссылок, без HTML/Markdown).
4) Хештеги (3-5 штук через пробел).

Тип: {"новость" if article_type == "news" else "мнение/дебаты"}.

Верни в таком виде:
ЗАГОЛОВОК: ...
РЕЗЮМЕ: ...
ДЕТАЛИ:
- ...
- ...
ХЕШТЕГИ: ...
""".strip()

    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Russian Cyrillic only. No Latin words. No HTML."},
            {"role": "user", "content": prompt},
        ],
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
    )
    raw = (r.choices[0].message.content or "").strip()

    headline = ""
    summary = ""
    details = ""

    m1 = re.search(r"ЗАГОЛОВОК:\s*(.+)", raw)
    m2 = re.search(r"РЕЗЮМЕ:\s*(.+)", raw)
    m3 = re.search(r"ДЕТАЛИ:\s*(.+?)(?=\nХЕШТЕГИ:|\Z)", raw, flags=re.S)
    m4 = re.search(r"ХЕШТЕГИ:\s*(.+)", raw)

    if m1:
        headline = m1.group(1).strip()
    if m2:
        summary = m2.group(1).strip()
    if m3:
        details = m3.group(1).strip()

    hashtags = []
    if m4:
        tags_raw = m4.group(1).strip()
        for t in tags_raw.split():
            clean_t = t.strip("#., ")
            if clean_t:
                hashtags.append(clean_t)

    details = re.sub(r"^\s*[-•]\s*", "• ", details, flags=re.M).strip()
    details = details.split("ХЕШТЕГИ:")[0].strip()
    details = re.sub(r"^\s*[-•]\s*", "• ", details, flags=re.M).strip()

    if not headline or not summary or not details:
        raise ValueError("OpenAI response missing HEADLINE, SUMMARY, or DETAILS.")

    # Deterministic scrub before validation
    headline = scrub_to_russian_only(headline)
    summary = scrub_to_russian_only(summary)
    details = scrub_to_russian_only(details)

    joined = f"{headline}\n{summary}\n{details}"
    if not is_russian_enough(joined) or has_unwanted_latin(joined):
        bad = unwanted_latin_words(joined)
        raise ValueError(f"OpenAI response was not consistent Russian or contained too much Latin (bad={bad[:12]}).")

    return headline, summary, details, hashtags


def openai_rewrite_russian_only(client: OpenAI, headline: str, summary: str, details: str) -> tuple[str, str, str]:
    """
    Second-chance rewrite: keep meaning, remove Latin, do NOT add facts.
    """
    prompt = f"""
Перепиши этот текст так, чтобы он был ТОЛЬКО на русском языке (кириллица).

СТРОГО:
- Не добавляй факты, цифры, имена, детали, которых нет в тексте.
- Если встречаются латинские слова (например, названия), замени их на русский эквивалент или транслитерацию кириллицей.
- Сохрани структуру из 3 частей.

Верни ровно в таком формате:
ЗАГОЛОВОК: ...
РЕЗЮМЕ: ...
ДЕТАЛИ:
- ...
- ...

Текст:
Заголовок: {headline}
Резюме: {summary}
Детали: {details}
""".strip()

    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Russian Cyrillic only. No Latin. No new facts."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=700,
    )
    raw = (r.choices[0].message.content or "").strip()

    m1 = re.search(r"ЗАГОЛОВОК:\s*(.+)", raw)
    m2 = re.search(r"РЕЗЮМЕ:\s*(.+)", raw)
    m3 = re.search(r"ДЕТАЛИ:\s*(.+)", raw, flags=re.S)

    nh = (m1.group(1).strip() if m1 else "").strip()
    ns = (m2.group(1).strip() if m2 else "").strip()
    nd = (m3.group(1).strip() if m3 else "").strip()

    nd = re.sub(r"^\s*[-•]\s*", "• ", nd, flags=re.M).strip()

    # Deterministic scrub
    nh = scrub_to_russian_only(nh)
    ns = scrub_to_russian_only(ns)
    nd = scrub_to_russian_only(nd)

    joined = f"{nh}\n{ns}\n{nd}"
    if not nh or not ns or not nd:
        raise ValueError("rewrite missing sections")
    if not is_russian_enough(joined) or has_unwanted_latin(joined):
        bad = unwanted_latin_words(joined)
        raise ValueError(f"rewrite still has Latin (bad={bad[:12]})")

    return nh, ns, nd


async def verify_facts(client: OpenAI, swedish_title: str, swedish_summary: str, russian_content: str) -> bool:
    """Cross-references Russian summary against Swedish source to catch hallucinations."""
    prompt = f"""
Verify if the Russian summary below contains any factual contradictions or hallucinations compared to the original Swedish source.
Common errors to check: names, dates, numbers, sports scores, or swapping of locations/countries.

Swedish source:
Title: {swedish_title}
Summary: {swedish_summary}

Russian content:
{russian_content}

Does the Russian content contain factual errors OR info NOT present in Swedish source?
Respond ONLY with 'VALID' if it is correct, or 'ERROR: <reason>' if there is a factual issue.
""".strip()

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60,
        )
        answer = (resp.choices[0].message.content or "").strip().upper()
        if "VALID" in answer:
            return True
        logger.warning(f"[FACT CHECK] Failed: {answer}")
        return False
    except Exception as e:
        logger.error(f"Fact verification failed: {e}")
        return True # Default to True to avoid blocking on API errors


async def generate_post(client: OpenAI, source: str, title: str, rss_summary_raw: str, link: str, article_type: str) -> tuple[str, str]:
    rss_summary = strip_html_text(rss_summary_raw)

    raw = openai_strict_blocks(client, source, title, rss_summary, link, article_type)

    has_labels = (
        re.search(r"\bHEADLINE:\s*\n", raw) and
        re.search(r"\bSUMMARY:\s*\n", raw) and
        re.search(r"\bDETAILS:\s*\n", raw)
    )

    if has_labels:
        headline = extract_block(raw, "HEADLINE")
        summ = extract_block(raw, "SUMMARY")
        details = extract_block(raw, "DETAILS")
        hashtags_raw = extract_block(raw, "HASHTAGS")

        ai_hashtags = []
        if hashtags_raw:
            for t in hashtags_raw.split():
                clean_t = t.strip("#., ")
                if clean_t:
                    ai_hashtags.append(clean_t)

        # Scrub + validate
        headline_s = scrub_to_russian_only(headline)
        summ_s = scrub_to_russian_only(summ)
        details_s = scrub_to_russian_only(details)

        joined = f"{headline_s}\n{summ_s}\n{details_s}"
        if headline_s and summ_s and details_s and is_russian_enough(joined) and not has_unwanted_latin(joined):
            if len(headline_s + summ_s + details_s) >= 260:
                msg_html = build_message_html(
                    headline=headline_s,
                    summary=summ_s,
                    details=details_s,
                    source=source,
                    link=link,
                    rss_title=title,
                    rss_summary=rss_summary,
                    ai_hashtags=ai_hashtags
                )
                return msg_html, headline_s

        # One rewrite retry if anything looks off
        if headline and summ and details:
            rh, rs, rd = openai_rewrite_russian_only(client, headline, summ, details)
            msg_html = build_message_html(
                headline=rh,
                summary=rs,
                details=rd,
                source=source,
                link=link,
                rss_title=title,
                rss_summary=rss_summary,
                ai_hashtags=ai_hashtags
            )
            return msg_html, rh

    # Fallback: translate/compose
    headline, summ, details, ai_hashtags = openai_translate_compose(client, title, rss_summary, article_type)

    # Final scrub (already scrubbed in openai_translate_compose, but safe)
    headline = scrub_to_russian_only(headline)
    summ = scrub_to_russian_only(summ)
    details = scrub_to_russian_only(details)

    joined2 = f"{headline}\n{summ}\n{details}"
    if not is_russian_enough(joined2) or has_unwanted_latin(joined2):
        bad = unwanted_latin_words(joined2)
        raise ValueError(f"OpenAI response was not consistent Russian or contained too much Latin (bad={bad[:12]}).")

    if len(headline + summ + details) < 260:
        raise ValueError(f"Generated content too short ({len(headline + summ + details)} chars).")

    msg_html = build_message_html(
        headline=headline,
        summary=summ,
        details=details,
        source=source,
        link=link,
        rss_title=title,
        rss_summary=rss_summary,
        ai_hashtags=ai_hashtags
    )

    # Optional fact check if enabled/requested
    if not await verify_facts(client, title, rss_summary, f"{headline}\n{summ}\n{details}"):
        # One last attempt to rewrite if facts failed
        rh, rs, rd = openai_rewrite_russian_only(client, headline, summ, details)
        msg_html = build_message_html(rh, rs, rd, source, link, title, rss_summary, ai_hashtags)
        return msg_html, rh

    return msg_html, headline


# ============================================================
# BRIEF MODE (PAYWALL / LAST RESORT)
# ============================================================

def translate_brief_to_russian(client: OpenAI, title: str, teaser: str) -> tuple[str, str]:
    t = (title or "").strip()
    s = (teaser or "").strip()

    prompt = f"""
Переведи на русский язык ТОЛЬКО то, что дано. Не добавляй факты, имена, цифры или детали, которых нет в исходном тексте.
Пиши кириллицей.

Верни РОВНО в 2 строки:
ЗАГОЛОВОК: <перевод заголовка>
АНОНС: <перевод анонса> (если анонса нет — напиши "нет")

Заголовок:
{t}

Анонс:
{s}
""".strip()

    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Russian Cyrillic only. No new facts."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=260,
    )

    raw = (r.choices[0].message.content or "").strip()

    m1 = re.search(r"ЗАГОЛОВОК:\s*(.+)", raw)
    m2 = re.search(r"АНОНС:\s*(.+)", raw)

    ru_title = (m1.group(1).strip() if m1 else "").strip()
    ru_teaser = (m2.group(1).strip() if m2 else "").strip()

    if ru_teaser.lower() in {"нет", "нет.", "—", "-", "n/a"}:
        ru_teaser = ""

    # Deterministic scrub BEFORE validation (fixes Latin leaks)
    ru_title = scrub_to_russian_only(ru_title)
    ru_teaser = scrub_to_russian_only(ru_teaser)

    joined = f"{ru_title}\n{ru_teaser}".strip()
    if not joined or not is_russian_enough(joined) or has_unwanted_latin(joined):
        bad = unwanted_latin_words(joined)
        raise ValueError(f"brief translation not Russian-only (bad={bad[:12]})")

    return ru_title, ru_teaser


async def build_brief_message_html_ru(client: OpenAI, source: str, title: str, rss_summary_raw: str, link: str) -> tuple[str, str]:
    teaser_src = strip_html_text(rss_summary_raw or "").strip()

    ru_title, ru_teaser = translate_brief_to_russian(client, title, teaser_src)

    headline = "📰 " + (ru_title or "").strip()

    teaser = (ru_teaser or "").strip()
    if not teaser:
        teaser = "Краткий анонс недоступен в RSS (возможно, платный материал)."

    if len(teaser) > 350:
        teaser = teaser[:330].rstrip() + "…"

    details = "Полный текст может быть доступен по подписке."

    msg_html = build_message_html(
        headline=headline,
        summary=teaser,
        details=details,
        source=source,
        link=link,
        rss_title=title,
        rss_summary=strip_html_text(rss_summary_raw),
        ai_hashtags=None
    )
    return msg_html, headline


# ============================================================
# PAYWALL DETECTION (fixes AB/Expressen misclassification)
# ============================================================

IS_FREE_RE = re.compile(r'"isAccessibleForFree"\s*:\s*(true|false)', re.I)


async def detect_is_accessible_for_free(client: httpx.AsyncClient, article_url: str) -> bool | None:
    """
    Returns:
      True  -> explicitly free
      False -> explicitly paywalled
      None  -> unknown (no signal / fetch failed)
    """
    u = (article_url or "").strip()
    if not u:
        return None
    try:
        resp = await client.get(u, timeout=20.0)
        resp.raise_for_status()
        txt = resp.text or ""
        m = IS_FREE_RE.search(txt)
        if not m:
            return None
        return m.group(1).lower() == "true"
    except Exception as e:
        logger.debug(f"detect_is_accessible_for_free failed for {u}: {e}")
        return None


async def is_paywalled_like(client: httpx.AsyncClient, source: str, link: str, rss_summary_raw: str = "") -> bool:
    """
    DN: always paywalled (per requirement).
    Aftonbladet/Expressen: check schema.org isAccessibleForFree if possible.
    Fallback: thin RSS snippet => likely paywalled.
    """
    host = ""
    try:
        host = (urlparse(link or "").netloc or "").lower()
    except Exception:
        host = ""

    # DN always paywalled
    if host.endswith("dn.se"):
        return True

    summ_clean = strip_html_text(rss_summary_raw or "")
    is_thin = len(summ_clean) < 180

    if host.endswith("aftonbladet.se") or host.endswith("expressen.se"):
        free_flag = await detect_is_accessible_for_free(client, link)
        if free_flag is True:
            return False
        if free_flag is False:
            return True
        return is_thin

    return False


# ============================================================
# RSS RUN
# ============================================================

async def run_rss_once(app: Application, reason: str = "tick") -> None:
    bot = app.bot
    openai_client: OpenAI = app.bot_data["openai_client"]
    http_client: httpx.AsyncClient = app.bot_data["http_client"]
    conn: sqlite3.Connection = app.bot_data["db_conn"]

    now = time.time()
    last_run = float(app.bot_data.get("last_run_time", 0.0))
    if now - last_run < RUN_COOLDOWN_SECONDS:
        return
    app.bot_data["last_run_time"] = now

    next_ai_time = float(app.bot_data.get("next_ai_time", 0.0))
    if now < next_ai_time:
        return

    recent_titles = get_recent_titles(conn)
    recent_headlines = get_recent_headlines(conn)

    candidates = []
    for source, url in RSS_FEEDS:
        try:
            feed = await fetch_feed(http_client, url)
            FEED_HEALTH[source] = 0 # Reset on success
        except Exception as e:
            logger.error(f"[RSS] fetch failed {source}: {e}")
            save_failure(conn, source, "", "fetch_feed", str(e))
            
            # Health Alert logic
            FEED_HEALTH[source] = FEED_HEALTH.get(source, 0) + 1
            if FEED_HEALTH[source] == 5:
                try:
                    await bot.send_message(
                        chat_id=EDITOR_CHAT_ID,
                        text=f"⚠️ Health Alert: Feed '{source}' has failed 5 times in a row. Please check the source URL/layout."
                    )
                except Exception:
                    pass
            continue

        for entry in (feed.entries or [])[:PER_FEED_CAP]:
            title = (entry.get("title") or "").strip()
            summ = (entry.get("summary") or entry.get("description") or "").strip()
            link = (entry.get("link") or "").strip()
            item_id = extract_item_id(entry)
            if not item_id:
                continue
            if already_posted(conn, item_id):
                continue

            s = score_entry(title, summ)
            if s < MIN_SCORE:
                continue

            if any(is_similar(title, rt) for rt in recent_titles):
                logger.debug(f"[RSS] skipping similar (to DB): {title} ({source})")
                continue

            if not await verify_link(http_client, link):
                logger.info(f"[RSS] skipping dead link: {link} ({source})")
                continue

            candidates.append((s, source, title, summ, link, item_id))

    candidates.sort(key=lambda x: x[0], reverse=True)
    unique_candidates = []
    seen_titles_this_run = []
    for cand in candidates:
        s, source, title, summ, link, item_id = cand
        if any(is_similar(title, st) for st in seen_titles_this_run):
            print(f"[RSS] skipping similar (in run): {title} ({source})", flush=True)
            continue
        unique_candidates.append(cand)
        seen_titles_this_run.append(title)
    candidates = unique_candidates

    if not candidates:
        print(f"[RSS] candidates=0 (reason={reason})", flush=True)
        return

    produced = 0
    dropped = 0

    for s, source, title, summ, link, item_id in candidates[:max(1, MAX_PER_RUN)]:
        article_type = detect_article_type(source, title, link)

        # Image extraction
        image_url = ""
        try:
            image_url = await fetch_article_image(http_client, link)
        except Exception as ie:
            logger.error(f"[IMG] fetch failed {source}: {ie}")
            save_failure(conn, source, item_id, "fetch_image", str(ie))
            image_url = ""

        if image_url and not await is_usable_image(http_client, image_url):
            image_url = ""

        # Paywall logic
        summ_clean = strip_html_text(summ or "")
        is_thin = len(summ_clean) < 180
        paywalled = await is_paywalled_like(http_client, source, link, summ)

        # Context enrichment: Fetch full body if snippet is thin and not paywalled
        ai_context = summ
        if is_thin and not paywalled:
            full_body = await fetch_article_body(http_client, link)
            if len(full_body) > len(summ_clean):
                ai_context = full_body
                logger.info(f"[RSS] Enriched context for {source} (+{len(full_body)} chars)")

        host = (urlparse(link or "").netloc or "").lower()
        is_dn = host.endswith("dn.se")

        # DN always brief; others brief if paywalled AND thin
        use_brief = is_dn or (paywalled and is_thin)

        try:
            if use_brief:
                msg_html, generated_headline = await build_brief_message_html_ru(openai_client, source, title, ai_context, link)
            else:
                msg_html, generated_headline = await generate_post(openai_client, source, title, ai_context, link, article_type)
        except Exception as ex:
            msg = str(ex)

            if "rate limit" in msg.lower() or "429" in msg:
                wait = parse_rate_limit_wait_seconds(msg)
                app.bot_data["next_ai_time"] = time.time() + wait
                logger.warning(f"[RATE LIMIT] pausing AI for {wait}s")
                try:
                    await bot.send_message(
                        chat_id=EDITOR_CHAT_ID,
                        text=f"⏳ OpenAI rate limit. Pausing generation for ~{max(1, wait // 60)} minutes.",
                        disable_web_page_preview=True
                    )
                except Exception:
                    pass
                return

            # FINAL fallback: brief mode
            try:
                msg_html, generated_headline = await build_brief_message_html_ru(openai_client, source, title, ai_context, link)
                logger.info(f"[FALLBACK] brief mode used for {source} due to: {ex}")
            except Exception as ex2:
                dropped += 1
                logger.error(f"[DROP] {source}: {ex} / fallback_failed: {ex2}")
                save_failure(conn, source, item_id, "generate_post", f"{ex}\n{traceback.format_exc()}")
                continue

        if any(is_similar(generated_headline, rh) for rh in recent_headlines):
            logger.debug(f"[RSS] skipping similar generated headline (string match): {generated_headline} ({source})")
            dropped += 1
            continue

        if await is_semantically_similar(openai_client, generated_headline, recent_headlines):
            logger.info(f"[RSS] skipping semantically similar headline: {generated_headline} ({source})")
            dropped += 1
            continue

        if AUTO_POST:
            draft_id = save_draft(conn, msg_html, status="posted", image_url=image_url)
            try:
                await publish_to_channel(bot, PUBLIC_CHANNEL_ID, msg_html, image_url, client=http_client)
                logger.info(f"[AUTO] posted draft #{draft_id} for {source}")
                await bot.send_message(chat_id=EDITOR_CHAT_ID, text=f"🚀 Auto-posted #{draft_id} from {source}")
                # Transaction safety: mark as posted AFTER success
                mark_posted(conn, item_id, title=title, headline=generated_headline)
            except Exception as pe:
                dropped += 1
                err = f"autopost_failed: {pe}"
                logger.error(f"[DROP] autopost {source}: {err}")
                save_failure(conn, source, item_id, "autopost", err)
                conn.execute("UPDATE drafts SET status='failed', error=? WHERE id=?", (err[:2000], draft_id))
                conn.commit()
                continue
        else:
            draft_id = save_draft(conn, msg_html, status="pending", image_url=image_url)
            mark_posted(conn, item_id, title=title, headline=generated_headline) # For pending, still mark as "seen" item_id
            editor_payload = f"📝 Draft #{draft_id}\n\n{msg_html}\n\n/post {draft_id} | /skip {draft_id}"
            try:
                await send_html_safe(bot, EDITOR_CHAT_ID, editor_payload)
            except Exception as te:
                dropped += 1
                err = f"telegram_send_failed: {te}"
                logger.error(f"[DROP] send editor {source}: {err}")
                save_failure(conn, source, item_id, "send_editor", err)
                conn.execute("UPDATE drafts SET status='failed', error=? WHERE id=?", (err[:2000], draft_id))
                conn.commit()
                continue

        produced += 1
        await asyncio.sleep(1.0)

    print(f"[RSS] produced={produced} dropped={dropped} candidates={len(candidates)} (reason={reason})", flush=True)


# ============================================================
# JOB + COMMANDS
# ============================================================

async def rss_job_callback(context: ContextTypes.DEFAULT_TYPE) -> None:
    app: Application = context.application
    print("[JOB] rss_job_callback fired", flush=True)
    try:
        await run_rss_once(app, reason="tick")
    except Exception as e:
        print(f"[RSS] job error: {e}", flush=True)


async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    await update.message.reply_text("🔄 Running RSS now…")
    await run_rss_once(context.application, reason="manual")
    await update.message.reply_text("✅ RSS run complete.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    app = context.application
    conn: sqlite3.Connection | None = app.bot_data.get("db_conn")
    next_ai_time = float(app.bot_data.get("next_ai_time", 0.0))
    wait = max(0, int(next_ai_time - time.time()))

    pending = posted = failed = failures = 0
    if conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM drafts WHERE status='pending'")
        pending = int(c.fetchone()[0])
        c.execute("SELECT COUNT(*) FROM drafts WHERE status='posted'")
        posted = int(c.fetchone()[0])
        c.execute("SELECT COUNT(*) FROM drafts WHERE status='failed'")
        failed = int(c.fetchone()[0])
        c.execute("SELECT COUNT(*) FROM failures")
        failures = int(c.fetchone()[0])

    await update.message.reply_text(
        "🤖 Status\n"
        f"- job_tick: {JOB_TICK_SECONDS}s\n"
        f"- MAX_PER_RUN: {MAX_PER_RUN}\n"
        f"- PER_FEED_CAP: {PER_FEED_CAP}\n"
        f"- MIN_SCORE: {MIN_SCORE}\n"
        f"- model: {OPENAI_MODEL}\n"
        f"- max_tokens: {OPENAI_MAX_TOKENS}\n"
        f"- rate_limit_wait: {wait}s\n"
        f"- drafts: pending={pending} posted={posted} failed={failed}\n"
        f"- failures table: {failures}\n"
        f"- previews_disabled: {DISABLE_PREVIEWS}\n"
        f"- AUTO_POST: {AUTO_POST}\n"
    )


async def cmd_queue(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("SELECT id, created_at FROM drafts WHERE status='pending' ORDER BY id DESC LIMIT 10")
        rows = c.fetchall()
        if not rows:
            await update.message.reply_text("Queue is empty (no pending drafts).")
            return
        lines = ["🗂 Pending drafts:"]
        for did, created_at in rows:
            lines.append(f"- #{did} ({created_at}) -> /post {did} | /skip {did}")
        await update.message.reply_text("\n".join(lines))
    finally:
        conn.close()


async def cmd_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Usage: /post <draft_id>")
        return

    did = int(context.args[0])
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("SELECT text, status, COALESCE(image_url,'') FROM drafts WHERE id=?", (did,))
        row = c.fetchone()
        if not row:
            await update.message.reply_text(f"Draft #{did} not found.")
            return
        text, status, image_url = row
        if status != "pending":
            await update.message.reply_text(f"Draft #{did} is not pending (status: {status}).")
            return

        image_url = (image_url or "").strip()
        client = context.application.bot_data.get("http_client")
        await publish_to_channel(context.bot, PUBLIC_CHANNEL_ID, text, image_url, client=client)

        c.execute("UPDATE drafts SET status='posted' WHERE id=?", (did,))
        conn.commit()
        await update.message.reply_text(f"✅ Posted draft #{did}")
    except Exception as e:
        await update.message.reply_text(f"❌ Channel post failed: {e}")
    finally:
        conn.close()


async def cmd_skip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Usage: /skip <draft_id>")
        return

    did = int(context.args[0])
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("UPDATE drafts SET status='skipped' WHERE id=? AND status='pending'", (did,))
        conn.commit()
        await update.message.reply_text(f"🗑 Skipped draft #{did}")
    finally:
        conn.close()


# ============================================================
# INIT + MAIN
# ============================================================

async def post_init(app: Application) -> None:
    if app.job_queue is None:
        raise RuntimeError('Install JobQueue: pip install -U "python-telegram-bot[job-queue]"')

    app.bot_data["openai_client"] = OpenAI(api_key=OPENAI_API_KEY)
    app.bot_data["db_conn"] = init_db()
    backup_db() # Daily backup on startup
    app.bot_data["http_client"] = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        follow_redirects=True,
        headers={"User-Agent": feedparser.USER_AGENT}
    )
    app.bot_data["next_ai_time"] = 0.0
    app.bot_data["last_run_time"] = 0.0

    try:
        await app.bot.send_message(
            chat_id=EDITOR_CHAT_ID,
            text="✅ Bot started. Russian-only scrub enabled. Paywall detection uses isAccessibleForFree for AB/Expressen; DN forced brief.",
            disable_web_page_preview=True
        )
    except Exception as e:
        print(f"[WARN] startup notify failed: {e}", flush=True)

    app.job_queue.run_repeating(
        rss_job_callback,
        interval=max(60, JOB_TICK_SECONDS),
        first=8,
        name="rss_tick"
    )


def main():
    import signal
    acquire_lock_or_exit()

    def handle_exit(sig, frame):
        logger.info(f"Received exit signal {sig}. Closing bot...")
        release_lock()
        os._exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    logger.info(f"[BOOT] env ok. job_tick={JOB_TICK_SECONDS}s db={DB_PATH}")
    logger.info("[BOOT] polling telegram…")

    request = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )

    app = Application.builder().token(BOT_TOKEN).request(request).post_init(post_init).build()

    app.add_handler(CommandHandler("run", cmd_run))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("queue", cmd_queue))
    app.add_handler(CommandHandler("post", cmd_post))
    app.add_handler(CommandHandler("skip", cmd_skip))

    try:
        app.run_polling(drop_pending_updates=True)
    finally:
        try:
            conn = app.bot_data.get("db_conn")
            if conn:
                conn.close()
        except Exception:
            pass
        try:
            client = app.bot_data.get("http_client")
            if client:
                asyncio.run(client.aclose())
        except Exception:
            pass
        release_lock()


if __name__ == "__main__":
    main()
