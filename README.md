🇸🇪➡️🇷🇺 News-to-Telegram AI Media Bot

An automated Telegram bot that monitors Swedish news sources, summarizes selected articles in Russian, and publishes them to a Telegram channel with clean formatting, smart prioritization, and optional article photos.

Designed for quality over quantity: no invented facts, no spammy previews, no clickbait.

✨ Features

📡 Aggregates news from multiple Swedish sources (RSS + Google News)

🧠 Uses OpenAI to generate Russian-language summaries

📰 Produces structured posts:

headline

short summary

expandable details

🏷️ Automatically adds relevant Russian hashtags

🖼️ Attaches article photos when a real photo exists

logos/placeholders are ignored

text-only posts when no usable photo is found

🔕 Link previews disabled (no Swedish snippets)

🧑‍💻 Editor review workflow (approve / skip before publishing)

🗄️ SQLite persistence (prevents duplicate posts)

⚙️ Fully configurable via environment variables

📢 How it works (high level)

Bot periodically checks configured RSS feeds

Articles are scored by relevance

Top-scoring items are processed (rate-limited)

AI generates Russian content using only RSS data

Drafts are sent to an editor chat

Approved drafts are posted to the public channel

🛠️ Requirements

Python 3.10+

Telegram Bot Token

OpenAI API Key

python-telegram-bot

feedparser, requests, python-dotenv

📦 Installation
git clone https://github.com/stoilsteve-hub/News-to-Telegram_AI_Media_Bot.git
cd News-to-Telegram_AI_Media_Bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

🔐 Environment setup

Create a local file config/.env (this file is ignored by Git):

BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_key
EDITOR_CHAT_ID=123456789
PUBLIC_CHANNEL_ID=-100123456789
TELEGRAM_HANDLE=@YourChannel


A safe template is provided in:

config/.env.example

▶️ Run the bot
python bot.py


The bot will:

start polling Telegram

schedule RSS checks automatically

send drafts to the editor chat

🧑‍✈️ Editor commands

In the editor chat:

/post <id> → publish draft

/skip <id> → discard draft

/queue → list pending drafts

/status → bot status

/run → force RSS run

🖼️ Photo logic (important)

If a real article photo is detected → it is attached

If the image is a logo / placeholder / too small → ignored

If no usable photo exists → text-only post

This ensures clean, professional posts.

🚫 What the bot will NOT do

❌ Invent facts

❌ Translate full articles

❌ Scrape paywalled content

❌ Show Swedish link previews

❌ Auto-publish without review

🧩 Customization

You can easily adjust:

RSS sources

keyword scoring

posting limits

hashtags

OpenAI model & temperature

job frequency

All major settings are environment-based.

⚠️ Security notes

Never commit real API keys

.env files are git-ignored

Rotate keys immediately if exposed

GitHub push protection is enabled

📄 License

MIT License — use, modify, and deploy freely.

🙋‍♂️ Author

Built and maintained by @stoilsteve
