# Grok-Telegram-Group-Chat-Bot
A responsive and simple Telegram group chat bot built with Python.

## Overview
The bot listens for user messages in group chats and responds to mentions.  
It’s lightweight, easy to run, and designed to add more fun to your group chats.

## Getting Started

### Prerequisites
- Telegram Bot Token (from [@BotFather](https://telegram.me/BotFather))  
- Grok API Token (from [xAI](https://x.ai/api))

### Installation
```bash
git clone https://github.com/ABS0LUTE888/Grok-Telegram-Group-Chat-Bot.git
cd Grok-Telegram-Group-Chat-Bot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Telegram bot token and xAI API Token
```
### Running the Bot
```bash
python src/main.py
```

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for details.