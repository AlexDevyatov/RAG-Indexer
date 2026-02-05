# 🚀 Инструкция по развёртыванию RAG Indexer

## Требования к серверу

- **ОС:** Ubuntu 20.04+ / Debian 11+ / CentOS 8+ / macOS
- **CPU:** 2+ ядра (рекомендуется 4+)
- **RAM:** 8GB минимум (рекомендуется 16GB для LLM)
- **Диск:** 20GB свободного места
- **Python:** 3.9+

---

## Быстрый старт (одна команда)

```bash
# Клонируем/копируем проект и запускаем
chmod +x deploy.sh && ./deploy.sh --install-all
```

---

## Пошаговая установка

### 1. Подготовка сервера

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git curl wget
```

**CentOS/RHEL:**
```bash
sudo dnf update -y
sudo dnf install -y python3 python3-pip git curl wget
```

**macOS:**
```bash
xcode-select --install
brew install python@3.11
```

### 2. Копирование проекта на сервер

**Вариант A: Git clone**
```bash
git clone <repository-url> /opt/rag-indexer
cd /opt/rag-indexer
```

**Вариант B: SCP/SFTP**
```bash
# С локальной машины:
scp -r ./IndexingOllama user@server:/opt/rag-indexer

# На сервере:
cd /opt/rag-indexer
```

### 3. Запуск скрипта развёртывания

```bash
chmod +x deploy.sh
./deploy.sh --install-all
```

**Опции скрипта:**
| Флаг | Описание |
|------|----------|
| `--install-all` | Установить всё (Ollama, зависимости, модели) |
| `--install-ollama` | Установить только Ollama |
| `--install-deps` | Установить только Python зависимости |
| `--port PORT` | Указать порт (по умолчанию 8080) |
| `--no-models` | Не скачивать модели Ollama |

### 4. Запуск приложения

**Режим разработки:**
```bash
./start.sh
```

**Production (systemd):**
```bash
sudo systemctl start rag-indexer
sudo systemctl enable rag-indexer  # автозапуск
```

---

## Конфигурация

### Переменные окружения (.env)

```bash
# Опционально: DeepSeek API (если используется)
DEEPSEEK_API_KEY=sk-your-key

# Настройки сервера
RAG_PORT=8080
RAG_HOST=0.0.0.0
```

### Настройка Nginx (рекомендуется)

```nginx
# /etc/nginx/sites-available/rag-indexer
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/rag-indexer /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### HTTPS с Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## Управление сервисами

### Systemd команды

```bash
# RAG Indexer
sudo systemctl start rag-indexer     # Запуск
sudo systemctl stop rag-indexer      # Остановка
sudo systemctl restart rag-indexer   # Перезапуск
sudo systemctl status rag-indexer    # Статус

# Ollama
sudo systemctl start ollama
sudo systemctl status ollama

# Логи
sudo journalctl -u rag-indexer -f    # В реальном времени
sudo journalctl -u rag-indexer -n 100  # Последние 100 строк
```

### Ручное управление

```bash
./start.sh    # Запуск
./stop.sh     # Остановка
```

---

## Обновление

```bash
# 1. Остановить сервис
sudo systemctl stop rag-indexer

# 2. Обновить код
git pull  # или scp новых файлов

# 3. Обновить зависимости
source venv/bin/activate
pip install -r requirements.txt

# 4. Запустить
sudo systemctl start rag-indexer
```

---

## Решение проблем

### Ollama не запускается

```bash
# Проверить статус
systemctl status ollama

# Запустить вручную для отладки
ollama serve

# Проверить порт
curl http://localhost:11434/api/tags
```

### Нехватка памяти для LLM

```bash
# Использовать меньшую модель
ollama pull llama3.2:1b  # 1B параметров вместо 3B

# Или настроить swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Порт занят

```bash
# Найти процесс на порту
sudo lsof -i :8080

# Убить процесс
sudo kill -9 <PID>

# Или использовать другой порт
./deploy.sh --port 8081
```

### Ошибки Python

```bash
# Пересоздать виртуальное окружение
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Структура файлов после деплоя

```
/opt/rag-indexer/
├── app.py              # Веб-сервер
├── deploy.sh           # Скрипт развёртывания
├── start.sh            # Скрипт запуска (создаётся)
├── stop.sh             # Скрипт остановки (создаётся)
├── requirements.txt
├── .env                # Конфигурация
├── venv/               # Виртуальное окружение
├── uploads/            # Загруженные файлы
├── index_data/         # FAISS индекс
│   ├── faiss.index
│   └── metadata.json
├── templates/
│   └── index.html
└── utils/
    ├── parser.py
    ├── chunker.py
    └── embedder.py
```

---

## Мониторинг

### Проверка здоровья

```bash
# API статус
curl http://localhost:8080/api/stats

# Ollama статус
curl http://localhost:11434/api/tags
```

### Простой healthcheck скрипт

```bash
#!/bin/bash
# healthcheck.sh
if curl -s http://localhost:8080/api/stats > /dev/null; then
    echo "✅ RAG Indexer OK"
else
    echo "❌ RAG Indexer DOWN"
    sudo systemctl restart rag-indexer
fi
```

### Добавить в cron

```bash
crontab -e
# Добавить:
*/5 * * * * /opt/rag-indexer/healthcheck.sh >> /var/log/rag-healthcheck.log 2>&1
```

---

## Контакты и поддержка

При возникновении проблем проверьте логи:
```bash
sudo journalctl -u rag-indexer -n 50
```
