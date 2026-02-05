#!/bin/bash
# =============================================================================
# СКРИПТ РАЗВЁРТЫВАНИЯ RAG INDEXER НА СЕРВЕРЕ
# =============================================================================
# Использование: ./deploy.sh [OPTIONS]
#
# Опции:
#   --install-all       Установить всё (Ollama + зависимости + модели)
#   --install-ollama    Установить только Ollama
#   --install-deps      Установить только Python зависимости
#   --port PORT         Указать порт (по умолчанию 8000)
#   --no-models         Не скачивать модели Ollama
#   --help              Показать справку
# =============================================================================

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Параметры по умолчанию
PORT=8000
INSTALL_OLLAMA=false
INSTALL_DEPS=false
INSTALL_MODELS=true
APP_DIR=$(cd "$(dirname "$0")" && pwd)
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="rag-indexer"

# Функция помощи
show_help() {
    echo -e "${CYAN}"
    echo "============================================================"
    echo "RAG INDEXER - СКРИПТ РАЗВЁРТЫВАНИЯ"
    echo "============================================================"
    echo -e "${NC}"
    echo "Использование: ./deploy.sh [OPTIONS]"
    echo ""
    echo "Опции:"
    echo "  --install-all       Установить всё (рекомендуется)"
    echo "  --install-ollama    Установить только Ollama"
    echo "  --install-deps      Установить только Python зависимости"
    echo "  --port PORT         Указать порт (по умолчанию 8000)"
    echo "  --no-models         Не скачивать модели Ollama"
    echo "  --help              Показать эту справку"
    echo ""
    echo "Примеры:"
    echo "  ./deploy.sh --install-all              # Полная установка"
    echo "  ./deploy.sh --install-deps --port 3000 # Только зависимости"
    echo "  ./deploy.sh --install-ollama           # Только Ollama"
    exit 0
}

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        --install-all)
            INSTALL_OLLAMA=true
            INSTALL_DEPS=true
            shift
            ;;
        --install-ollama)
            INSTALL_OLLAMA=true
            shift
            ;;
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --no-models)
            INSTALL_MODELS=false
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}❌ Неизвестный параметр: $1${NC}"
            echo "Используйте --help для справки"
            exit 1
            ;;
    esac
done

# Если флаги не указаны - устанавливаем всё
if [ "$INSTALL_OLLAMA" = false ] && [ "$INSTALL_DEPS" = false ]; then
    INSTALL_OLLAMA=true
    INSTALL_DEPS=true
fi

echo -e "${BLUE}"
echo "============================================================"
echo "🚀 РАЗВЁРТЫВАНИЕ RAG INDEXER"
echo "============================================================"
echo -e "${NC}"
echo -e "📁 Директория: ${GREEN}$APP_DIR${NC}"
echo -e "🔌 Порт: ${GREEN}$PORT${NC}"
echo ""

# =============================================================================
# ПРОВЕРКА И УСТАНОВКА СИСТЕМНЫХ ЗАВИСИМОСТЕЙ
# =============================================================================

echo -e "${YELLOW}📦 Проверка системных зависимостей...${NC}"

# Определение ОС
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    # Определяем дистрибутив
    if [ -f /etc/debian_version ]; then
        DISTRO="debian"
    elif [ -f /etc/redhat-release ]; then
        DISTRO="redhat"
    else
        DISTRO="unknown"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    DISTRO="macos"
else
    OS="unknown"
    DISTRO="unknown"
fi

echo -e "  ОС: ${GREEN}$OS ($DISTRO)${NC}"

# Установка системных пакетов
install_system_deps() {
    echo -e "\n${YELLOW}📦 Установка системных пакетов...${NC}"
    
    if [ "$DISTRO" = "debian" ]; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3 python3-pip python3-venv curl wget git
    elif [ "$DISTRO" = "redhat" ]; then
        sudo dnf install -y -q python3 python3-pip curl wget git
    elif [ "$DISTRO" = "macos" ]; then
        if ! command -v brew &> /dev/null; then
            echo -e "${YELLOW}Установка Homebrew...${NC}"
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install python@3.11 curl wget git 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✅ Системные пакеты установлены${NC}"
}

# Проверка Python
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 не найден${NC}"
        install_system_deps
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "  Python: ${GREEN}$PYTHON_VERSION${NC}"
    
    # Проверка версии >= 3.9
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
        echo -e "${RED}❌ Требуется Python 3.9+${NC}"
        exit 1
    fi
}

check_python

# =============================================================================
# УСТАНОВКА OLLAMA
# =============================================================================

if [ "$INSTALL_OLLAMA" = true ]; then
    echo -e "\n${YELLOW}📦 Установка Ollama...${NC}"
    
    if command -v ollama &> /dev/null; then
        OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
        echo -e "  Ollama уже установлена: ${GREEN}$OLLAMA_VERSION${NC}"
    else
        if [ "$OS" = "linux" ]; then
            echo "  Загрузка и установка Ollama для Linux..."
            curl -fsSL https://ollama.com/install.sh | sh
        elif [ "$OS" = "macos" ]; then
            echo "  Установка Ollama через Homebrew..."
            brew install --cask ollama 2>/dev/null || brew upgrade --cask ollama 2>/dev/null || true
        else
            echo -e "${RED}❌ Неподдерживаемая ОС для автоустановки Ollama${NC}"
            echo "  Установите вручную: https://ollama.com"
            exit 1
        fi
        echo -e "${GREEN}✅ Ollama установлена${NC}"
    fi
fi

# =============================================================================
# СОЗДАНИЕ ВИРТУАЛЬНОГО ОКРУЖЕНИЯ И УСТАНОВКА ЗАВИСИМОСТЕЙ
# =============================================================================

if [ "$INSTALL_DEPS" = true ]; then
    echo -e "\n${YELLOW}📦 Создание виртуального окружения...${NC}"
    
    if [ -d "$VENV_DIR" ]; then
        echo -e "  Удаление старого venv..."
        rm -rf "$VENV_DIR"
    fi
    
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✅ Виртуальное окружение создано: $VENV_DIR${NC}"
    
    echo -e "\n${YELLOW}📦 Установка Python зависимостей...${NC}"
    
    # Активируем venv
    source "$VENV_DIR/bin/activate"
    
    # Обновляем pip
    pip install --upgrade pip -q
    
    # Устанавливаем зависимости
    if [ -f "$APP_DIR/requirements.txt" ]; then
        pip install -r "$APP_DIR/requirements.txt" -q
        echo -e "${GREEN}✅ Зависимости установлены из requirements.txt${NC}"
    else
        # Устанавливаем базовые зависимости
        pip install flask flask-cors gunicorn faiss-cpu PyMuPDF python-docx requests python-dotenv tqdm ollama numpy -q
        echo -e "${GREEN}✅ Базовые зависимости установлены${NC}"
    fi
    
    # Показываем установленные пакеты
    echo -e "\n${CYAN}📋 Установленные пакеты:${NC}"
    pip list --format=freeze | grep -E "^(flask|faiss|PyMuPDF|python-docx|ollama|tqdm|gunicorn)" | while read line; do
        echo -e "  ${GREEN}✓${NC} $line"
    done
fi

# =============================================================================
# ЗАПУСК OLLAMA И ЗАГРУЗКА МОДЕЛЕЙ
# =============================================================================

if [ "$INSTALL_MODELS" = true ] && command -v ollama &> /dev/null; then
    echo -e "\n${YELLOW}📦 Настройка Ollama и загрузка моделей...${NC}"
    
    # Проверяем, запущена ли Ollama
    if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo "  Запуск Ollama сервера..."
        
        if [ "$OS" = "macos" ]; then
            # На macOS запускаем приложение
            open -a Ollama 2>/dev/null || ollama serve &
        else
            # На Linux запускаем в фоне
            ollama serve &
        fi
        
        # Ждём запуска
        echo -n "  Ожидание запуска"
        for i in {1..30}; do
            if curl -s http://localhost:11434/api/tags &> /dev/null; then
                echo ""
                break
            fi
            echo -n "."
            sleep 1
        done
    fi
    
    # Проверяем ещё раз
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo -e "${GREEN}✅ Ollama запущена${NC}"
        
        # Загружаем модели
        echo -e "\n${YELLOW}📥 Загрузка моделей (это может занять время)...${NC}"
        
        echo "  Загрузка nomic-embed-text (для эмбеддингов)..."
        ollama pull nomic-embed-text 2>/dev/null && echo -e "  ${GREEN}✓${NC} nomic-embed-text" || echo -e "  ${RED}✗${NC} Ошибка загрузки nomic-embed-text"
        
        echo "  Загрузка llama3.2 (для генерации ответов)..."
        ollama pull llama3.2 2>/dev/null && echo -e "  ${GREEN}✓${NC} llama3.2" || echo -e "  ${RED}✗${NC} Ошибка загрузки llama3.2"
        
        echo -e "${GREEN}✅ Модели загружены${NC}"
    else
        echo -e "${RED}⚠️  Не удалось запустить Ollama. Запустите вручную: ollama serve${NC}"
    fi
fi

# =============================================================================
# СОЗДАНИЕ SYSTEMD СЕРВИСОВ (только для Linux)
# =============================================================================

if [ "$OS" = "linux" ]; then
    echo -e "\n${YELLOW}⚙️  Настройка systemd сервисов...${NC}"
    
    # Сервис для RAG Indexer
    RAG_SERVICE="/etc/systemd/system/${SERVICE_NAME}.service"
    
    sudo tee "$RAG_SERVICE" > /dev/null << EOF
[Unit]
Description=RAG Indexer Web Service
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$APP_DIR
Environment=PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin
Environment=PYTHONUNBUFFERED=1
ExecStart=$VENV_DIR/bin/python app.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    echo -e "  ${GREEN}✓${NC} Создан сервис: $SERVICE_NAME"
    
    # Сервис для Ollama (если не существует)
    OLLAMA_SERVICE="/etc/systemd/system/ollama.service"
    if [ ! -f "$OLLAMA_SERVICE" ]; then
        OLLAMA_PATH=$(which ollama 2>/dev/null || echo "/usr/local/bin/ollama")
        
        sudo tee "$OLLAMA_SERVICE" > /dev/null << EOF
[Unit]
Description=Ollama LLM Service
After=network.target

[Service]
Type=simple
User=$USER
Group=$USER
ExecStart=$OLLAMA_PATH serve
Restart=always
RestartSec=10
Environment=OLLAMA_HOST=0.0.0.0

[Install]
WantedBy=multi-user.target
EOF
        echo -e "  ${GREEN}✓${NC} Создан сервис: ollama"
    fi
    
    # Перезагружаем systemd
    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE_NAME" 2>/dev/null || true
    sudo systemctl enable ollama 2>/dev/null || true
    
    echo -e "${GREEN}✅ Systemd сервисы настроены${NC}"
fi

# =============================================================================
# СОЗДАНИЕ СКРИПТОВ ЗАПУСКА
# =============================================================================

echo -e "\n${YELLOW}📝 Создание скриптов управления...${NC}"

# Скрипт запуска
cat > "$APP_DIR/start.sh" << EOF
#!/bin/bash
# Скрипт запуска RAG Indexer
cd "$APP_DIR"
source "$VENV_DIR/bin/activate"

# Проверяем Ollama
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "🚀 Запуск Ollama..."
    if [[ "\$OSTYPE" == "darwin"* ]]; then
        open -a Ollama 2>/dev/null || ollama serve &
    else
        ollama serve &
    fi
    sleep 5
fi

echo "🚀 Запуск RAG Indexer на порту $PORT..."
echo "📍 URL: http://localhost:$PORT"
python app.py
EOF

# Скрипт остановки
cat > "$APP_DIR/stop.sh" << EOF
#!/bin/bash
# Скрипт остановки RAG Indexer
echo "🛑 Остановка RAG Indexer..."
pkill -f "python.*app.py" 2>/dev/null || true
echo "✅ Остановлено"
EOF

# Скрипт перезапуска
cat > "$APP_DIR/restart.sh" << EOF
#!/bin/bash
# Скрипт перезапуска RAG Indexer
"$APP_DIR/stop.sh"
sleep 2
"$APP_DIR/start.sh"
EOF

# Скрипт проверки здоровья
cat > "$APP_DIR/healthcheck.sh" << EOF
#!/bin/bash
# Проверка здоровья RAG Indexer
RAG_OK=false
OLLAMA_OK=false

if curl -s http://localhost:$PORT/api/stats > /dev/null 2>&1; then
    RAG_OK=true
fi

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    OLLAMA_OK=true
fi

echo "RAG Indexer: \$([ "\$RAG_OK" = true ] && echo '✅ OK' || echo '❌ DOWN')"
echo "Ollama:      \$([ "\$OLLAMA_OK" = true ] && echo '✅ OK' || echo '❌ DOWN')"

if [ "\$RAG_OK" = false ] || [ "\$OLLAMA_OK" = false ]; then
    exit 1
fi
EOF

chmod +x "$APP_DIR/start.sh" "$APP_DIR/stop.sh" "$APP_DIR/restart.sh" "$APP_DIR/healthcheck.sh"

echo -e "  ${GREEN}✓${NC} start.sh"
echo -e "  ${GREEN}✓${NC} stop.sh"
echo -e "  ${GREEN}✓${NC} restart.sh"
echo -e "  ${GREEN}✓${NC} healthcheck.sh"

# =============================================================================
# СОЗДАНИЕ ДИРЕКТОРИЙ
# =============================================================================

echo -e "\n${YELLOW}📁 Создание рабочих директорий...${NC}"

mkdir -p "$APP_DIR/uploads"
mkdir -p "$APP_DIR/index_data"
mkdir -p "$APP_DIR/templates"
mkdir -p "$APP_DIR/static"

echo -e "  ${GREEN}✓${NC} uploads/"
echo -e "  ${GREEN}✓${NC} index_data/"
echo -e "  ${GREEN}✓${NC} templates/"
echo -e "  ${GREEN}✓${NC} static/"

# =============================================================================
# ИТОГОВЫЙ ВЫВОД
# =============================================================================

echo -e "\n${BLUE}"
echo "============================================================"
echo "✅ РАЗВЁРТЫВАНИЕ ЗАВЕРШЕНО"
echo "============================================================"
echo -e "${NC}"

echo -e "${GREEN}📁 Директория:${NC} $APP_DIR"
echo -e "${GREEN}🐍 Python venv:${NC} $VENV_DIR"
echo -e "${GREEN}🌐 URL:${NC} http://localhost:$PORT"

echo -e "\n${CYAN}📋 КОМАНДЫ УПРАВЛЕНИЯ:${NC}"
echo -e "  ${GREEN}./start.sh${NC}       - Запустить приложение"
echo -e "  ${GREEN}./stop.sh${NC}        - Остановить"
echo -e "  ${GREEN}./restart.sh${NC}     - Перезапустить"
echo -e "  ${GREEN}./healthcheck.sh${NC} - Проверить статус"

if [ "$OS" = "linux" ]; then
    echo -e "\n${CYAN}📋 SYSTEMD КОМАНДЫ:${NC}"
    echo -e "  ${GREEN}sudo systemctl start $SERVICE_NAME${NC}"
    echo -e "  ${GREEN}sudo systemctl stop $SERVICE_NAME${NC}"
    echo -e "  ${GREEN}sudo systemctl status $SERVICE_NAME${NC}"
    echo -e "  ${GREEN}sudo journalctl -u $SERVICE_NAME -f${NC}"
fi

echo -e "\n${YELLOW}🚀 БЫСТРЫЙ СТАРТ:${NC}"
echo -e "  ${GREEN}./start.sh${NC}"
echo ""

# Предложение запустить
read -p "Запустить приложение сейчас? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    "$APP_DIR/start.sh"
fi
