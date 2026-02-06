#!/bin/bash
#
# RAG Indexer - Скрипт обновления и деплоя
# Использование: ./deploy.sh [команда]
#
# Команды:
#   update    - Обновить код из git и перезапустить сервер
#   start     - Запустить сервер
#   stop      - Остановить сервер
#   restart   - Перезапустить сервер
#   status    - Показать статус сервера
#   logs      - Показать логи в реальном времени
#   install   - Установить зависимости
#

set -e

# Конфигурация
APP_NAME="RAG Indexer"
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$APP_DIR/.server.pid"
LOG_FILE="$APP_DIR/logs/app.log"
PORT=8001
PYTHON="${PYTHON:-python3}"
VENV_DIR="$APP_DIR/venv"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  $APP_NAME${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Активация виртуального окружения
activate_venv() {
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
        print_info "Виртуальное окружение активировано"
    fi
}

# Проверка, запущен ли сервер
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        fi
    fi
    # Проверка по порту
    if lsof -i :$PORT > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Получение PID сервера
get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    else
        lsof -t -i :$PORT 2>/dev/null || echo ""
    fi
}

# Команда: install
cmd_install() {
    print_header
    echo "📦 Установка зависимостей..."
    
    cd "$APP_DIR"
    
    # Создание виртуального окружения
    if [ ! -d "$VENV_DIR" ]; then
        print_info "Создание виртуального окружения..."
        $PYTHON -m venv "$VENV_DIR"
    fi
    
    activate_venv
    
    # Установка зависимостей
    if [ -f "requirements.txt" ]; then
        print_info "Установка из requirements.txt..."
        pip install -r requirements.txt
        print_success "Зависимости установлены"
    else
        print_error "requirements.txt не найден"
        exit 1
    fi
    
    # Создание необходимых директорий
    mkdir -p logs uploads index_data templates static
    
    # Проверка .env
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_warning ".env не найден, копирую из .env.example"
            cp .env.example .env
            print_warning "Не забудьте заполнить DEEPSEEK_API_KEY в .env"
        fi
    fi
    
    print_success "Установка завершена"
}

# Команда: start
cmd_start() {
    print_header
    
    if is_running; then
        print_warning "Сервер уже запущен (PID: $(get_pid))"
        return 0
    fi
    
    cd "$APP_DIR"
    activate_venv
    
    print_info "Запуск сервера на порту $PORT..."
    
    # Запуск в фоне
    nohup $PYTHON app.py > /dev/null 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    
    # Ждём запуска
    sleep 2
    
    if is_running; then
        print_success "Сервер запущен (PID: $PID)"
        print_info "URL: http://localhost:$PORT"
        print_info "Логи: $LOG_FILE"
    else
        print_error "Не удалось запустить сервер"
        rm -f "$PID_FILE"
        exit 1
    fi
}

# Команда: stop
cmd_stop() {
    print_header
    
    if ! is_running; then
        print_warning "Сервер не запущен"
        rm -f "$PID_FILE"
        return 0
    fi
    
    PID=$(get_pid)
    print_info "Остановка сервера (PID: $PID)..."
    
    kill "$PID" 2>/dev/null || true
    
    # Ждём завершения
    for i in {1..10}; do
        if ! is_running; then
            break
        fi
        sleep 1
    done
    
    # Force kill если ещё работает
    if is_running; then
        print_warning "Принудительная остановка..."
        kill -9 "$PID" 2>/dev/null || true
    fi
    
    rm -f "$PID_FILE"
    print_success "Сервер остановлен"
}

# Команда: restart
cmd_restart() {
    cmd_stop
    sleep 1
    cmd_start
}

# Команда: status
cmd_status() {
    print_header
    
    if is_running; then
        PID=$(get_pid)
        print_success "Сервер запущен (PID: $PID)"
        print_info "URL: http://localhost:$PORT"
        
        # Проверка доступности
        if curl -s "http://localhost:$PORT/api/stats" > /dev/null 2>&1; then
            STATS=$(curl -s "http://localhost:$PORT/api/stats")
            VECTORS=$(echo "$STATS" | grep -o '"total_vectors":[0-9]*' | cut -d: -f2)
            DOCS=$(echo "$STATS" | grep -o '"total_documents":[0-9]*' | cut -d: -f2)
            CHUNKS=$(echo "$STATS" | grep -o '"total_chunks":[0-9]*' | cut -d: -f2)
            print_info "Индекс: $VECTORS векторов, $DOCS документов, $CHUNKS чанков"
        fi
    else
        print_error "Сервер не запущен"
    fi
}

# Команда: logs
cmd_logs() {
    print_header
    
    if [ -f "$LOG_FILE" ]; then
        print_info "Логи: $LOG_FILE (Ctrl+C для выхода)"
        echo ""
        tail -f "$LOG_FILE"
    else
        print_warning "Файл логов не найден: $LOG_FILE"
    fi
}

# Команда: update
cmd_update() {
    print_header
    echo "🔄 Обновление $APP_NAME..."
    
    cd "$APP_DIR"
    
    # Проверка git
    if [ ! -d ".git" ]; then
        print_error "Это не git репозиторий"
        exit 1
    fi
    
    # Сохраняем текущий статус
    WAS_RUNNING=false
    if is_running; then
        WAS_RUNNING=true
        print_info "Остановка сервера для обновления..."
        cmd_stop
    fi
    
    # Git pull
    print_info "Получение обновлений из git..."
    git fetch origin
    
    # Показываем изменения
    CHANGES=$(git log HEAD..origin/main --oneline 2>/dev/null || echo "")
    if [ -z "$CHANGES" ]; then
        print_success "Уже актуальная версия"
    else
        print_info "Новые коммиты:"
        echo "$CHANGES"
        echo ""
        
        git pull origin main
        print_success "Код обновлён"
    fi
    
    # Обновление зависимостей если requirements.txt изменился
    if git diff HEAD@{1} --name-only 2>/dev/null | grep -q "requirements.txt"; then
        print_info "requirements.txt изменился, обновление зависимостей..."
        activate_venv
        pip install -r requirements.txt
    fi
    
    # Перезапуск если был запущен
    if [ "$WAS_RUNNING" = true ]; then
        print_info "Перезапуск сервера..."
        cmd_start
    fi
    
    print_success "Обновление завершено"
}

# Команда: help
cmd_help() {
    echo "Использование: $0 [команда]"
    echo ""
    echo "Команды:"
    echo "  install   - Установить зависимости"
    echo "  start     - Запустить сервер"
    echo "  stop      - Остановить сервер"
    echo "  restart   - Перезапустить сервер"
    echo "  status    - Показать статус сервера"
    echo "  logs      - Показать логи в реальном времени"
    echo "  update    - Обновить код из git и перезапустить"
    echo "  help      - Показать эту справку"
    echo ""
    echo "Примеры:"
    echo "  $0 install    # Первоначальная установка"
    echo "  $0 start      # Запустить сервер"
    echo "  $0 update     # Обновить и перезапустить"
}

# Главная логика
case "${1:-help}" in
    install)
        cmd_install
        ;;
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart
        ;;
    status)
        cmd_status
        ;;
    logs)
        cmd_logs
        ;;
    update)
        cmd_update
        ;;
    help|--help|-h)
        cmd_help
        ;;
    *)
        print_error "Неизвестная команда: $1"
        echo ""
        cmd_help
        exit 1
        ;;
esac
