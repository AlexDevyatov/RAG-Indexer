"""
Модуль парсинга документов различных форматов.
Поддерживает: PDF, DOCX, TXT, MD, код (py, js, ts, java, cpp, c, h, go, rs, rb, php, sql)
"""

import os
from pathlib import Path
from typing import List, Tuple


# Расширения файлов с кодом
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
    '.go', '.rs', '.rb', '.php', '.sql', '.sh', '.bash', '.yaml', '.yml',
    '.json', '.xml', '.html', '.css', '.scss', '.less'
}

# Текстовые расширения
TEXT_EXTENSIONS = {'.txt', '.md', '.markdown', '.rst', '.log'}


def parse_pdf(file_path: str) -> str:
    """
    Парсинг PDF файла с использованием PyMuPDF (fitz).
    
    Args:
        file_path: путь к PDF файлу
        
    Returns:
        извлечённый текст из всех страниц
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("📦 Установите PyMuPDF: pip install PyMuPDF")
    
    text_parts = []
    
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(page_text)
        doc.close()
    except Exception as e:
        raise RuntimeError(f"❌ Ошибка при парсинге PDF '{file_path}': {e}")
    
    return "\n\n".join(text_parts)


def parse_docx(file_path: str) -> str:
    """
    Парсинг DOCX файла с использованием python-docx.
    
    Args:
        file_path: путь к DOCX файлу
        
    Returns:
        извлечённый текст из всех параграфов
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("📦 Установите python-docx: pip install python-docx")
    
    try:
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        raise RuntimeError(f"❌ Ошибка при парсинге DOCX '{file_path}': {e}")


def parse_text(file_path: str) -> str:
    """
    Парсинг текстового файла (TXT, MD, код).
    
    Args:
        file_path: путь к текстовому файлу
        
    Returns:
        содержимое файла
    """
    encodings = ['utf-8', 'cp1251', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise RuntimeError(f"❌ Ошибка при чтении файла '{file_path}': {e}")
    
    raise RuntimeError(f"❌ Не удалось определить кодировку файла '{file_path}'")


def get_file_type(file_path: str) -> str:
    """
    Определение типа файла по расширению.
    
    Args:
        file_path: путь к файлу
        
    Returns:
        тип файла: 'pdf', 'docx', 'text', 'code' или 'unknown'
    """
    ext = Path(file_path).suffix.lower()
    
    if ext == '.pdf':
        return 'pdf'
    elif ext == '.docx':
        return 'docx'
    elif ext in TEXT_EXTENSIONS:
        return 'text'
    elif ext in CODE_EXTENSIONS:
        return 'code'
    else:
        return 'unknown'


def parse_file(file_path: str) -> Tuple[str, str]:
    """
    Универсальный парсер файла.
    
    Args:
        file_path: путь к файлу
        
    Returns:
        кортеж (текст, тип_файла)
    """
    file_type = get_file_type(file_path)
    
    if file_type == 'pdf':
        text = parse_pdf(file_path)
    elif file_type == 'docx':
        text = parse_docx(file_path)
    elif file_type in ('text', 'code'):
        text = parse_text(file_path)
    else:
        raise ValueError(f"⚠️ Неподдерживаемый формат файла: {file_path}")
    
    return text, file_type


def parse_directory(dir_path: str, recursive: bool = True) -> List[Tuple[str, str, str]]:
    """
    Парсинг всех поддерживаемых файлов в директории.
    
    Args:
        dir_path: путь к директории
        recursive: рекурсивный обход поддиректорий
        
    Returns:
        список кортежей (путь_к_файлу, текст, тип_файла)
    """
    results = []
    dir_path = Path(dir_path)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"❌ Директория не найдена: {dir_path}")
    
    # Собираем все файлы
    if recursive:
        files = list(dir_path.rglob('*'))
    else:
        files = list(dir_path.glob('*'))
    
    # Фильтруем только файлы (не директории)
    files = [f for f in files if f.is_file()]
    
    for file_path in files:
        file_type = get_file_type(str(file_path))
        
        if file_type == 'unknown':
            continue
        
        try:
            text, _ = parse_file(str(file_path))
            if text.strip():  # Пропускаем пустые файлы
                results.append((str(file_path), text, file_type))
        except Exception as e:
            print(f"⚠️ Пропущен файл {file_path}: {e}")
            continue
    
    return results


if __name__ == "__main__":
    # Пример использования
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python parser.py <путь_к_файлу_или_директории>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isfile(path):
        text, file_type = parse_file(path)
        print(f"📄 Тип файла: {file_type}")
        print(f"📝 Длина текста: {len(text)} символов")
        print(f"\n🔍 Первые 500 символов:\n{text[:500]}...")
    elif os.path.isdir(path):
        results = parse_directory(path)
        print(f"📁 Найдено файлов: {len(results)}")
        for file_path, text, file_type in results:
            print(f"  📄 {file_path} ({file_type}): {len(text)} символов")
    else:
        print(f"❌ Путь не существует: {path}")
