"""
Модуль для генерации эмбеддингов через Ollama.
Использует модель nomic-embed-text.
"""

import os
from typing import List, Optional

import requests


# Конфигурация Ollama Embeddings
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 2  # секунды


class OllamaEmbedder:
    """Класс для работы с Ollama Embeddings API."""
    
    def __init__(self, model: str = OLLAMA_EMBED_MODEL):
        """
        Инициализация эмбеддера.
        
        Args:
            model: модель для эмбеддингов (по умолчанию nomic-embed-text)
        """
        self.model = model
        self.embedding_dim = None  # Определится при первом запросе
        
        # Проверяем доступность Ollama
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Ollama не отвечает")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "❌ Не удалось подключиться к Ollama!\n"
                "Убедитесь, что Ollama запущена: ollama serve"
            )
    
    def _make_request(self, text: str) -> List[float]:
        """
        Выполнение запроса к API для одного текста.
        
        Args:
            text: текст для эмбеддинга
            
        Returns:
            эмбеддинг (список чисел)
        """
        payload = {
            "model": self.model,
            "input": text
        }
        
        try:
            response = requests.post(
                OLLAMA_EMBED_URL,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"❌ Ошибка Ollama ({response.status_code}): {response.text}")
            
            data = response.json()
            
            # Ollama возвращает embeddings как список списков
            if "embeddings" in data and len(data["embeddings"]) > 0:
                embedding = data["embeddings"][0]
            elif "embedding" in data:
                embedding = data["embedding"]
            else:
                raise RuntimeError(f"❌ Неожиданный формат ответа: {data}")
            
            # Сохраняем размерность эмбеддинга
            if self.embedding_dim is None:
                self.embedding_dim = len(embedding)
            
            return embedding
            
        except requests.exceptions.Timeout:
            raise RuntimeError("❌ Таймаут при запросе к Ollama")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("❌ Потеряно соединение с Ollama")
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Генерация эмбеддингов для списка текстов.
        
        Args:
            texts: список текстов
            batch_size: размер батча (не используется, Ollama обрабатывает по одному)
            show_progress: показывать прогресс-бар
            
        Returns:
            список эмбеддингов
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="🧠 Генерация эмбеддингов")
            except ImportError:
                iterator = texts
                print(f"🧠 Генерация эмбеддингов (всего: {len(texts)})")
        else:
            iterator = texts
        
        for text in iterator:
            embedding = self._make_request(text)
            all_embeddings.append(embedding)
        
        return all_embeddings
    
    def embed_single(self, text: str) -> List[float]:
        """
        Генерация эмбеддинга для одного текста.
        
        Args:
            text: текст для эмбеддинга
            
        Returns:
            эмбеддинг (список чисел)
        """
        return self._make_request(text)
    
    def get_embedding_dim(self) -> int:
        """
        Получение размерности эмбеддинга.
        Если ещё не определена - делает тестовый запрос.
        
        Returns:
            размерность эмбеддинга
        """
        if self.embedding_dim is None:
            self.embed_single("test")
        return self.embedding_dim


# Алиас для обратной совместимости
DeepSeekEmbedder = OllamaEmbedder


if __name__ == "__main__":
    # Пример использования
    print("🧪 Тестирование Ollama Embedder\n")
    
    try:
        embedder = OllamaEmbedder()
        
        # Тестовые тексты
        test_texts = [
            "Python — это язык программирования.",
            "Machine learning helps computers learn from data.",
            "Нейронные сети используются для обработки данных.",
        ]
        
        print(f"📝 Тестовых текстов: {len(test_texts)}")
        
        # Генерируем эмбеддинги
        embeddings = embedder.embed_texts(test_texts)
        
        print(f"\n✅ Успешно получено эмбеддингов: {len(embeddings)}")
        print(f"📐 Размерность эмбеддинга: {len(embeddings[0])}")
        
        # Показываем первые значения
        for i, (text, emb) in enumerate(zip(test_texts, embeddings)):
            print(f"\n📄 Текст {i + 1}: {text[:50]}...")
            print(f"🔢 Первые 5 значений: {emb[:5]}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
