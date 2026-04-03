"""
MediTutor AI - Configuration
All settings loaded from environment variables with safe defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = DATA_DIR / "vectors"
DB_DIR = DATA_DIR / "db"
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"

for d in [DATA_DIR, VECTOR_DIR, DB_DIR, UPLOAD_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── API Keys ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# ─── LLM Model Configuration ─────────────────────────────────────────────────
# Primary: Groq (fastest free tier)
GROQ_MODELS = [
    "llama-3.1-8b-instant",       # fastest
    "llama3-8b-8192",             # fallback groq model
    "gemma2-9b-it",               # secondary fallback
]

# Secondary: HuggingFace Inference API
HF_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/phi-2",
]

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HF_API_URL = "https://api-inference.huggingface.co/models/{model}"

# ─── Embedding Model (Local, FREE - no API needed) ───────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ─── PDF Processing ───────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_PDF_SIZE_MB = 50

# ─── RAG Settings ─────────────────────────────────────────────────────────────
TOP_K_CHUNKS = 5
MAX_CONTEXT_LENGTH = 3000

# ─── Generation Settings ──────────────────────────────────────────────────────
MAX_TOKENS = 1024
TEMPERATURE = 0.3
FLASHCARD_COUNT = 10
MCQ_COUNT = 5

# ─── Rate Limiting ────────────────────────────────────────────────────────────
GROQ_RPM = 30          # requests per minute (free tier)
HF_RPM = 10            # HuggingFace free tier is slower
REQUEST_TIMEOUT = 60   # seconds
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2        # seconds between retries

# ─── Caching ──────────────────────────────────────────────────────────────────
CACHE_TTL = 3600       # 1 hour in seconds
MAX_CACHE_SIZE = 500   # max cached items

# ─── Database ─────────────────────────────────────────────────────────────────
DATABASE_URL = f"sqlite:///{DB_DIR}/meditutor.db"

# ─── CORS ─────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://meditutor-frontend.onrender.com",
).split(",")

# ─── App Settings ─────────────────────────────────────────────────────────────
APP_NAME = "MediTutor AI"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
