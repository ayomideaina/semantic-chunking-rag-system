# Semantic Chunking RAG System (Node.js)

A Retrieval-Augmented Generation (RAG) API. Upload documents, then ask questions that are answered using only the content of those documents.

- **Semantic chunking**: text is split where the topic changes, with chunks capped at `CHUNK_LENGTH` characters
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, run locally with HuggingFace transformers.js
- **Vector storage**: ChromaDB
- **Generation**: Gemini Flash
- **Configuration**: environment variables from a `.env` file
- No LangChain

## Requirements

- Node.js 18 or later
- Python 3 (only to run the ChromaDB server), or Docker

## Setup

1. **Install dependencies**
   ```bash
   pnpm install      # or: npm install
   ```

2. **Create `.env`** by copying `.env.example` and filling in your keys:

   | Variable | Description | Example |
   |---|---|---|
   | `HF_API_KEY` | HuggingFace token (used to download the embedding model) | `hf_...` |
   | `EMBED_MODEL_NAME` | Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
   | `GEMINI_API_KEY` | Google AI Studio API key | `AIza...` |
   | `LLM_MODEL_NAME` | Gemini model | `gemini-2.5-flash` |
   | `CHROMA_DB_HOST` | ChromaDB host | `localhost` |
   | `CHROMA_DB_PORT` | ChromaDB port | `8000` |
   | `RAG_DATA_DIR` | Folder where uploaded files are saved | `rag_data` |
   | `CHUNK_LENGTH` | Maximum characters per chunk | `1000` |
   | `SERVER_PORT` | Port for this API | `3000` |

   Optional variables (with defaults, no need to set them):

   | Variable | Default | Description |
   |---|---|---|
   | `LLM_FALLBACK_MODEL` | `gemini-flash-lite-latest` | Used if the main model stays overloaded (503/429) |
   | `BREAKPOINT_PERCENTILE` | `25` | Lower values give fewer, larger chunks |
   | `EMBED_BATCH_SIZE` | `32` | Texts embedded per batch |

   > **Note:** Google no longer offers `gemini-2.5-flash` to newly created API keys. If you get a
   > `404 ... no longer available to new users` error, set `LLM_MODEL_NAME` to a current Flash
   > model (for example `gemini-3.8-flash`).

3. **Start ChromaDB** (in its own terminal) on the host and port set in `.env`:
   ```bash
   pip install chromadb
   chroma run --path ./chroma_data --port 8000
   ```
   Or with Docker: `docker run -p 8000:8000 chromadb/chroma`.
   (The npm `npx chroma run` CLI does not support Windows x64.)

4. **Start the API** (in another terminal):
   ```bash
   node main.js
   ```
   You should see:
   ```
   RAG API listening on port 3000
   LLM model: ... | Embedding model: sentence-transformers/all-MiniLM-L6-v2
   ```
   The first upload downloads the embedding model (about 90 MB). After that it is cached.

## Endpoints

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/health` | none | `200 {"status":"ok"}` |
| POST | `/upload` | `multipart/form-data`, field `files` (one or more `.txt`, `.md`, `.pdf`, `.docx`) | `{"uploaded":[{"filename","chunks"}],"total_chunks"}` |
| POST | `/prompt` | `application/json`, `{"query": "..."}` | `{"answer","sources"}` |

Uploading a file with the same name again replaces its old chunks instead of duplicating them.

## Try it

```bash
cat > sample.txt << 'EOF'
Lagos is the largest city in Nigeria, with over 15 million people. It was the capital of Nigeria until 1991, when the capital moved to Abuja. The city is known for its music scene and busy markets.
Python is a programming language created by Guido van Rossum. It was first released in 1991. Python is popular for data science, web development, and automation.
Honey never spoils if it is stored properly. Archaeologists have found edible honey in ancient Egyptian tombs. Bees make honey from the nectar of flowers.
EOF

curl http://localhost:3000/health
curl -F "files=@sample.txt" http://localhost:3000/upload
curl -X POST http://localhost:3000/prompt -H "Content-Type: application/json" -d '{"query":"What is the capital of Nigeria now?"}'
curl -X POST http://localhost:3000/prompt -H "Content-Type: application/json" -d '{"query":"What happened in 1991?"}'
curl -X POST http://localhost:3000/prompt -H "Content-Type: application/json" -d '{"query":"What is the population of Tokyo?"}'
```

The upload should report about 3 chunks (one per topic). The last question is not covered by the file, so the API should say it does not have enough information.

## How it works

**Upload:** file → extract text → split into sentences → embed sentences → group into chunks by topic → embed chunks → store in ChromaDB

**Prompt:** question → embed → ChromaDB returns the 4 most similar chunks → Gemini answers using only those chunks

### Semantic chunking

1. Split the text into sentences. Any sentence longer than `CHUNK_LENGTH` is broken on word boundaries.
2. Embed each sentence.
3. Compute the cosine similarity between each sentence and the next one.
4. Start a new chunk where the similarity is in the lowest `BREAKPOINT_PERCENTILE` percent (a topic change), or where adding the next sentence would exceed `CHUNK_LENGTH`.

### Reliability

If Gemini returns 429, 500 or 503, the request is retried after 1, 2 and 4 seconds. If the model is still overloaded, the answer is generated with `LLM_FALLBACK_MODEL`.

## Troubleshooting

| Error | Fix |
|---|---|
| `curl: (26) Failed to open/read local data` | The file you named in `files=@...` does not exist in the current folder |
| `Failed to connect to chromadb` | Start ChromaDB, and check `CHROMA_DB_HOST` and `CHROMA_DB_PORT` |
| `404 ... no longer available to new users` | Change `LLM_MODEL_NAME` (see the note in Setup) |
| `503 ... high demand` | Temporary on Google's side. The API retries and falls back automatically |
| `chroma: command not found` | Add Python's `Scripts` folder to your PATH, or run `python -m chromadb.cli.cli run --path ./chroma_data --port 8000` |
| Changes to `.env` have no effect | Restart `node main.js`, because `.env` is only read at startup |
