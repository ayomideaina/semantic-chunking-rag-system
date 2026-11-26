const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const { GoogleGenerativeAI } = require('@google/generative-ai');
const { ChromaClient } = require('chromadb');

let pdfParse;
try {
  pdfParse = require('pdf-parse');
} catch {
  pdfParse = null;
}

let mammoth;
try {
  mammoth = require('mammoth');
} catch {
  mammoth = null;
}

// Configuration
const HF_API_KEY = process.env.HF_API_KEY || '';
const EMBED_MODEL_NAME = process.env.EMBED_MODEL_NAME || 'sentence-transformers/all-MiniLM-L6-v2';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';
const LLM_MODEL_NAME = process.env.LLM_MODEL_NAME || 'gemini-2.5-flash';
const CHROMA_DB_HOST = process.env.CHROMA_DB_HOST || 'localhost';
const CHROMA_DB_PORT = parseInt(process.env.CHROMA_DB_PORT || '8000', 10);
const RAG_DATA_DIR = process.env.RAG_DATA_DIR || './data';
const CHUNK_LENGTH = parseInt(process.env.CHUNK_LENGTH || '500', 10);
const SERVER_PORT = parseInt(process.env.SERVER_PORT || '3000', 10);


const BREAKPOINT_PERCENTILE = parseFloat(process.env.BREAKPOINT_PERCENTILE || '25');
const EMBED_BATCH_SIZE = parseInt(process.env.EMBED_BATCH_SIZE || '32', 10);
// Used only when LLM_MODEL_NAME keeps returning "overloaded" errors.
const LLM_FALLBACK_MODEL = process.env.LLM_FALLBACK_MODEL || 'gemini-flash-lite-latest';

if (!fs.existsSync(RAG_DATA_DIR)) {
  fs.mkdirSync(RAG_DATA_DIR, { recursive: true });
}

if (HF_API_KEY && !process.env.HF_TOKEN) {
  process.env.HF_TOKEN = HF_API_KEY;
}

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const llmModel = genAI.getGenerativeModel({ model: LLM_MODEL_NAME });

const chromaHost = CHROMA_DB_HOST === 'local' ? 'localhost' : CHROMA_DB_HOST;
const chromaClient = new ChromaClient({ host: chromaHost, port: CHROMA_DB_PORT });
const COLLECTION_NAME = 'rag_documents';
let collection = null;

async function getCollection() {
  if (!collection) {
    collection = await chromaClient.getOrCreateCollection({ name: COLLECTION_NAME });
  }
  return collection;
}

// Helper functions

// Extract plain text out of common upload formats.
async function extractText(filename, buffer) {
  const lower = filename.toLowerCase();

  if (lower.endsWith('.txt') || lower.endsWith('.md')) {
    return buffer.toString('utf-8');
  }

  if (lower.endsWith('.pdf')) {
    if (!pdfParse) throw new Error('pdf-parse not installed');
    const parsed = await pdfParse(buffer);
    return parsed.text;
  }

  if (lower.endsWith('.docx')) {
    if (!mammoth) throw new Error('mammoth not installed');
    const result = await mammoth.extractRawText({ buffer });
    return result.value;
  }

  // Fall back to treating anything else as plain text.
  return buffer.toString('utf-8');
}

// Step 1: split text into sentences
function splitIntoSentences(text, maxLength) {
  const cleaned = text.replace(/\s+/g, ' ').trim();
  if (!cleaned) return [];

  const sentences = cleaned
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);

  const result = [];
  for (const sentence of sentences) {
    if (sentence.length <= maxLength) {
      result.push(sentence);
      continue;
    }
    let piece = '';
    for (const word of sentence.split(' ')) {
      if (piece && piece.length + 1 + word.length > maxLength) {
        result.push(piece);
        piece = word;
      } else {
        piece = piece ? `${piece} ${word}` : word;
      }
    }
    if (piece) result.push(piece);
  }
  return result;
}

// Step 2: embeddings with the HuggingFace sentence-transformer model,
let extractorPromise = null;

async function loadExtractor() {
  const { pipeline } = await import('@huggingface/transformers');
  try {
    return await pipeline('feature-extraction', EMBED_MODEL_NAME);
  } catch (err) {
    const fallback = `Xenova/${EMBED_MODEL_NAME.split('/').pop()}`;
    console.warn(`Could not load ${EMBED_MODEL_NAME} (${err.message}); trying ${fallback}`);
    return pipeline('feature-extraction', fallback);
  }
}

function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = loadExtractor().catch((err) => {
      extractorPromise = null;
      throw err;
    });
  }
  return extractorPromise;
}


async function embedTexts(texts) {
  const extractor = await getExtractor();
  const embeddings = [];
  for (let i = 0; i < texts.length; i += EMBED_BATCH_SIZE) {
    const batch = texts.slice(i, i + EMBED_BATCH_SIZE);
    const output = await extractor(batch, { pooling: 'mean', normalize: true });
    embeddings.push(...output.tolist());
  }
  return embeddings;
}

function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function percentile(values, p) {
  const sorted = [...values].sort((x, y) => x - y);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.floor((p / 100) * sorted.length)));
  return sorted[index];
}

// Step 3: semantic chunking

async function semanticChunk(text, chunkLength) {
  const sentences = splitIntoSentences(text, chunkLength);
  if (sentences.length <= 1) return sentences;

  const sentenceEmbeddings = await embedTexts(sentences);

  const similarities = [];
  for (let i = 0; i < sentences.length - 1; i++) {
    similarities.push(cosineSimilarity(sentenceEmbeddings[i], sentenceEmbeddings[i + 1]));
  }
  const threshold = percentile(similarities, BREAKPOINT_PERCENTILE);

  const chunks = [];
  let current = sentences[0];

  for (let i = 1; i < sentences.length; i++) {
    const sentence = sentences[i];
    const topicChanged = similarities[i - 1] < threshold;
    const tooLong = current.length + 1 + sentence.length > chunkLength;

    if (topicChanged || tooLong) {
      chunks.push(current);
      current = sentence;
    } else {
      current += ' ' + sentence;
    }
  }
  chunks.push(current);

  return chunks;
}

async function generateAnswer(query, contextChunks) {
  const contextText = contextChunks.length
    ? contextChunks.map((c) => `- ${c}`).join('\n\n')
    : '(no relevant context found)';

  const prompt = `You are a helpful assistant. Answer the user's question using ONLY the context provided below. If the answer is not contained in the context, say you don't have enough information to answer.

Context:
${contextText}

Question: ${query}

Answer:`;

  try {
    const result = await withRetry(() => llmModel.generateContent(prompt));
    return result.response.text();
  } catch (err) {
    // If the main model is still overloaded after retrying, answer with a
    // lighter model instead of failing the request.
    const status = String(err.status || (String(err.message).match(/\[(\d{3}) /) || [])[1]);
    if (!['429', '503'].includes(status) || LLM_FALLBACK_MODEL === LLM_MODEL_NAME) throw err;
    console.warn(`${LLM_MODEL_NAME} unavailable (${status}); falling back to ${LLM_FALLBACK_MODEL}`);
    const fallbackModel = genAI.getGenerativeModel({ model: LLM_FALLBACK_MODEL });
    const result = await withRetry(() => fallbackModel.generateContent(prompt));
    return result.response.text();
  }
}


async function withRetry(fn, maxAttempts = 4) {
  for (let attempt = 1; ; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const status = err.status || (String(err.message).match(/\[(\d{3}) /) || [])[1];
      const retryable = ['429', '500', '503'].includes(String(status));
      if (!retryable || attempt >= maxAttempts) throw err;
      const delayMs = 1000 * 2 ** (attempt - 1);
      console.warn(`Gemini returned ${status}; retrying in ${delayMs / 1000}s (attempt ${attempt + 1}/${maxAttempts})`);
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
}


const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage() });

app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok' });
});

app.post('/upload', upload.array('files'), async (req, res) => {
  try {
    const files = req.files || [];
    const uploadedSummary = [];
    let totalChunks = 0;
    const col = await getCollection();

    if (files.length === 0) {
      return res.status(400).json({ error: "No files received. Send them in the 'files' field." });
    }

    for (const file of files) {
      const filename = path.basename(file.originalname);
      fs.writeFileSync(path.join(RAG_DATA_DIR, filename), file.buffer);

      const text = await extractText(filename, file.buffer);
      const chunks = await semanticChunk(text, CHUNK_LENGTH);
      await col.delete({ where: { source: filename } });

      if (chunks.length > 0) {
        const embeddings = await embedTexts(chunks);
        const ids = chunks.map((_, i) => `${filename}-${i}`);
        const metadatas = chunks.map(() => ({ source: filename }));
        await col.add({ ids, embeddings, documents: chunks, metadatas });
        totalChunks += chunks.length;
      }

      uploadedSummary.push({ filename, chunks: chunks.length });
    }

    res.json({ uploaded: uploadedSummary, total_chunks: totalChunks });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.post('/prompt', async (req, res) => {
  try {
    const { query } = req.body || {};
    if (!query) {
      return res.status(400).json({ error: "'query' is required" });
    }

    const [queryEmbedding] = await embedTexts([query]);
    const col = await getCollection();
    const results = await col.query({ queryEmbeddings: [queryEmbedding], nResults: 4 });
    const retrievedDocs = (results.documents && results.documents[0]) || [];

    const answer = await generateAnswer(query, retrievedDocs);

    res.json({ answer, sources: retrievedDocs });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(SERVER_PORT, () => {
  console.log(`RAG API listening on port ${SERVER_PORT}`);
  console.log(`LLM model: ${LLM_MODEL_NAME} | Embedding model: ${EMBED_MODEL_NAME}`);
});
