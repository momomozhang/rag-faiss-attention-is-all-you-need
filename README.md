# FAISS-powered Document Q&A

Built this after working through the Medium analyzer project - wanted to try FAISS instead of Pinecone for vector storage. Uses the "Attention Is All You Need" paper as a test case since it's pretty dense material.

## What I learned

The "Attention Is All You Need" paper makes for a good test case - it's technical enough to be challenging but well-written enough that retrieval actually works.

**FAISS vs Pinecone differences:**
- FAISS: Faster for small datasets, no API costs, runs offline
- Pinecone: Better for production scale, managed service, easier setup
- Both: Pretty similar accuracy for document Q&A

**Chunking strategy matters more than I expected.** Started with 500-char chunks but answers were too fragmented. 1000 chars with 30 overlap works better for academic papers, though you might need different settings for other content types.

## How it works

Takes the Transformer paper, breaks it down into chunks, creates embeddings, then lets you ask questions about it. Got tired of digging through academic papers manually, so this seemed like a good use case.

## Setup

You'll need an OpenAI API key first.

```bash
pipenv install
```

Add your key to a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

Then just run:
```bash
python main.py
```

## Tech details
- FAISS handles the vector search
- OpenAI's ada-002 for embeddings  
- GPT-4 for the actual responses
- PyPDF to parse the paper
- Everything gets saved locally