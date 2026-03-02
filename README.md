# CCPA Compliance Analyzer

A CCPA (California Consumer Privacy Act) compliance analysis system using RAG (Retrieval-Augmented Generation) combined with an LLM to classify business practices and cite relevant legal articles.

## Solution Overview

### Architecture
This solution uses a hybrid approach combining:

1. **RAG System**: 
   - Parses the CCPA statute into semantic chunks
   - Builds a TF-IDF based search index for efficient retrieval
   - Retrieves top-k relevant CCPA sections based on the input prompt

2. **LLM Classification**:
   - Uses DeepSeek-V3-0324 (via Hugging Face Inference API)
   - Receives retrieved CCPA sections as context
   - Classifies business practice as harmful/safe
   - Extracts specific CCPA article citations

3. **Pipeline Flow**:
   ```
   User Prompt → RAG Retrieval (Top-K CCPA sections) → 
   LLM with Context → JSON Classification + Citations
   ```

### Components
- **ccpa_parser.py**: Parses CCPA statute into structured chunks
- **ccpa_indexer.py**: Builds TF-IDF index for semantic search
- **ccpa_searcher.py**: Retrieves relevant sections and formats for LLM
- **api.py**: FastAPI server exposing `/analyze` and `/health` endpoints

### Key Features
- RAG-based retrieval ensures citations come from actual statute text
- LLM validates reasoning and extracts specific article numbers
- Strict JSON output validation
- Fast startup with pre-built index
- Stateless API design

## Docker Run Command

Pull and run the container:

```bash
docker pull yourusername/ccpa-compliance:latest
docker run --gpus all -p 8000:8000 -e HF_TOKEN=<your_hf_token> yourusername/ccpa-compliance:latest
```

For CPU-only environments:

```bash
docker run -p 8000:8000 -e HF_TOKEN=<your_hf_token> yourusername/ccpa-compliance:latest
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | Hugging Face API token for accessing Inference API. Must have read permissions. |
| `PORT` | No | Server port (default: 8000). Only change if you need a different internal port. |

## GPU Requirements

- **Recommended**: Not GPU-dependent (uses Hugging Face Inference API)
- **CPU-only**: Fully supported
- **VRAM**: N/A (model runs on Hugging Face's infrastructure)
- **Inference API**: Requires active internet connection and valid HF_TOKEN

**Note**: This solution uses Hugging Face's Inference API, so the model runs on HF's servers rather than locally. This means:
- No GPU required on deployment machine
- Fast startup time (no model download/loading)
- Requires internet connectivity
- Rate limits apply based on HF plan

## Local Setup Instructions (Fallback)

If Docker fails, follow these steps to run manually on Linux/Ubuntu:

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Internet connection

### Step 1: Clone/Extract Source Code
```bash
cd /path/to/ccpa-rag-explorer
```

### Step 2: Create Virtual Environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Set Environment Variable
```bash
export HF_TOKEN="your_huggingface_token_here"
```

### Step 5: Start the Server
```bash
python api.py
```

Or using uvicorn directly:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Step 6: Verify Server is Running
```bash
curl http://localhost:8000/health
```

Expected output:
```json
{"status":"ok"}
```

## API Usage Examples

### Health Check

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{"status": "ok"}
```

### Analyze Business Practice (Violation)

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "We sell customer browsing history to ad networks without notifying them."
  }'
```

**Response:**
```json
{
  "harmful": true,
  "articles": [
    "Section 1798.100",
    "Section 1798.120"
  ]
}
```

### Analyze Business Practice (Compliant)

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "We provide a clear privacy policy and honor all deletion requests."
  }'
```

**Response:**
```json
{
  "harmful": false,
  "articles": []
}
```

### Using Python requests

```python
import requests

# Analyze a business practice
response = requests.post(
    "http://localhost:8000/analyze",
    json={"prompt": "We share user data with third parties without consent."}
)

result = response.json()
print(f"Harmful: {result['harmful']}")
print(f"Articles: {result['articles']}")
```

## Building the Docker Image

To build the image yourself:

```bash
# Build the image
docker build -t ccpa-compliance:latest .

# Test locally
docker run -p 8000:8000 -e HF_TOKEN=<your_token> ccpa-compliance:latest

# Tag for Docker Hub
docker tag ccpa-compliance:latest yourusername/ccpa-compliance:latest

# Push to Docker Hub
docker push yourusername/ccpa-compliance:latest
```

## Testing

Run the test suite:

```bash
pip install pytest
pytest test_api.py -v
```

## Response Format Specification

All responses from `/analyze` endpoint follow this strict format:

```json
{
  "harmful": true | false,
  "articles": ["Section 1798.xxx", ...]
}
```

**Rules:**
- `harmful`: boolean (not string)
- `articles`: empty list `[]` when `harmful` is `false`
- `articles`: non-empty list when `harmful` is `true`
- Article format: Must contain "Section" and/or "1798"

## Troubleshooting

### Container fails to start
- Check that port 8000 is not already in use
- Verify HF_TOKEN is set correctly
- Check Docker logs: `docker logs <container_id>`

### Health check fails
- Wait up to 60 seconds for initialization
- Check if data/ccpa_statute.txt exists
- Verify network connectivity for HF API

### API returns errors
- Ensure HF_TOKEN has valid permissions
- Check internet connectivity
- Review server logs for detailed error messages

### Invalid JSON responses
- Check that response contains both `harmful` and `articles` keys
- Verify `harmful` is boolean (true/false), not string ("true"/"false")
- Ensure `articles` is a list, not null or undefined

## License

This project is created for OPEN HACK 2026 hackathon submission.
