from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from ccpa_parser import parse_statute
from ccpa_indexer import CCPAIndexer
from ccpa_searcher import CCPASearcher
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import json 
import re
import os

# Load environment variables
load_dotenv()

# Initialize models and indexes
chunks = parse_statute(Path("data/ccpa_statute.txt"))
indexer = CCPAIndexer(chunks)
searcher = CCPASearcher(indexer, chunks)

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN"),
    provider="auto",
)

# FastAPI app
app = FastAPI(title="CCPA Compliance Analyzer")

# Request/Response models
class AnalyzeRequest(BaseModel):
    prompt: str

class AnalyzeResponse(BaseModel):
    harmful: bool
    articles: list[str]

def analyze(prompt: str) -> dict:
    """Analyze a business practice for CCPA compliance violations."""
    context = searcher.format_for_llm(prompt, top_k=8)
    
    prompt_text = f"""You are a CCPA compliance classifier. Reply ONLY with valid JSON, no other text.

Relevant CCPA sections:
{context}

Business practice: "{prompt}"

Reply ONLY with this exact JSON format:
{{"harmful": true/false, "articles": ["Section 1798.xxx", ...]}}

Rules:
- Only cite sections from the list above
- articles must be [] if harmful is false
- If harmful is true, articles must be non-empty"""
    
    try:
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324",
            messages=[{"role": "user", "content": prompt_text}]
        )
        
        text = completion.choices[0].message.content.strip()
        
        # Strip markdown code blocks if present
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = text.strip()
        
        # Extract JSON if model adds extra text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            result = json.loads(text)
        
        # Ensure articles is empty list if harmful is false
        if not result.get("harmful", False):
            result["articles"] = []
        
        return result
    except Exception as e:
        print(f"Error: {e}")
        return {"harmful": False, "articles": [], "error": str(e)}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze a business practice for CCPA compliance violations.
    
    Request Body:
    {
        "prompt": "<natural language description of a business practice>"
    }
    
    Response Body:
    {
        "harmful": true/false,
        "articles": ["Section 1798.xxx", ...]
    }
    """
    result = analyze(request.prompt)
    
    # Handle error case
    if "error" in result:
        return AnalyzeResponse(
            harmful=False,
            articles=[]
        )
    
    return AnalyzeResponse(
        harmful=result.get("harmful", False),
        articles=result.get("articles", [])
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
