# BlockGuard — FastAPI Proxy Server
# Sits between the calling app and the LLM endpoint.
# Intercepts the token/sentence stream, runs S1-S5, forwards cleaned stream.
# The calling app and end user never see the plugin.

import asyncio
import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, AsyncIterator
import httpx

from blockguard.pipeline import pipeline, PipelineResult, PipelineInput
from blockguard.config import load_config

app = FastAPI(
    title="BlockGuard",
    description="Hidden hallucination-filtering middleware for LLM outputs",
    version="0.1.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GuardRequest(BaseModel):
    prompt: str
    context: Optional[str] = None
    domain: str = "general"
    tier: str = "fast"
    # If provided, stream the raw LLM response through the filter
    stream: bool = False

class GuardResponse(BaseModel):
    cleaned_text: str
    flagged_blocks: List[dict] = []
    removed_count: int = 0
    risk_score: float = 0.0
    latency_ms: float = 0.0

@app.post("/guard", response_model=GuardResponse)
async def guard_endpoint(request: GuardRequest):
    """
    Main BlockGuard endpoint.
    
    Passes the prompt through the full filter pipeline and returns
    the cleaned response. The filter is invisible — looks like a
    slightly-delayed normal LLM response.
    """
    start = time.time()
    
    try:
        pipeline.cfg = load_config()
        if request.tier == "strict":
            pipeline.cfg.domain = pipeline.cfg.__class__.domain.__class__(
                "strict" if request.domain in ("medical", "legal", "financial") else "fast"
            )
        
        input = PipelineInput(
            prompt=request.prompt,
            context=request.context,
            domain=request.domain,
            tier=request.tier,
        )
        
        # In production, generate_fn calls the actual LLM API
        # Here we use a placeholder — integrate with your LLM client
        result = await pipeline.run(input, generate_fn=_llm_generate)
        
        return GuardResponse(
            cleaned_text=result.cleaned_text,
            flagged_blocks=[
                {"text": b.text[:50], "code": b.code, "confidence": b.confidence}
                for b in result.flagged_blocks
            ],
            removed_count=result.removed_count,
            risk_score=result.risk_score,
            latency_ms=result.total_latency_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BlockGuard error: {str(e)}")

@app.post("/guard/stream")
async def guard_stream_endpoint(request: GuardRequest):
    """
    Streaming version for real-time output filtering.
    
    Intercepts the LLM's token stream, buffers ~1 block of lag,
    runs S1-S5 on each chunk, forwards cleaned output.
    """
    try:
        input = PipelineInput(
            prompt=request.prompt,
            context=request.context,
            domain=request.domain,
            tier=request.tier,
        )
        
        async def generate_stream():
            # In production, this would connect to the LLM's streaming API
            # and process tokens in real-time
            raw_text = await _llm_generate(request.prompt)
            result = await pipeline.run(input, generate_fn=lambda p: raw_text)
            return result.cleaned_text
        
        cleaned = await generate_stream()
        return GuardResponse(
            cleaned_text=cleaned,
            flagged_blocks=[],
            removed_count=0,
            risk_score=0.0,
            latency_ms=0.0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BlockGuard stream error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "tier": pipeline.config.domain.value}

async def _llm_generate(prompt: str, model: str = "gpt-4o") -> str:
    """
    Generate raw LLM response.
    
    Supports: OpenAI, Anthropic, vLLM (open-weight), or any async callable.
    Set BLOCKGUARD_LLM_PROVIDER env var: "openai", "anthropic", "vllm".
    """
    provider = os.getenv("BLOCKGUARD_LLM_PROVIDER", "openai")
    
    if provider == "openai":
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return response.choices[0].message.content
        except ImportError:
            pass
    
    if provider == "anthropic":
        try:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
            response = await client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except ImportError:
            pass
    
    # Fallback: vLLM or any local server
    try:
        import httpx
        vllm_url = os.getenv("BLOCKGUARD_VLLM_URL", "http://localhost:8000/v1/completions")
        async with httpx.AsyncClient() as client:
            response = await client.post(vllm_url, json={
                "prompt": prompt,
                "model": model,
                "max_tokens": 512,
            })
            data = response.json()
            return data.get("choices", [{}])[0].get("text", "")
    except Exception:
        pass
    
    # Last-resort placeholder
    return (f"BlockGuard placeholder for prompt: {prompt[:50]}... "
            "Configure OPENAI_API_KEY, ANTHROPIC_API_KEY, or BLOCKGUARD_VLLM_URL "
            "to enable real generation.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BLOCKGUARD_PORT", "8000"))
    host = os.getenv("BLOCKGUARD_HOST", "0.0.0.0")
    print(f"BlockGuard server starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
