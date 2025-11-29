import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import src
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from the root .env file
env_path = Path(__file__).resolve().parent.parent / ".env"
print(f"Loading .env from: {env_path}")
load_dotenv(dotenv_path=env_path, override=True)

if not os.getenv("DASHSCOPE_API_KEY"):
    print("WARNING: DASHSCOPE_API_KEY not found in environment after loading .env")
else:
    print("DASHSCOPE_API_KEY loaded successfully.")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.config import AgentConfig
from src.graph import build_app

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
print("Initializing Agent...")
try:
    cfg = AgentConfig()
    # Ensure API key is present (in case it wasn't loaded when module was imported)
    if not cfg.api_key:
        cfg.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        
    # Optional: Override trace path to avoid cluttering main logs
    # cfg.trace_path = "demo/demo_trace.jsonl"
    run_agent = build_app(cfg)
    print("Agent initialized successfully.")
except Exception as e:
    print(f"Error initializing agent: {e}")
    run_agent = None

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    if not run_agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        print(f"Received question: {query.question}")
        # Run the agent
        result = run_agent(query.question)
        return result
    except Exception as e:
        print(f"Error running agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
