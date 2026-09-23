# BlockGuard — Entry Point
# Run: python -m blockguard
# Or: python main.py

import uvicorn
from blockguard.proxy import app
from blockguard.config import cfg, load_config

def main():
    """Start BlockGuard proxy server."""
    config = load_config()
    print(f"BlockGuard starting — tier: {config.domain.value}")
    print(f"Risk budget: {config.risk_budget}")
    print(f"Sampling count: {config.sampling_count}")
    print(f"Server: http://0.0.0.0:8000")
    print(f"Health: http://0.0.0.0:8000/health")
    print(f"Guard endpoint: http://0.0.0.0:8000/guard")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
