# Run dashboard server independently
import os
import uvicorn
from .app import app

if __name__ == "__main__":
    host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.getenv("DASHBOARD_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, reload=False)