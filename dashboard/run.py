# Run dashboard server independently
import os
import uvicorn
from .app import app

if __name__ == "__main__":
    # Railway sets HOST=0.0.0.0 by default; local dev uses 127.0.0.1
    host = os.getenv("HOST", os.getenv("DASHBOARD_HOST", "0.0.0.0"))
    port = int(os.getenv("PORT", os.getenv("DASHBOARD_PORT", "8000")))
    uvicorn.run(app, host=host, port=port, reload=False)