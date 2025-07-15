import uvicorn
from dotenv import load_dotenv

from core import settings

load_dotenv()

if __name__ == "__main__":
    print(f"Starting IntelLab AI service on port {settings.is_dev()}")
    uvicorn.run("service:app", host="0.0.0.0", port=settings.PORT, reload=settings.is_dev())