# File: main.py
from dotenv import load_dotenv
load_dotenv()
import warnings

#import logging
#logging.basicConfig(level=logging.DEBUG)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from app.routers import (
    auth,
    temp_cleanup,
    compare, 
    browse_user_routes,
)





warnings.filterwarnings("ignore", category=FutureWarning, module=".*common.*")

app = FastAPI(debug=True)

@app.get("/", include_in_schema=False)
async def _root():
    return {"status": "ok"}

app.mount("/static", StaticFiles(directory="temp_uploads"), name="static")

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://3.17.191.244", 
        "https://routemap-react-app.s3.us-east-2.amazonaws.com/vite.svg",     
        ],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers

app.include_router(auth.router, prefix="/api", tags=["Authentication"])
app.include_router(temp_cleanup.router, prefix="/api", tags=["Temp File Cleanup"])
app.include_router(compare.router, prefix="/api", tags=["Keypoint Comparison"])
app.include_router(browse_user_routes.router, prefix="/api", tags=["Browse User Routes"])

#for route in app.routes:
    #print("Route:", route.path)

#if __name__ == "__main__":
    #uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)


