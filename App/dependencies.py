from fastapi import Header, HTTPException
from App.config import API_SECRET

def verify_api_secret(x_api_secret: str = Header(None)):
    if x_api_secret != API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")