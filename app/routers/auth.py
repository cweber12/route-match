from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import boto3
import bcrypt
import os
from dotenv import load_dotenv

load_dotenv()

# Setup AWS DynamoDB connection
dynamodb = boto3.resource(
    'dynamodb',
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)
table = dynamodb.Table('users')

# Set the prefix and tags here!
router = APIRouter()

# User models
class UserIn(BaseModel):
    username: str
    email: str
    password: str

class LoginIn(BaseModel):
    username: str
    password: str

@router.post("/register")
async def register(user: UserIn):
    print("/register endpoint hit!")
    response = table.get_item(Key={'userName': user.username})

    print("Table name:", table.name)
    print("Region:", os.getenv("AWS_REGION"))

    if 'Item' in response:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_pw = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode()
    table.put_item(Item={
        'userName': user.username,
        'email': user.email,
        'password_hash': hashed_pw,

    })
    return {"message": "User registered successfully"}

@router.post("/login")
async def login(data: LoginIn):
    response = table.get_item(Key={'userName': data.username})
    user = response.get("Item")
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if not bcrypt.checkpw(data.password.encode(), user['password_hash'].encode()):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {"message": "Login successful", "userName": data.username}
