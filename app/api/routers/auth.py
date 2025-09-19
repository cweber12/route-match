# app/routers/auth.py
# ------------------------------------------------------------------------
# This file handles user authentication, including registration and login.
# It uses AWS DynamoDB to store user credentials securely.
# ------------------------------------------------------------------------

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

# Define the DynamoDB users table
# Ensure the table exists before using it
table = dynamodb.Table('users')

router = APIRouter()

# Define Pydantic models for registration input
class UserIn(BaseModel):
    username: str
    email: str
    password: str

# Define Pydantic model for login input
class LoginIn(BaseModel):
    username: str
    password: str

# Register a new user
@router.post("/register")
async def register(user: UserIn):
    print("/register endpoint hit!")
    response = table.get_item(Key={'userName': user.username})
    print("Table name:", table.name)
    print("Region:", os.getenv("AWS_REGION"))
    # If user already exists, raise an error
    if 'Item' in response:
        raise HTTPException(status_code=400, detail="Username already exists")
    # Hash the password and store the user
    hashed_pw = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode()
    table.put_item(Item={
        'userName': user.username,
        'email': user.email,
        'password_hash': hashed_pw,

    })
    return {"message": "User registered successfully"}

# Login a user
@router.post("/login")
async def login(data: LoginIn):
    # Check if user exists
    response = table.get_item(Key={'userName': data.username})
    # Get the user item from the response
    user = response.get("Item")
    # Validate input for username and password
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not bcrypt.checkpw(data.password.encode(), user['password_hash'].encode()):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {"message": "Login successful", "userName": data.username}
