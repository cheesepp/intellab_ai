import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)

DB_NAME = "chat_db"
COLLECTION_NAME = "chatbot_histories"

db = client[DB_NAME]
collection = db[COLLECTION_NAME]
