import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)

DB_NAME = "chat_db"
GLOBAL_CHATBOT_COLLECTION_NAME = "chatbot_histories"
PROBLEM_CHATBOT_COLLECTION_NAME = "problem_chatbot_histories"
LESSON_CHATBOT_COLLECTION_NAME = "lesson_chatbot_histories"

db = client[DB_NAME]
global_chatbot_collection = db[GLOBAL_CHATBOT_COLLECTION_NAME]
problem_chatbot_collection = db[PROBLEM_CHATBOT_COLLECTION_NAME]
lesson_chatbot_collection = db[LESSON_CHATBOT_COLLECTION_NAME]