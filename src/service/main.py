
import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
# uri = "mongodb+srv://graduation21072003:NKRUTRHmtv2hlTlk@intellab.6yiaf.mongodb.net/?retryWrites=true&w=majority&appName=Intellab"
uri = os.getenv("MONGODB_URI")
print(uri)
# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)