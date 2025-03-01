import requests

url = "http://localhost:8006/global_chatbot/stream"
headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
data = {
    "message": "This is a test prompt at 15h33",
    "user_id": "d",
    "model": "llama3",
    "thread_id": "1"
}

response = requests.post(url, json=data, headers=headers, stream=True)

for line in response.iter_lines():
    if line:
        print(line.decode("utf-8"))  # Print each streamed line
