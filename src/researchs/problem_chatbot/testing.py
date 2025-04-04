import uuid
import requests

# Define the URL
url = "http://localhost:8006/ai/stream/summarize_assistant"

# Read the problem from the problem.md file
with open('/Users/mac/HCMUS/datn/agent-service-toolkit/src/researchs/problem_chatbot/problem.md', 'r') as file:
    problem = file.read()

# # Define the parameters
# params = {
#     "message": f"Problem: {problem} Problem_id: 123 Question: List of strategies which can solve this problem",
#     "user_id": "d",
#     "thread_id": str(uuid.uuid4()),
#     "model": "llama3.2"
# }
# Define the parameters
params = {
    "message": f"course name: Stack, id: 95713603-63d1-4b75-8a89-1acdc0977459, regenerate: true",
    # "user_id": "d",
    "thread_id": str(uuid.uuid4()),
    "model": "llama3.2"
}

# Make the POST request
response = requests.post(url, json=params)

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())

# Write the AI response to a file
with open(f'/Users/mac/HCMUS/datn/agent-service-toolkit/src/researchs/problem_chatbot/threads/ai_response_{params["thread_id"]}.txt', 'a') as file:
    file.write(response.text + '\n')
# for line in response.iter_lines():
#     if line:
#         print(line.decode("utf-8"))  # Print each streamed line
