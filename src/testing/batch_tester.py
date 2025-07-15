"""
Batch testing framework for all chatbot types with comprehensive metrics and reporting.
"""
import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import requests
from dataclasses import asdict
import re
import os

from batch_test_questions import (
    ChatbotType, TestQuestion, ResponseMetrics, 
    get_test_questions_for_chatbot, get_all_test_questions,
    calculate_total_score
)

class BatchTester:
    def __init__(self, base_url: str = "http://localhost:8006"):
        self.base_url = base_url
        self.results = []
        self.test_session_id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        
    def _calculate_response_metrics(self, response_text: str, response_time: float, 
                                   error_occurred: bool = False, error_message: str = "") -> ResponseMetrics:
        """Calculate metrics for a response."""
        
        # Basic metrics
        word_count = len(response_text.split()) if response_text else 0
        has_code_examples = bool(re.search(r'```|`[^`]+`', response_text))
        has_step_by_step = bool(re.search(r'\d+\.\s|\n\*\s|\n-\s', response_text))
        contains_urls = bool(re.search(r'https?://', response_text))
        contains_markdown = bool(re.search(r'[*_#]', response_text))
        
        # Initialize with default scores (will be manually evaluated later)
        return ResponseMetrics(
            response_time=response_time,
            word_count=word_count,
            has_code_examples=has_code_examples,
            has_step_by_step=has_step_by_step,
            relevance_score=0.0,  # To be filled manually
            accuracy_score=0.0,   # To be filled manually  
            completeness_score=0.0,  # To be filled manually
            clarity_score=0.0,    # To be filled manually
            total_score=0.0,
            contains_urls=contains_urls,
            contains_markdown=contains_markdown,
            error_occurred=error_occurred,
            error_message=error_message
        )
    
    def _test_agent_endpoint(self, agent_type: str, question: str, context: str = "", 
                           model: str = "llama3.2") -> tuple[str, float, bool, str]:
        """Test a specific agent endpoint."""
        url = f"{self.base_url}/ai/stream/{agent_type}"
        
        # Prepare payload based on agent type
        if agent_type == "summarize_assistant":
            # Use context as the message for summarize assistant
            message = context if context else question
            payload = {
                "message": message,
                "model": model
            }
        elif agent_type in ["problem_chatbot", "lesson_chatbot"]:
            # Use context as the message for structured agents
            message = context if context else question
            payload = {
                "message": message,
                "user_id": "batch_test_user",
                "thread_id": str(uuid.uuid4()),
                "model": model
            }
        else:
            # For other agents, use the question directly
            payload = {
                "message": question,
                "user_id": "batch_test_user", 
                "thread_id": str(uuid.uuid4()),
                "model": model
            }
        
        headers = {
            "Content-Type": "application/json",
            "X-UserRole": "admin,PREMIUM_PLAN"  # Use premium to avoid rate limits
        }
        
        start_time = time.time()
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                # Parse streaming response
                response_text = self._parse_streaming_response(response.text)
                return response_text, response_time, False, ""
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                return "", response_time, True, error_msg
                
        except Exception as e:
            response_time = time.time() - start_time
            return "", response_time, True, str(e)
    
    def _parse_streaming_response(self, response_text: str) -> str:
        """Parse streaming response to extract the final message content."""
        lines = response_text.strip().split('\n')
        final_content = ""
        
        for line in lines:
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                    if data.get('type') == 'message':
                        content = data.get('content', {})
                        if isinstance(content, dict) and content.get('type') == 'ai':
                            final_content = content.get('content', '')
                    elif data.get('type') == 'token':
                        # For token streaming, accumulate tokens
                        token_content = data.get('content', '')
                        final_content += token_content
                except json.JSONDecodeError:
                    continue
        
        return final_content.strip()
    
    def test_single_chatbot(self, chatbot_type: ChatbotType, models: List[str] = ["llama3.2"]) -> List[Dict]:
        """Test a single chatbot type with all its questions."""
        print(f"\n🤖 Testing {chatbot_type.value}...")
        
        questions = get_test_questions_for_chatbot(chatbot_type)
        chatbot_results = []
        
        for question in questions:
            print(f"  📝 Question {question.id}: {question.question[:50]}...")
            
            for model in models:
                print(f"    🧠 Model: {model}")
                
                # Test the question
                response_text, response_time, error_occurred, error_message = self._test_agent_endpoint(
                    chatbot_type.value, question.question, question.context, model
                )
                
                # Calculate metrics
                metrics = self._calculate_response_metrics(
                    response_text, response_time, error_occurred, error_message
                )
                
                # Store result
                result = {
                    "test_session_id": self.test_session_id,
                    "timestamp": self.timestamp,
                    "chatbot_type": chatbot_type.value,
                    "model": model,
                    "question_id": question.id,
                    "question": question.question,
                    "context": question.context,
                    "expected_type": question.expected_type,
                    "difficulty": question.difficulty,
                    "category": question.category,
                    "response": response_text,
                    **asdict(metrics)
                }
                
                chatbot_results.append(result)
                self.results.append(result)
                
                # Small delay between requests
                time.sleep(1)
        
        return chatbot_results
    
    def test_all_chatbots(self, models: List[str] = ["llama3.2"], 
                         exclude_types: List[ChatbotType] = None) -> Dict[str, List[Dict]]:
        """Test all chatbot types."""
        print("🚀 Starting comprehensive batch testing...")
        print(f"📊 Test Session ID: {self.test_session_id}")
        print(f"🕐 Started at: {self.timestamp}")
        
        exclude_types = exclude_types or []
        all_results = {}
        
        for chatbot_type in ChatbotType:
            if chatbot_type not in exclude_types:
                try:
                    results = self.test_single_chatbot(chatbot_type, models)
                    all_results[chatbot_type.value] = results
                except Exception as e:
                    print(f"❌ Error testing {chatbot_type.value}: {e}")
                    continue
        
        print(f"\n✅ Testing completed! Total tests: {len(self.results)}")
        return all_results
    
    def save_results_to_csv(self, filename: str = None) -> str:
        """Save results to CSV file."""
        if not filename:
            filename = f"batch_test_results_{self.test_session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame(self.results)
        
        # Ensure the testing directory exists
        os.makedirs("/Users/johan/HCMUS/datn/intellab_ai/src/testing/results", exist_ok=True)
        filepath = f"/Users/johan/HCMUS/datn/intellab_ai/src/testing/results/{filename}"
        
        df.to_csv(filepath, index=False)
        print(f"📁 Results saved to: {filepath}")
        return filepath
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of the test results."""
        if not self.results:
            return {"error": "No test results available"}
        
        df = pd.DataFrame(self.results)
        
        summary = {
            "test_session_id": self.test_session_id,
            "timestamp": self.timestamp,
            "total_tests": len(self.results),
            "unique_chatbots": df['chatbot_type'].nunique(),
            "unique_models": df['model'].nunique(),
            "success_rate": (df['error_occurred'] == False).mean() * 100,
            "average_response_time": df['response_time'].mean(),
            "avg_word_count": df['word_count'].mean(),
            "responses_with_code": (df['has_code_examples'] == True).sum(),
            "responses_with_steps": (df['has_step_by_step'] == True).sum(),
            "responses_with_urls": (df['contains_urls'] == True).sum(),
            "by_chatbot": {},
            "by_model": {},
            "by_difficulty": {}
        }
        
        # Breakdown by chatbot type
        for chatbot_type in df['chatbot_type'].unique():
            chatbot_df = df[df['chatbot_type'] == chatbot_type]
            summary["by_chatbot"][chatbot_type] = {
                "total_tests": len(chatbot_df),
                "success_rate": (chatbot_df['error_occurred'] == False).mean() * 100,
                "avg_response_time": chatbot_df['response_time'].mean(),
                "avg_word_count": chatbot_df['word_count'].mean()
            }
        
        # Breakdown by model
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            summary["by_model"][model] = {
                "total_tests": len(model_df),
                "success_rate": (model_df['error_occurred'] == False).mean() * 100,
                "avg_response_time": model_df['response_time'].mean(),
                "avg_word_count": model_df['word_count'].mean()
            }
        
        # Breakdown by difficulty
        for difficulty in df['difficulty'].unique():
            diff_df = df[df['difficulty'] == difficulty]
            summary["by_difficulty"][difficulty] = {
                "total_tests": len(diff_df),
                "success_rate": (diff_df['error_occurred'] == False).mean() * 100,
                "avg_response_time": diff_df['response_time'].mean(),
                "avg_word_count": diff_df['word_count'].mean()
            }
        
        return summary
    
    def save_summary_report(self, filename: str = None) -> str:
        """Save summary report to JSON file."""
        summary = self.generate_summary_report()
        
        if not filename:
            filename = f"test_summary_{self.test_session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs("/Users/johan/HCMUS/datn/intellab_ai/src/testing/results", exist_ok=True)
        filepath = f"/Users/johan/HCMUS/datn/intellab_ai/src/testing/results/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Summary report saved to: {filepath}")
        return filepath

def main():
    """Main function to run batch testing."""
    tester = BatchTester()
    
    # Test with multiple models if available
    models = ["llama3.2"]  # Add more models as needed: ["llama3.2", "gpt-4o-mini"]
    
    # You can exclude certain chatbot types if needed
    # exclude_types = [ChatbotType.BG_TASK_AGENT]  # Add problematic agents here
    exclude_types = []
    
    try:
        # Run all tests
        results = tester.test_all_chatbots(models=models, exclude_types=exclude_types)
        
        # Save results
        csv_file = tester.save_results_to_csv()
        summary_file = tester.save_summary_report()
        
        # Print quick summary
        summary = tester.generate_summary_report()
        print(f"\n📈 Quick Summary:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Avg Response Time: {summary['average_response_time']:.2f}s")
        print(f"   Avg Word Count: {summary['avg_word_count']:.0f}")
        
        return csv_file, summary_file
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
        if tester.results:
            csv_file = tester.save_results_to_csv()
            summary_file = tester.save_summary_report()
            return csv_file, summary_file
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return None, None

if __name__ == "__main__":
    main()
