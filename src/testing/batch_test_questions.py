"""
Comprehensive test questions for all chatbot types with meaningful metrics.
"""
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class ChatbotType(Enum):
    GLOBAL_CHATBOT = "global_chatbot"
    PROBLEM_CHATBOT = "problem_chatbot"
    LESSON_CHATBOT = "lesson_chatbot"
    SIMPLE_CHATBOT = "chatbot"
    RESEARCH_ASSISTANT = "research_assistant"
    SUMMARIZE_ASSISTANT = "summarize_assistant"
    TITLE_GENERATOR = "title_generator"

@dataclass
class TestQuestion:
    id: str
    question: str
    expected_type: str  # e.g., "explanation", "code", "step_by_step", etc.
    difficulty: str  # "easy", "medium", "hard"
    category: str
    context: str = ""  # Additional context if needed

# Test questions for each chatbot type
TEST_QUESTIONS = {
    ChatbotType.GLOBAL_CHATBOT: [
        TestQuestion(
            id="gc_001",
            question="What programming languages should I learn for web development?",
            expected_type="recommendation",
            difficulty="easy",
            category="general_advice"
        ),
        TestQuestion(
            id="gc_002", 
            question="Explain the difference between frontend and backend development",
            expected_type="explanation",
            difficulty="easy",
            category="concepts"
        ),
        TestQuestion(
            id="gc_003",
            question="What are the best practices for code documentation?",
            expected_type="best_practices",
            difficulty="medium",
            category="software_engineering"
        ),
        TestQuestion(
            id="gc_004",
            question="How do I optimize the performance of a web application?",
            expected_type="step_by_step",
            difficulty="hard",
            category="optimization"
        ),
        TestQuestion(
            id="gc_005",
            question="What is the difference between SQL and NoSQL databases?",
            expected_type="comparison",
            difficulty="medium",
            category="databases"
        )
    ],
    
    ChatbotType.PROBLEM_CHATBOT: [
        TestQuestion(
            id="pc_001",
            question="How do I reverse a string in Python?",
            expected_type="code_solution",
            difficulty="easy",
            category="string_manipulation",
            context="Problem: Write a function to reverse a string. Problem_id: test_001 Question: How do I reverse a string in Python?"
        ),
        TestQuestion(
            id="pc_002",
            question="Implement a binary search algorithm",
            expected_type="algorithm_implementation",
            difficulty="medium", 
            category="searching",
            context="Problem: Implement binary search to find an element in a sorted array. Problem_id: test_002 Question: Implement a binary search algorithm"
        ),
        TestQuestion(
            id="pc_003",
            question="Find the longest common subsequence between two strings",
            expected_type="dynamic_programming",
            difficulty="hard",
            category="dynamic_programming",
            context="Problem: Given two strings, find the length of their longest common subsequence. Problem_id: test_003 Question: Find the longest common subsequence between two strings"
        ),
        TestQuestion(
            id="pc_004",
            question="Check if a binary tree is balanced",
            expected_type="tree_algorithm",
            difficulty="medium",
            category="tree_data_structure",
            context="Problem: Determine if a binary tree is height-balanced. Problem_id: test_004 Question: Check if a binary tree is balanced"
        ),
        TestQuestion(
            id="pc_005",
            question="Implement a stack using two queues",
            expected_type="data_structure_design",
            difficulty="medium",
            category="data_structures",
            context="Problem: Design a stack that supports push, pop, top, and empty operations using only two queues. Problem_id: test_005 Question: Implement a stack using two queues"
        )
    ],
    
    ChatbotType.LESSON_CHATBOT: [
        TestQuestion(
            id="lc_001",
            question="Explain how variables work in Python",
            expected_type="concept_explanation",
            difficulty="easy",
            category="python_basics",
            context="Lesson: Python variables are containers for storing data values. Python has no command for declaring a variable. Lesson_id: test_lesson_001 Question: Explain how variables work in Python"
        ),
        TestQuestion(
            id="lc_002",
            question="What are the key principles of object-oriented programming?",
            expected_type="concept_explanation",
            difficulty="medium",
            category="oop_concepts",
            context="Lesson: Object-oriented programming (OOP) is a programming paradigm based on the concept of objects. Lesson_id: test_lesson_002 Question: What are the key principles of object-oriented programming?"
        ),
        TestQuestion(
            id="lc_003",
            question="How does garbage collection work in Java?",
            expected_type="technical_explanation",
            difficulty="hard",
            category="memory_management",
            context="Lesson: Garbage collection in Java is the process of automatically freeing memory by deleting unreachable objects. Lesson_id: test_lesson_003 Question: How does garbage collection work in Java?"
        ),
        TestQuestion(
            id="lc_004",
            question="Explain the difference between GET and POST HTTP methods",
            expected_type="comparison",
            difficulty="easy",
            category="web_development",
            context="Lesson: HTTP methods define the action to be performed on a resource. GET and POST are the most common methods. Lesson_id: test_lesson_004 Question: Explain the difference between GET and POST HTTP methods"
        ),
        TestQuestion(
            id="lc_005",
            question="What is Big O notation and why is it important?",
            expected_type="concept_explanation",
            difficulty="medium",
            category="algorithms",
            context="Lesson: Big O notation describes the performance or complexity of an algorithm. Lesson_id: test_lesson_005 Question: What is Big O notation and why is it important?"
        )
    ],
    
    ChatbotType.RESEARCH_ASSISTANT: [
        TestQuestion(
            id="ra_001",
            question="What are the latest trends in artificial intelligence for 2024?",
            expected_type="research_summary",
            difficulty="medium",
            category="ai_trends"
        ),
        TestQuestion(
            id="ra_002",
            question="Calculate the compound interest for $10,000 invested at 5% annually for 10 years",
            expected_type="calculation",
            difficulty="easy",
            category="mathematics"
        ),
        TestQuestion(
            id="ra_003",
            question="Find information about the current job market for software engineers",
            expected_type="market_research",
            difficulty="medium",
            category="career"
        ),
        TestQuestion(
            id="ra_004",
            question="What are the environmental impacts of blockchain technology?",
            expected_type="research_analysis",
            difficulty="hard",
            category="technology_impact"
        ),
        TestQuestion(
            id="ra_005",
            question="Compare the performance of different machine learning frameworks",
            expected_type="comparison_research",
            difficulty="hard",
            category="ml_tools"
        )
    ],
    
    ChatbotType.SUMMARIZE_ASSISTANT: [
        TestQuestion(
            id="sa_001",
            question="Summarize Stack data structure course",
            expected_type="course_summary",
            difficulty="medium",
            category="data_structures",
            context="course name: Stack, id: 95713603-63d1-4b75-8a89-1acdc0977459, regenerate: false"
        ),
        TestQuestion(
            id="sa_002",
            question="Generate summary for Python Basics course",
            expected_type="course_summary",
            difficulty="easy",
            category="programming_basics",
            context="course name: Python Basics, id: test-course-002, regenerate: true"
        ),
        TestQuestion(
            id="sa_003",
            question="Create comprehensive summary for Algorithms course",
            expected_type="course_summary",
            difficulty="hard",
            category="algorithms",
            context="course name: Algorithms, id: test-course-003, regenerate: false"
        ),
        TestQuestion(
            id="sa_004",
            question="Summarize Web Development Fundamentals",
            expected_type="course_summary",
            difficulty="medium",
            category="web_development",
            context="course name: Web Development Fundamentals, id: test-course-004, regenerate: true"
        ),
        TestQuestion(
            id="sa_005",
            question="Generate Object-Oriented Programming course summary",
            expected_type="course_summary",
            difficulty="medium",
            category="oop",
            context="course name: Object-Oriented Programming, id: test-course-005, regenerate: false"
        )
    ],
    
    ChatbotType.TITLE_GENERATOR: [
        TestQuestion(
            id="tg_001",
            question="I want to learn how to build a REST API using Python Flask",
            expected_type="title",
            difficulty="easy",
            category="web_api"
        ),
        TestQuestion(
            id="tg_002", 
            question="Can you help me understand machine learning algorithms and their applications in real-world scenarios?",
            expected_type="title",
            difficulty="medium",
            category="machine_learning"
        ),
        TestQuestion(
            id="tg_003",
            question="I'm struggling with dynamic programming problems, especially the knapsack problem and its variations",
            expected_type="title",
            difficulty="hard",
            category="algorithms"
        ),
        TestQuestion(
            id="tg_004",
            question="What are the best practices for database design and normalization?",
            expected_type="title",
            difficulty="medium",
            category="database_design"
        ),
        TestQuestion(
            id="tg_005",
            question="How do I implement authentication and authorization in a web application?",
            expected_type="title",
            difficulty="medium",
            category="web_security"
        )
    ]
}

# Metrics for evaluation
@dataclass
class ResponseMetrics:
    response_time: float  # in seconds
    word_count: int
    has_code_examples: bool
    has_step_by_step: bool
    relevance_score: float  # 0-10 (manual evaluation)
    accuracy_score: float   # 0-10 (manual evaluation)
    completeness_score: float  # 0-10 (manual evaluation)
    clarity_score: float    # 0-10 (manual evaluation)
    total_score: float     # calculated average
    contains_urls: bool
    contains_markdown: bool
    error_occurred: bool
    error_message: str = ""

def get_test_questions_for_chatbot(chatbot_type: ChatbotType) -> List[TestQuestion]:
    """Get test questions for a specific chatbot type."""
    return TEST_QUESTIONS.get(chatbot_type, [])

def get_all_test_questions() -> Dict[ChatbotType, List[TestQuestion]]:
    """Get all test questions organized by chatbot type."""
    return TEST_QUESTIONS

def calculate_total_score(metrics: ResponseMetrics) -> float:
    """Calculate total score from individual metrics."""
    scores = [
        metrics.relevance_score,
        metrics.accuracy_score, 
        metrics.completeness_score,
        metrics.clarity_score
    ]
    return sum(scores) / len(scores) if scores else 0.0
