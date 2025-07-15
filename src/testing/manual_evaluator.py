"""
Manual evaluation helper for batch test results.
This script helps testers manually score responses and update the metrics.
"""
import pandas as pd
import json
import os
from typing import Dict, List, Optional
import sys

class ManualEvaluator:
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.df = pd.read_csv(csv_file_path)
        self.current_index = 0
        
    def display_response_for_evaluation(self, index: int) -> Dict:
        """Display a response for manual evaluation."""
        if index >= len(self.df):
            return None
        
        row = self.df.iloc[index]
        
        print("\n" + "="*80)
        print(f"📝 Evaluation {index + 1} of {len(self.df)}")
        print("="*80)
        print(f"🤖 Chatbot: {row['chatbot_type']}")
        print(f"🧠 Model: {row['model']}")
        print(f"🆔 Question ID: {row['question_id']}")
        print(f"📊 Difficulty: {row['difficulty']}")
        print(f"🏷️ Category: {row['category']}")
        print(f"📋 Expected Type: {row['expected_type']}")
        print(f"⏱️ Response Time: {row['response_time']:.2f}s")
        print(f"📝 Word Count: {row['word_count']}")
        
        print(f"\n❓ QUESTION:")
        print(f"   {row['question']}")
        
        if row['context'] and str(row['context']) != 'nan':
            print(f"\n📋 CONTEXT:")
            print(f"   {row['context']}")
        
        print(f"\n🤖 RESPONSE:")
        print("-" * 60)
        response = str(row['response'])
        if len(response) > 1000:
            print(response[:1000] + "\n... [TRUNCATED - showing first 1000 chars] ...")
        else:
            print(response)
        print("-" * 60)
        
        # Current scores
        print(f"\n📊 CURRENT SCORES:")
        print(f"   Relevance: {row['relevance_score']}/10")
        print(f"   Accuracy: {row['accuracy_score']}/10") 
        print(f"   Completeness: {row['completeness_score']}/10")
        print(f"   Clarity: {row['clarity_score']}/10")
        print(f"   Total: {row['total_score']}/10")
        
        # Auto-detected features
        print(f"\n🔍 AUTO-DETECTED FEATURES:")
        print(f"   Has Code Examples: {'✅' if row['has_code_examples'] else '❌'}")
        print(f"   Has Step-by-Step: {'✅' if row['has_step_by_step'] else '❌'}")
        print(f"   Contains URLs: {'✅' if row['contains_urls'] else '❌'}")
        print(f"   Contains Markdown: {'✅' if row['contains_markdown'] else '❌'}")
        print(f"   Error Occurred: {'❌' if row['error_occurred'] else '✅'}")
        
        if row['error_occurred'] and row['error_message']:
            print(f"   Error Message: {row['error_message']}")
        
        return {
            'index': index,
            'question_id': row['question_id'],
            'chatbot_type': row['chatbot_type'],
            'model': row['model']
        }
    
    def get_manual_scores(self) -> Optional[Dict[str, float]]:
        """Get manual scores from user input."""
        print(f"\n🎯 MANUAL EVALUATION:")
        print("Enter scores from 0-10 for each metric (or 'skip' to skip this response):")
        
        try:
            relevance = input("Relevance (how well does it answer the question?) [0-10]: ").strip()
            if relevance.lower() == 'skip':
                return None
            
            accuracy = input("Accuracy (is the information correct?) [0-10]: ").strip()
            if accuracy.lower() == 'skip':
                return None
                
            completeness = input("Completeness (is the answer thorough?) [0-10]: ").strip()
            if completeness.lower() == 'skip':
                return None
                
            clarity = input("Clarity (is it easy to understand?) [0-10]: ").strip()
            if clarity.lower() == 'skip':
                return None
            
            scores = {
                'relevance_score': float(relevance),
                'accuracy_score': float(accuracy),
                'completeness_score': float(completeness),
                'clarity_score': float(clarity)
            }
            
            # Calculate total score
            scores['total_score'] = sum(scores.values()) / len(scores)
            
            return scores
            
        except ValueError:
            print("❌ Invalid input. Please enter numbers between 0-10.")
            return self.get_manual_scores()
    
    def update_scores(self, index: int, scores: Dict[str, float]):
        """Update scores in the dataframe."""
        for metric, score in scores.items():
            self.df.at[index, metric] = score
    
    def save_updated_results(self, output_file: Optional[str] = None):
        """Save updated results to CSV."""
        if not output_file:
            base_name = os.path.splitext(self.csv_file_path)[0]
            output_file = f"{base_name}_evaluated.csv"
        
        self.df.to_csv(output_file, index=False)
        print(f"💾 Updated results saved to: {output_file}")
        return output_file
    
    def interactive_evaluation(self):
        """Run interactive evaluation session."""
        print("🎯 INTERACTIVE BATCH TEST EVALUATION")
        print("="*50)
        print("Instructions:")
        print("- Review each response and provide scores 0-10")
        print("- Type 'skip' to skip a response")
        print("- Type 'quit' to exit and save progress")
        print("- Type 'back' to go to previous response")
        print("- Type 'jump <number>' to jump to specific response")
        
        current_index = 0
        
        while current_index < len(self.df):
            try:
                # Display current response
                response_info = self.display_response_for_evaluation(current_index)
                if not response_info:
                    break
                
                # Get user input
                user_input = input(f"\nEnter scores or command [current: {current_index + 1}/{len(self.df)}]: ").strip().lower()
                
                if user_input == 'quit':
                    break
                elif user_input == 'back' and current_index > 0:
                    current_index -= 1
                    continue
                elif user_input.startswith('jump '):
                    try:
                        jump_to = int(user_input.split()[1]) - 1
                        if 0 <= jump_to < len(self.df):
                            current_index = jump_to
                            continue
                        else:
                            print(f"❌ Invalid jump target. Must be between 1 and {len(self.df)}")
                            continue
                    except (IndexError, ValueError):
                        print("❌ Invalid jump command. Use 'jump <number>'")
                        continue
                elif user_input == 'skip':
                    current_index += 1
                    continue
                else:
                    # Get manual scores
                    scores = self.get_manual_scores()
                    if scores:
                        self.update_scores(current_index, scores)
                        print("✅ Scores updated!")
                    
                    current_index += 1
                    
            except KeyboardInterrupt:
                print("\n⚠️ Evaluation interrupted by user")
                break
        
        # Save results
        output_file = self.save_updated_results()
        self.generate_evaluation_summary(output_file)
    
    def generate_evaluation_summary(self, csv_file: str):
        """Generate summary of manual evaluations."""
        df = pd.read_csv(csv_file)
        
        # Filter evaluated responses (those with scores > 0)
        evaluated_df = df[df['total_score'] > 0]
        
        if len(evaluated_df) == 0:
            print("📊 No responses have been manually evaluated yet.")
            return
        
        summary = {
            "evaluation_summary": {
                "total_responses": len(df),
                "evaluated_responses": len(evaluated_df),
                "evaluation_percentage": (len(evaluated_df) / len(df)) * 100,
                "average_scores": {
                    "relevance": evaluated_df['relevance_score'].mean(),
                    "accuracy": evaluated_df['accuracy_score'].mean(),
                    "completeness": evaluated_df['completeness_score'].mean(),
                    "clarity": evaluated_df['clarity_score'].mean(),
                    "total": evaluated_df['total_score'].mean()
                },
                "by_chatbot": {},
                "by_difficulty": {},
                "top_performers": [],
                "low_performers": []
            }
        }
        
        # By chatbot type
        for chatbot in evaluated_df['chatbot_type'].unique():
            chatbot_df = evaluated_df[evaluated_df['chatbot_type'] == chatbot]
            summary["evaluation_summary"]["by_chatbot"][chatbot] = {
                "count": len(chatbot_df),
                "avg_total_score": chatbot_df['total_score'].mean(),
                "avg_relevance": chatbot_df['relevance_score'].mean(),
                "avg_accuracy": chatbot_df['accuracy_score'].mean()
            }
        
        # By difficulty
        for difficulty in evaluated_df['difficulty'].unique():
            diff_df = evaluated_df[evaluated_df['difficulty'] == difficulty]
            summary["evaluation_summary"]["by_difficulty"][difficulty] = {
                "count": len(diff_df),
                "avg_total_score": diff_df['total_score'].mean()
            }
        
        # Top and low performers
        top_5 = evaluated_df.nlargest(5, 'total_score')[['question_id', 'chatbot_type', 'model', 'total_score']]
        low_5 = evaluated_df.nsmallest(5, 'total_score')[['question_id', 'chatbot_type', 'model', 'total_score']]
        
        summary["evaluation_summary"]["top_performers"] = top_5.to_dict('records')
        summary["evaluation_summary"]["low_performers"] = low_5.to_dict('records')
        
        # Save summary
        summary_file = csv_file.replace('.csv', '_evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"   Evaluated: {len(evaluated_df)}/{len(df)} responses ({(len(evaluated_df) / len(df)) * 100:.1f}%)")
        print(f"   Average Total Score: {evaluated_df['total_score'].mean():.2f}/10")
        print(f"   Summary saved to: {summary_file}")

def main():
    """Main function for manual evaluation."""
    if len(sys.argv) != 2:
        print("Usage: python manual_evaluator.py <csv_file_path>")
        print("Example: python manual_evaluator.py results/batch_test_results_12345678_20241218_143000.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)
    
    evaluator = ManualEvaluator(csv_file)
    evaluator.interactive_evaluation()

if __name__ == "__main__":
    main()
