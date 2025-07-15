"""
Advanced results analyzer for batch test results with visualizations and insights.
"""
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import numpy as np
import os
from datetime import datetime

class ResultsAnalyzer:
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.df = pd.read_csv(csv_file_path)
        self.evaluated_df = self.df[self.df['total_score'] > 0]  # Only evaluated responses
        
    def generate_performance_analysis(self) -> Dict:
        """Generate comprehensive performance analysis."""
        analysis = {
            "dataset_overview": self._get_dataset_overview(),
            "performance_metrics": self._get_performance_metrics(),
            "chatbot_comparison": self._get_chatbot_comparison(),
            "difficulty_analysis": self._get_difficulty_analysis(),
            "model_comparison": self._get_model_comparison(),
            "feature_analysis": self._get_feature_analysis(),
            "recommendations": self._get_recommendations()
        }
        return analysis
    
    def _get_dataset_overview(self) -> Dict:
        """Get overview of the dataset."""
        return {
            "total_responses": len(self.df),
            "evaluated_responses": len(self.evaluated_df),
            "evaluation_rate": (len(self.evaluated_df) / len(self.df)) * 100 if len(self.df) > 0 else 0,
            "unique_chatbots": self.df['chatbot_type'].nunique(),
            "unique_models": self.df['model'].nunique(),
            "success_rate": ((self.df['error_occurred'] == False).sum() / len(self.df)) * 100,
            "average_response_time": self.df['response_time'].mean(),
            "date_range": {
                "start": self.df['timestamp'].min(),
                "end": self.df['timestamp'].max()
            }
        }
    
    def _get_performance_metrics(self) -> Dict:
        """Get performance metrics for evaluated responses."""
        if len(self.evaluated_df) == 0:
            return {"error": "No evaluated responses available"}
        
        return {
            "average_scores": {
                "relevance": float(self.evaluated_df['relevance_score'].mean()),
                "accuracy": float(self.evaluated_df['accuracy_score'].mean()),
                "completeness": float(self.evaluated_df['completeness_score'].mean()),
                "clarity": float(self.evaluated_df['clarity_score'].mean()),
                "total": float(self.evaluated_df['total_score'].mean())
            },
            "score_distribution": {
                "excellent": len(self.evaluated_df[self.evaluated_df['total_score'] >= 8]),
                "good": len(self.evaluated_df[(self.evaluated_df['total_score'] >= 6) & (self.evaluated_df['total_score'] < 8)]),
                "fair": len(self.evaluated_df[(self.evaluated_df['total_score'] >= 4) & (self.evaluated_df['total_score'] < 6)]),
                "poor": len(self.evaluated_df[self.evaluated_df['total_score'] < 4])
            },
            "response_characteristics": {
                "avg_word_count": float(self.evaluated_df['word_count'].mean()),
                "avg_response_time": float(self.evaluated_df['response_time'].mean()),
                "code_examples_rate": (self.evaluated_df['has_code_examples'].sum() / len(self.evaluated_df)) * 100,
                "step_by_step_rate": (self.evaluated_df['has_step_by_step'].sum() / len(self.evaluated_df)) * 100
            }
        }
    
    def _get_chatbot_comparison(self) -> Dict:
        """Compare performance across different chatbot types."""
        comparison = {}
        
        for chatbot in self.evaluated_df['chatbot_type'].unique():
            chatbot_df = self.evaluated_df[self.evaluated_df['chatbot_type'] == chatbot]
            
            comparison[chatbot] = {
                "count": len(chatbot_df),
                "avg_total_score": float(chatbot_df['total_score'].mean()),
                "avg_relevance": float(chatbot_df['relevance_score'].mean()),
                "avg_accuracy": float(chatbot_df['accuracy_score'].mean()),
                "avg_completeness": float(chatbot_df['completeness_score'].mean()),
                "avg_clarity": float(chatbot_df['clarity_score'].mean()),
                "avg_response_time": float(chatbot_df['response_time'].mean()),
                "avg_word_count": float(chatbot_df['word_count'].mean()),
                "success_rate": ((chatbot_df['error_occurred'] == False).sum() / len(chatbot_df)) * 100,
                "features": {
                    "code_examples_rate": (chatbot_df['has_code_examples'].sum() / len(chatbot_df)) * 100,
                    "step_by_step_rate": (chatbot_df['has_step_by_step'].sum() / len(chatbot_df)) * 100,
                    "urls_rate": (chatbot_df['contains_urls'].sum() / len(chatbot_df)) * 100
                }
            }
        
        return comparison
    
    def _get_difficulty_analysis(self) -> Dict:
        """Analyze performance by question difficulty."""
        difficulty_analysis = {}
        
        for difficulty in self.evaluated_df['difficulty'].unique():
            diff_df = self.evaluated_df[self.evaluated_df['difficulty'] == difficulty]
            
            difficulty_analysis[difficulty] = {
                "count": len(diff_df),
                "avg_total_score": float(diff_df['total_score'].mean()),
                "avg_response_time": float(diff_df['response_time'].mean()),
                "avg_word_count": float(diff_df['word_count'].mean()),
                "success_rate": ((diff_df['error_occurred'] == False).sum() / len(diff_df)) * 100
            }
        
        return difficulty_analysis
    
    def _get_model_comparison(self) -> Dict:
        """Compare performance across different models."""
        model_comparison = {}
        
        for model in self.evaluated_df['model'].unique():
            model_df = self.evaluated_df[self.evaluated_df['model'] == model]
            
            model_comparison[model] = {
                "count": len(model_df),
                "avg_total_score": float(model_df['total_score'].mean()),
                "avg_response_time": float(model_df['response_time'].mean()),
                "avg_word_count": float(model_df['word_count'].mean()),
                "success_rate": ((model_df['error_occurred'] == False).sum() / len(model_df)) * 100
            }
        
        return model_comparison
    
    def _get_feature_analysis(self) -> Dict:
        """Analyze relationship between features and scores."""
        if len(self.evaluated_df) == 0:
            return {"error": "No evaluated responses available"}
        
        # Correlation analysis
        correlations = {}
        
        # Response characteristics vs scores
        characteristics = ['word_count', 'response_time']
        scores = ['relevance_score', 'accuracy_score', 'completeness_score', 'clarity_score', 'total_score']
        
        for char in characteristics:
            correlations[char] = {}
            for score in scores:
                corr = self.evaluated_df[char].corr(self.evaluated_df[score])
                correlations[char][score] = float(corr) if not pd.isna(corr) else 0.0
        
        # Binary features vs scores
        binary_features = ['has_code_examples', 'has_step_by_step', 'contains_urls', 'contains_markdown']
        feature_impact = {}
        
        for feature in binary_features:
            feature_true = self.evaluated_df[self.evaluated_df[feature] == True]['total_score'].mean()
            feature_false = self.evaluated_df[self.evaluated_df[feature] == False]['total_score'].mean()
            
            feature_impact[feature] = {
                "with_feature": float(feature_true) if not pd.isna(feature_true) else 0.0,
                "without_feature": float(feature_false) if not pd.isna(feature_false) else 0.0,
                "impact": float(feature_true - feature_false) if not pd.isna(feature_true) and not pd.isna(feature_false) else 0.0
            }
        
        return {
            "correlations": correlations,
            "feature_impact": feature_impact
        }
    
    def _get_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if len(self.evaluated_df) == 0:
            return ["No evaluated responses available for generating recommendations."]
        
        # Overall performance recommendations
        avg_score = self.evaluated_df['total_score'].mean()
        if avg_score < 6:
            recommendations.append("🔴 Overall performance is below acceptable level (< 6/10). Consider model fine-tuning or prompt optimization.")
        elif avg_score < 8:
            recommendations.append("🟡 Performance is good but has room for improvement. Focus on specific weak areas identified below.")
        else:
            recommendations.append("🟢 Excellent overall performance! Continue current approach.")
        
        # Chatbot-specific recommendations
        chatbot_scores = self.evaluated_df.groupby('chatbot_type')['total_score'].mean()
        worst_chatbot = chatbot_scores.idxmin()
        best_chatbot = chatbot_scores.idxmax()
        
        if chatbot_scores[worst_chatbot] < 6:
            recommendations.append(f"🎯 Priority: Improve {worst_chatbot} performance (avg: {chatbot_scores[worst_chatbot]:.1f}/10)")
        
        recommendations.append(f"✨ Best performing chatbot: {best_chatbot} (avg: {chatbot_scores[best_chatbot]:.1f}/10) - consider applying similar techniques to other chatbots")
        
        # Response time recommendations
        avg_response_time = self.evaluated_df['response_time'].mean()
        if avg_response_time > 10:
            recommendations.append("⏱️ Response times are high (>10s). Consider optimizing model inference or caching strategies.")
        
        # Feature recommendations
        feature_analysis = self._get_feature_analysis()
        if 'feature_impact' in feature_analysis:
            for feature, impact in feature_analysis['feature_impact'].items():
                if impact['impact'] > 1.0:
                    recommendations.append(f"📈 Encourage {feature}: responses with this feature score {impact['impact']:.1f} points higher on average")
        
        # Difficulty-based recommendations
        difficulty_scores = self.evaluated_df.groupby('difficulty')['total_score'].mean()
        if 'hard' in difficulty_scores and difficulty_scores['hard'] < 5:
            recommendations.append("🧠 Hard questions show poor performance. Consider specialized prompts or model tuning for complex queries.")
        
        return recommendations
    
    def create_visualizations(self, output_dir: str = None):
        """Create visualization charts for the analysis."""
        if not output_dir:
            output_dir = "/Users/johan/HCMUS/datn/intellab_ai/src/testing/results/visualizations"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Overall Score Distribution
        plt.figure(figsize=(10, 6))
        if len(self.evaluated_df) > 0:
            plt.hist(self.evaluated_df['total_score'], bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Total Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Total Scores')
            plt.axvline(self.evaluated_df['total_score'].mean(), color='red', linestyle='--', label='Mean')
            plt.legend()
            plt.savefig(f"{output_dir}/score_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Chatbot Performance Comparison
        if len(self.evaluated_df) > 0:
            plt.figure(figsize=(12, 8))
            chatbot_scores = self.evaluated_df.groupby('chatbot_type')[['relevance_score', 'accuracy_score', 'completeness_score', 'clarity_score']].mean()
            
            ax = chatbot_scores.plot(kind='bar', figsize=(12, 8))
            plt.title('Performance Comparison by Chatbot Type')
            plt.xlabel('Chatbot Type')
            plt.ylabel('Average Score')
            plt.xticks(rotation=45)
            plt.legend(title='Metrics')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/chatbot_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Response Time vs Score Scatter
        if len(self.evaluated_df) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.evaluated_df['response_time'], self.evaluated_df['total_score'], alpha=0.6)
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Total Score')
            plt.title('Response Time vs Total Score')
            
            # Add trend line
            z = np.polyfit(self.evaluated_df['response_time'], self.evaluated_df['total_score'], 1)
            p = np.poly1d(z)
            plt.plot(self.evaluated_df['response_time'], p(self.evaluated_df['response_time']), "r--", alpha=0.8)
            
            plt.savefig(f"{output_dir}/response_time_vs_score.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Difficulty Analysis
        if len(self.evaluated_df) > 0:
            plt.figure(figsize=(10, 6))
            difficulty_scores = self.evaluated_df.groupby('difficulty')['total_score'].mean()
            difficulty_scores.plot(kind='bar', color=['green', 'orange', 'red'])
            plt.title('Average Score by Difficulty Level')
            plt.xlabel('Difficulty')
            plt.ylabel('Average Total Score')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/difficulty_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Visualizations saved to: {output_dir}")
        return output_dir
    
    def save_analysis_report(self, output_file: str = None) -> str:
        """Save comprehensive analysis report."""
        analysis = self.generate_performance_analysis()
        
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"/Users/johan/HCMUS/datn/intellab_ai/src/testing/results/analysis_report_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"📋 Analysis report saved to: {output_file}")
        return output_file
    
    def print_summary(self):
        """Print a concise summary of the analysis."""
        analysis = self.generate_performance_analysis()
        
        print("\n" + "="*60)
        print("📊 BATCH TEST ANALYSIS SUMMARY")
        print("="*60)
        
        overview = analysis['dataset_overview']
        print(f"📈 Dataset Overview:")
        print(f"   Total Responses: {overview['total_responses']}")
        print(f"   Evaluated: {overview['evaluated_responses']} ({overview['evaluation_rate']:.1f}%)")
        print(f"   Success Rate: {overview['success_rate']:.1f}%")
        print(f"   Avg Response Time: {overview['average_response_time']:.2f}s")
        
        if 'performance_metrics' in analysis and 'error' not in analysis['performance_metrics']:
            metrics = analysis['performance_metrics']
            print(f"\n🎯 Performance Metrics:")
            print(f"   Average Total Score: {metrics['average_scores']['total']:.2f}/10")
            print(f"   Relevance: {metrics['average_scores']['relevance']:.2f}/10")
            print(f"   Accuracy: {metrics['average_scores']['accuracy']:.2f}/10")
            print(f"   Completeness: {metrics['average_scores']['completeness']:.2f}/10")
            print(f"   Clarity: {metrics['average_scores']['clarity']:.2f}/10")
            
            dist = metrics['score_distribution']
            print(f"\n📊 Score Distribution:")
            print(f"   Excellent (≥8): {dist['excellent']}")
            print(f"   Good (6-8): {dist['good']}")
            print(f"   Fair (4-6): {dist['fair']}")
            print(f"   Poor (<4): {dist['poor']}")
        
        print(f"\n💡 Key Recommendations:")
        for i, rec in enumerate(analysis['recommendations'][:3], 1):
            print(f"   {i}. {rec}")
        
        print("="*60)

def main():
    """Main function for results analysis."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python results_analyzer.py <csv_file_path>")
        print("Example: python results_analyzer.py results/batch_test_results_12345678_evaluated.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)
    
    analyzer = ResultsAnalyzer(csv_file)
    
    # Print summary
    analyzer.print_summary()
    
    # Generate full analysis report
    report_file = analyzer.save_analysis_report()
    
    # Create visualizations
    viz_dir = analyzer.create_visualizations()
    
    print(f"\n✅ Analysis complete!")
    print(f"📋 Report: {report_file}")
    print(f"📊 Visualizations: {viz_dir}")

if __name__ == "__main__":
    main()
