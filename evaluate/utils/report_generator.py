"""Report generation utilities."""

import csv
import json
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict


class ReportGenerator:
    """Generates evaluation reports in multiple formats."""
    
    @staticmethod
    def _format_score(score, precision=4):
        """Format score, handling NaN values."""
        if score is None or (isinstance(score, float) and math.isnan(score)):
            return "N/A"
        try:
            return f"{float(score):.{precision}f}"
        except (ValueError, TypeError):
            return "N/A"
    
    @staticmethod
    def save_json(results: List[Dict], output_file: Path):
        """Save results to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def save_csv(results: List[Dict], output_file: Path):
        """Save results to CSV file."""
        if not results:
            return
        
        fieldnames = [
            'case_id', 'diagnosis', 'status',
            'context_precision', 'context_recall', 'faithfulness', 'answer_relevancy',
            'timestamp', 'error'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                scores = result.get('scores', {})
                writer.writerow({
                    'case_id': result.get('case_id', ''),
                    'diagnosis': result.get('diagnosis', ''),
                    'status': result.get('status', ''),
                    'context_precision': scores.get('context_precision', ''),
                    'context_recall': scores.get('context_recall', ''),
                    'faithfulness': scores.get('faithfulness', ''),
                    'answer_relevancy': scores.get('answer_relevancy', ''),
                    'timestamp': result.get('timestamp', ''),
                    'error': result.get('error', '')
                })
    
    @staticmethod
    def generate_markdown(results: List[Dict], output_file: Path):
        """Generate markdown evaluation report."""
        completed = [r for r in results if r.get("status") == "completed"]
        failed = [r for r in results if r.get("status") != "completed"]
        
        # Calculate averages (excluding NaN values)
        avg_scores = {}
        if completed:
            for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
                scores = [r["scores"][metric] for r in completed 
                         if r["scores"].get(metric) is not None 
                         and not (isinstance(r["scores"].get(metric), float) and math.isnan(r["scores"].get(metric)))]
                avg_scores[metric] = sum(scores) / len(scores) if scores else float('nan')
        
        # Build report
        lines = [
            "# Medical_LMM Batch Evaluation Report\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## Summary\n",
            f"- **Total Cases:** {len(results)}",
            f"- **Completed:** {len(completed)}",
            f"- **Failed:** {len(failed)}\n",
            "## Average RAGAS Scores\n",
            "| Metric | Score |",
            "|--------|-------|",
            f"| Context Precision | {ReportGenerator._format_score(avg_scores.get('context_precision'))} |",
            f"| Context Recall | {ReportGenerator._format_score(avg_scores.get('context_recall'))} |",
            f"| Faithfulness | {ReportGenerator._format_score(avg_scores.get('faithfulness'))} |",
            f"| Answer Relevancy | {ReportGenerator._format_score(avg_scores.get('answer_relevancy'))} |\n",
            "## Individual Case Results\n"
        ]
        
        # Add individual results
        for result in completed:
            scores = result.get("scores", {})
            eval_errors = result.get("evaluation_errors", "")
            
            lines.extend([
                f"### {result['case_id']} - {result['diagnosis']}\n",
                "**Scores:**",
                f"- Context Precision: {ReportGenerator._format_score(scores.get('context_precision'))}",
                f"- Context Recall: {ReportGenerator._format_score(scores.get('context_recall'))}",
                f"- Faithfulness: {ReportGenerator._format_score(scores.get('faithfulness'))}",
                f"- Answer Relevancy: {ReportGenerator._format_score(scores.get('answer_relevancy'))}\n"
            ])
            
            # Add evaluation errors if any
            if eval_errors:
                lines.append(f"**⚠️ Evaluation Issues:** {eval_errors}\n")
            
            lines.append("---\n")
        
        # Add failed cases
        if failed:
            lines.append("\n## Failed Cases\n")
            for result in failed:
                lines.append(f"- **{result['case_id']}**: {result.get('status', 'unknown')} - "
                           f"{result.get('error', 'N/A')}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
