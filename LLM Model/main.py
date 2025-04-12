import qanalysis
import json
import csv
from typing import List, Dict

def process_batch(articles: List[Dict], start_idx: int, batch_size = int) -> List[Dict]:
    results = []
    for article in articles[start_idx:start_idx + batch_size]:
        summary = article.get('NewsArticle_Summary', '')
        caption = article.get('NewsArticle_Image_Caption', '')
        
        if summary and caption:
            classification = qanalysis.classify_relationship(summary, caption)
            results.append({
                'article_id': article.get('ID', 'Unknown'),
                'summary': summary,
                'caption': caption,
                'classification': classification
            })
            print(f"Processed article {article.get('ID', 'Unknown')}")
    return results



def main():

    #loading data
    with open('sumcap_dataset.json', 'r') as f:
        articles = json.load(f)
    
    articles = articles[:200]
    batch_size = 200
    all_results = []


    for i in range(0, len(articles), batch_size):
        print(f"\nProcessing batch {i//batch_size + 1}")
        batch_results = process_batch(articles, i, batch_size)
        all_results.extend(batch_results)

        # with open(f'analysis_results_batch_{i//batch_size + 1}.json', 'w') as f:
        #     json.dump(batch_results, f, indent=2)
    
    with open('final_analysis_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Article ID', 'Summary', 'Caption', 'Category', 'Explanation'])
        
        for result in all_results:
            writer.writerow([
                result['article_id'],
                result['summary'],
                result['caption'],
                result['classification'].get('category', ''),
                result['classification'].get('explanation', '')
            ])


if __name__ == "__main__":
    main()
    