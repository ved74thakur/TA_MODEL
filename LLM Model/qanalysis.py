

from dotenv import load_dotenv
import json
import csv

from google import genai
import os


load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)



def classify_relationship(summary_text, caption_text):
    prompt = f"""You are a classification engine that assigns a relationship type between an article summary and its associated image caption. 
    1. The relationship category and sub-category
    2. A brief explanation of why this classification was chosen
    Use the following rules:

1. Equivalence
   - **Literal Equivalence**
     - *Token-Token (Exact Match)*
       - Rule: Verify if both texts refer to an entity or concept with identical, unambiguous tokens.
       - Example: "John Doe" in the summary and a headshot of John Doe in the caption.
     - *Type-Token (Class vs. Instance)*
       - Rule: Identify if one text namevs a general category (e.g., "chef") while the other shows a specific instance (e.g., an image of a chef at work).
       - Example: The text refers to safety equipment, and the caption shows a helmet.
   - **Figurative Equivalence**
     - *Metonymy*
       - Rule: Look for cases where both texts point to entities from the same domain or context even if not identical.
       - Example: The text states "in Athens" and the caption shows a landmark typically associated with Athens.
     - *Metaphor*
       - Rule: Detect if the texts use elements that transfer qualities from one domain to another, where similarity is drawn rather than identity.
       - Example: The text uses "serene" non-literally and the caption shows an abstract representation (e.g., flowing water) that conveys calm.

2. Complementarity
   - **Essential Complementarity**
     - *Essential Exophora*
       - Rule: If the summary uses a deictic element (like "this" or "that") and the caption supplies the missing referent.
       - Example: "This is beautiful" paired with a caption resolving which landmark is referenced.
     - *Essential Agent–Object*
       - Rule: The action described in one text requires the caption to identify the subject or object.
       - Example: "Hold this" with a caption clarifying what "this" is.
     - *Defining Apposition*
       - Rule: When one text identifies an entity uniquely (e.g., "President of Greece") and the caption distinguishes the specific individual.
   - **Non-Essential Complementarity**
     - *Non-Essential Exophora*
       - Rule: One text clarifies or enriches the message of the other but is not necessary to understand the core message.
     - *Non-Essential Agent–Object*
       - Rule: The caption adds supplementary context that is not needed for the primary message.
     - *Adjunct*
       - Rule: The caption functions adverbially (e.g., providing details about place, time, or manner).
     - *Non-Defining Apposition*
       - Rule: The caption provides extra identifying details that enrich the context.
       
3. Independence
   - *Contradiction*
     - Rule: The summary and caption express semantically opposite or incompatible messages.
   - *Symbiosis*
     - Rule: The texts are largely independent but grouped together thematically to create a broader effect.
   - *Meta-Information*
     - Rule: One text reveals extra, contextual information (e.g., production context) about the other.

Your output must be in JSON format with three keys: 
- "category": The main category

- "explanation": A brief explanation (2-3 sentences) of why this classification was chosen


- Equivalence:
  - Literal Equivalence:
    - "Token-Token"
    - "Type-Token"
  - Figurative Equivalence:
    - "Metonymy"
    - "Metaphor"

- Complementarity:
  - Essential Complementarity:
    - "Essential Exophora"
    - "Essential Agent–Object"
    - "Defining Apposition"
  - Non-Essential Complementarity:
    - "Non-Essential Exophora"
    - "Non-Essential Agent–Object"
    - "Adjunct"
    - "Non-Defining Apposition"

- Independence:
  - "Contradiction"
  - "Symbiosis"
  - "Meta-Information"

Now, based on these rules, analyze the following texts:

Summary: "{summary_text}"
Caption: "{caption_text}"

IMPORTANT: Return your answer as valid JSON only (without code fences or markdown).
"""
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
        
    )
    try:
        result  = response.text.strip()
        if '```json' in result:
            result = result.split('```json')[1].split('```')[0].strip()
        elif '```' in result:
            result = result.split('```')[1].strip()

        classification = json.loads(result)

        if not all(key in classification for key in ['category', 'sub_category', 'explanation']):
            raise ValueError("Missing required fields in classification response")
    
    except json.JSONDecodeError:
        classification = {"error": "Failed to parse the JSON output. ", "raw_output": result, "category": "Error", "sub_category": "Parse Error", "explanation": "Failed to process the classification"}
    except Exception as e:
        classification = {"error": f"Error: {str(e)}", "raw_output": response.text if hasattr(response, 'text') else str(response), "category": "Error", "sub_category": "Processing Error", "explanation": f"An error occurred: {str(e)}"}

    return classification

    

if __name__ == "__main__":
    summary = """The article discusses remarks by a Prime Minister’s adviser suggesting that herd immunity could be the key to ending the coronavirus lockdown.  
Herd immunity, in this context, involves allowing enough of the population to build resistance to Covid-19 to curb its spread.  
Critics worry this approach could overwhelm healthcare systems and put vulnerable groups at risk.  
Nonetheless, it has sparked debate on whether and how such a strategy could guide the easing of restrictions."""
    caption = "The government’s chief epidemic modeller says an exit plan is needed from the lockdown (Picture: Getty, AFP)"

    classification_result = classify_relationship(summary, caption)

    csv_file = "classification_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Summary", "Caption", "Category", "Explanation"])
        writer.writerow([
            summary, 
            caption, 
            classification_result.get("category", ""), 
            classification_result.get("explanation", "")
        ])

    print(f"Classification result saved in {csv_file}")
    print("")
    

