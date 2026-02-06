"""
Script to create a new dataset from Salesforce/APIGen-MT-5k
For each human turn in a conversation, creates an entry with:
- conversation_history: conversation up to that human turn
- domain_rules: copied from original
- required_rules: rules needed to answer (via GPT-4)
- reasoning: CoT reasoning for rule selection (via GPT-4)
"""


import os
import json
import re
from datasets import load_dataset, Dataset
from openai import OpenAI

# Initialize OpenAI client - set your API key
client = OpenAI(api_key="sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

OUTPUT_FILE = "rules_dataset.json"

# System prompt for GPT-4
RULE_ANALYZER_PROMPT = """You are an expert policy analyzer for conversational AI agents.

Given:
1. A conversation history ending with a user request
2. The assistant's response to that request
3. Domain rules/policies the assistant must follow

Your task: Identify which specific rules from the domain policy are REQUIRED to properly handle the user's immediate request.

Instructions:
- Read the user's latest request carefully
- Look at how the assistant responded
- Identify ONLY the rules that directly apply to answering this specific request
- Be precise - don't include rules that aren't relevant to this turn

You MUST respond with ONLY valid JSON in this exact format (no other text):
{
    "required_rules": ["rule 1 text", "rule 2 text"],
    "reasoning": "Step-by-step explanation of why each rule is needed..."
}

Be thorough in your reasoning - explain the connection between the user request and each rule."""


def extract_json(text: str) -> dict:
    """Extract JSON from GPT response."""
    try:
        return json.loads(text)
    except:
        pass
    
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    json_match = re.search(r'\{[\s\S]*"required_rules"[\s\S]*"reasoning"[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    return {"required_rules": [], "reasoning": "Failed to parse response"}


def save_to_json(entries: list, filename: str):
    """Save entries to JSON file (overwrites)."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def append_entry_to_json(entry: dict, filename: str):
    """Append single entry to JSON file dynamically."""
    # Read existing data
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                data = []
    else:
        data = []
    
    # Append new entry
    data.append(entry)
    
    # Write back
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return len(data)


def get_required_rules(conversation_history: list, assistant_response: str, domain_rules: str) -> dict:
    """Use GPT-4 to identify which rules are needed."""
    
    conv_text = ""
    for turn in conversation_history:
        role = turn["from"]
        value = turn["value"]
        if role == "human":
            conv_text += f"USER: {value}\n\n"
        elif role == "gpt":
            conv_text += f"ASSISTANT: {value}\n\n"
        elif role == "function_call":
            conv_text += f"[Function Call]: {value}\n\n"
        elif role == "observation":
            conv_text += f"[Observation]: {value}\n\n"
    
    user_message = f"""## Conversation History (ending with user request):
{conv_text}

## Assistant's Response:
{assistant_response}

## Domain Rules/Policy:
{domain_rules}

Return ONLY valid JSON with required_rules and reasoning."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": RULE_ANALYZER_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2
        )
        return extract_json(response.choices[0].message.content)
    
    except Exception as e:
        print(f"    Error: {e}")
        return {"required_rules": [], "reasoning": f"Error: {str(e)}"}


def process_row_and_save(row: dict, row_idx: int, output_file: str) -> int:
    """Process one row and save each entry dynamically."""
    
    conversations = row["conversations"]
    domain_rules = row["system"]
    entries_created = 0
    
    for i, turn in enumerate(conversations):
        if turn["from"] == "human":
            conv_history = conversations[:i+1]
            
            # Find next response
            next_response = ""
            for j in range(i+1, len(conversations)):
                if conversations[j]["from"] in ["gpt", "function_call"]:
                    next_response = conversations[j]["value"]
                    break
            
            if not next_response:
                continue
            
            # Get analysis from GPT-4
            print(f"    Analyzing turn {i+1}...")
            analysis = get_required_rules(conv_history, next_response, domain_rules)
            
            # Create entry
            entry = {
                "source_row": row_idx,
                "turn_index": i,
                "conversation_history": conv_history,
                "domain_rules": domain_rules,
                "required_rules": analysis.get("required_rules", []),
                "reasoning": analysis.get("reasoning", "")
            }
            
            # Save immediately to JSON
            total = append_entry_to_json(entry, output_file)
            entries_created += 1
            print(f"    ✓ Entry saved! (Total: {total})")
    
    return entries_created


def main():
    print("=" * 50)
    print("APIGen-MT-5k Rule Extractor")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 50)
    
    # Clear output file
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Cleared existing {OUTPUT_FILE}")
    
    # Initialize empty JSON array
    save_to_json([], OUTPUT_FILE)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("Salesforce/APIGen-MT-5k", split="train")
    print(f"Loaded {len(dataset)} rows\n")
    
    total_entries = 0
    
    for idx, row in enumerate(dataset):
        print(f"\n[Row {idx + 1}/{len(dataset)}]")
        entries = process_row_and_save(row, idx, OUTPUT_FILE)
        total_entries += entries
        print(f"  Created {entries} entries from this row")
        print(f"  Running total: {total_entries} entries")
    
    print("\n" + "=" * 50)
    print(f"DONE! Total entries: {total_entries}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":

    main()
