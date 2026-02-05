#!/usr/bin/env python3
"""
generate_policy_qa.py

Pipeline that:
- Parses a policy markdown file into canonical rule chunks (with stable IDs).
- Uses a teacher LLM to generate:
    * canonical question for each rule
    * canonical verbatim answer: exact span from the rule (enforced)
    * 1-2 sentence rationale (flexible)
    * user-facing phrasing
- Generates paraphrases for each scenario and plausible negatives.
- Filters and saves JSONL with fields matching the schema discussed in the design.

Usage:
    export OPENAI_API_KEY="sk-..."
    python generate_policy_qa.py --policy /mnt/data/wiki.md --out seed.jsonl

Requirements:
    pip install openai tqdm regex
"""

import re
import json
import time
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm

# ---------- LLM call wrapper (litellm) ----------
# Uses litellm for provider-agnostic LLM calls.
try:
    from litellm import completion
except Exception:
    completion = None

def call_llm(prompt: str,
             model: str = "gpt-4o",  # change to your available model
             provider: str = "openai",  # litellm provider
             temperature: float = 0.0,
             max_tokens: int = 512,
             timeout: int = 60,
             n_retries: int = 3) -> str:
    """
    Call LLM using litellm for provider-agnostic completions.
    """
    if completion is None:
        raise RuntimeError("litellm package not installed or failed to import")

    for attempt in range(n_retries):
        try:
            print(f"LLM call attempt {attempt+1}...")
            res = completion(
                model=model,
                custom_llm_provider=provider,
                messages=[{"role": "system", "content": "You are a generator of canonical policy questions and answers."},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = res.choices[0].message.content
            print("LLM response received: ", text[:200].replace("\n", " "))
            return text
        except Exception as e:
            if attempt + 1 == n_retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")

# ---------- Policy parsing ----------
def split_policy_into_chunks(policy_text: str) -> List[Dict]:
    """
    Split markdown policy into canonical chunks. Strategy:
      - Use headings (lines that start with #) as chunk boundaries if present.
      - Fallback: split by double newline paragraphs of some length.
    Returns list of {"chunk_id": "...", "text": "<RULE>...</RULE>"}
    """
    chunks = []
    # Normalize line endings
    text = policy_text.replace("\r\n", "\n")
    # Heuristic: split on H1/H2/H3 headings
    headings = list(re.finditer(r"(?m)^(#{1,3})\s*(.+)$", text))
    if headings:
        for i, h in enumerate(headings):
            start = h.end()
            end = headings[i+1].start() if i+1 < len(headings) else len(text)
            title = h.group(2).strip()
            body = text[start:end].strip()
            if not body:
                continue
            # canonical id: slugify title or use index
            slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
            chunk_id = f"{slug or 'rule'}_{i+1}"
            rule_text = body
            canonical = f"<RULE id=\"{chunk_id}\">\n{rule_text}\n</RULE>"
            chunks.append({"chunk_id": chunk_id, "text": canonical})
    else:
        # fallback: split by paragraphs longer than 40 chars
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
        for i, p in enumerate(paras):
            chunk_id = f"clause_{i+1}"
            canonical = f"<RULE id=\"{chunk_id}\">\n{p}\n</RULE>"
            chunks.append({"chunk_id": chunk_id, "text": canonical})
    return chunks

# ---------- Prompt templates ----------
QA_PROMPT_TEMPLATE = """
You are given a policy chunk delimited by <RULE id="...">...</RULE>.
Your task: produce a canonical question a support agent might receive that *this rule answers*, a canonical exact answer that must be **verbatim** the rule text (i.e., the output must include the exact rule span from the provided chunk), a one-sentence rationale that cites the rule id, and a short user-facing phrasing the agent should use. Output JSON only with fields:
{{
  "question": "...",
  "answer_span": "<RULE id=\\"{chunk_id}\\">...the rule text...</RULE>",
  "rationale": "...one sentence referencing {chunk_id}...",
  "user_facing": "..."
}}
If the chunk is not actionable (e.g., header or meta text), return {{ "skip": true }}.
Make the question a natural user utterance (one-liner). The "answer_span" must reproduce the canonical rule verbatim as provided.
"""

MULTI_QA_PROMPT_TEMPLATE = """
You are given a policy chunk delimited by <RULE id="...">...</RULE>.
Your task: produce {num_questions} DIVERSE canonical questions that a support agent might receive that *this rule answers*.

IMPORTANT - Each question MUST be SIGNIFICANTLY DIFFERENT from the others:
- Use different phrasings and sentence structures
- Vary the user's perspective (asking about eligibility vs. process vs. exceptions vs. edge cases)
- Include different levels of specificity (general inquiry vs. specific scenario)
- Vary the tone (formal inquiry vs. casual question vs. confused user vs. frustrated complaint)
- Ask about different aspects of the same rule when possible

For each question, provide:
- question: a natural user utterance (one-liner, unique from other questions)
- answer_span: the exact verbatim rule text as provided
- rationale: one sentence referencing {chunk_id}
- user_facing: a short agent response

Output a JSON array of {num_questions} objects:
[
  {{"question": "...", "answer_span": "<RULE id=\\"{chunk_id}\\">...</RULE>", "rationale": "...", "user_facing": "..."}},
  ...
]

If the chunk is not actionable (e.g., header or meta text), return {{ "skip": true }}.
The "answer_span" must reproduce the canonical rule verbatim as provided.
"""

PARAPHRASE_PROMPT_TEMPLATE = """
Paraphrase the following user question into {n} natural variants (short, diverse), preserving meaning.
Input question:
\"\"\"{question}\"\"\"
Output: JSON array of strings.
"""

NEGATIVE_PROMPT_TEMPLATE = """
Given the canonical rule text:
\"\"\"{rule_text}\"\"\"
Generate {n} plausible-but-wrong single-sentence answers that a mistaken agent might say (these should be tempting but contradict the rule). Output JSON array of strings.
"""

# ---------- Helpers: parsing LLM JSON outputs ----------
def parse_json_from_llm(text: str) -> Dict:
    """
    Try to parse a JSON object from free text. Strips markdown fences and trailing text.
    """
    # Trim leading/trailing whitespace
    s = text.strip()
    # Remove ```json fences if present
    s = re.sub(r"^```(json)?\n", "", s)
    s = re.sub(r"\n```$", "", s)
    # Try to extract first { ... } block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # try to fix single quotes -> double quotes
            candidate2 = candidate.replace("'", '"')
            try:
                return json.loads(candidate2)
            except json.JSONDecodeError:
                raise
    # As fallback, try to parse as array
    m2 = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if m2:
        try:
            return json.loads(m2.group(0))
        except json.JSONDecodeError:
            pass
    # If nothing works, raise
    raise ValueError("Failed to parse JSON from LLM output:\n" + text[:1000])

# ---------- Main pipeline functions ----------
def generate_canonical_qa_for_chunk(chunk: Dict, call_llm_fn, model_kwargs: Dict) -> Dict:
    """
    For a given policy chunk, ask teacher LLM to produce canonical question, answer_span (must equal chunk.text), rationale, and user_facing.
    Returns dict with keys question, answer_span, rationale, user_facing, skip (optional).
    """
    # Inject chunk id and chunk text into prompt
    chunk_id = chunk["chunk_id"]
    prompt = QA_PROMPT_TEMPLATE.format(chunk_id=chunk_id)
    # Append the chunk text to the prompt to ensure the teacher sees exact canonical text
    full_prompt = f"{chunk['text']}\n\n{prompt}"
    raw = call_llm_fn(full_prompt, **model_kwargs)
    try:
        parsed = parse_json_from_llm(raw)
    except Exception as e:
        # If parsing fails, return raw text for debugging
        return {"_llm_raw": raw, "parse_error": str(e)}
    # Basic sanity checks: answer_span must contain the chunk canonical text
    if parsed.get("skip"):
        return {"skip": True}
    ans = parsed.get("answer_span", "")
    # Normalize whitespace
    if chunk["text"].strip() not in ans:
        # enforce answer_span to be the canonical chunk if LLM didn't include it
        parsed["answer_span"] = chunk["text"]
    return parsed

def normalize_question(q: str) -> str:
    """Normalize a question for deduplication comparison."""
    return re.sub(r'[^\w\s]', '', q.lower()).strip()

def deduplicate_qa_list(qa_list: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
    """
    Remove duplicate or near-duplicate questions from the list.
    Uses simple word overlap ratio for similarity detection.
    """
    seen_normalized = []
    unique_qa = []

    for qa in qa_list:
        if not qa.get("question"):
            continue

        norm_q = normalize_question(qa["question"])
        words_q = set(norm_q.split())

        is_duplicate = False
        for seen_norm in seen_normalized:
            seen_words = set(seen_norm.split())
            if not words_q or not seen_words:
                continue
            # Jaccard similarity
            intersection = len(words_q & seen_words)
            union = len(words_q | seen_words)
            similarity = intersection / union if union > 0 else 0

            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            seen_normalized.append(norm_q)
            unique_qa.append(qa)

    return unique_qa

def generate_multiple_qa_for_chunk(chunk: Dict, call_llm_fn, model_kwargs: Dict, num_questions: int = 3) -> List[Dict]:
    """
    Generate multiple diverse Q&A pairs for a single chunk.
    Returns a list of dicts, each with keys: question, answer_span, rationale, user_facing.
    Deduplicates to ensure questions are distinct.
    """
    chunk_id = chunk["chunk_id"]

    # Use higher temperature for diversity when generating multiple questions
    diverse_kwargs = {**model_kwargs}
    diverse_kwargs["temperature"] = max(model_kwargs.get("temperature", 0.0), 0.7)
    diverse_kwargs["max_tokens"] = max(model_kwargs.get("max_tokens", 512), 2048)  # Need more tokens for multiple Q&As

    prompt = MULTI_QA_PROMPT_TEMPLATE.format(chunk_id=chunk_id, num_questions=num_questions)
    full_prompt = f"{chunk['text']}\n\n{prompt}"

    raw = call_llm_fn(full_prompt, **diverse_kwargs)

    try:
        parsed = parse_json_from_llm(raw)
    except Exception as e:
        print(f"Failed to parse multi-QA response: {e}")
        return []

    # Handle skip response
    if isinstance(parsed, dict) and parsed.get("skip"):
        return []

    # Ensure we have a list
    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        return []

    # Validate and fix answer_span for each QA
    valid_qa = []
    for qa in parsed:
        if not isinstance(qa, dict):
            continue
        if not qa.get("question"):
            continue
        # Enforce answer_span
        ans = qa.get("answer_span", "")
        if chunk["text"].strip() not in ans:
            qa["answer_span"] = chunk["text"]
        valid_qa.append(qa)

    # Deduplicate questions
    unique_qa = deduplicate_qa_list(valid_qa)

    print(f"Generated {len(valid_qa)} questions, {len(unique_qa)} unique after deduplication")

    return unique_qa

def generate_paraphrases(question: str, call_llm_fn, n: int = 6, model_kwargs: Dict = None) -> List[str]:
    prompt = PARAPHRASE_PROMPT_TEMPLATE.format(question=question, n=n)
    raw = call_llm_fn(prompt, **(model_kwargs or {}))
    arr = parse_json_from_llm(raw)
    if not isinstance(arr, list):
        raise ValueError("Paraphrase LLM did not return a list")
    return arr

def generate_negatives(rule_text: str, call_llm_fn, n: int = 4, model_kwargs: Dict = None) -> List[str]:
    prompt = NEGATIVE_PROMPT_TEMPLATE.format(rule_text=rule_text, n=n)
    raw = call_llm_fn(prompt, **(model_kwargs or {}))
    arr = parse_json_from_llm(raw)
    return arr

def basic_filter_example(example: Dict) -> bool:
    """
    Simple checks:
      - required fields present
      - answer_span contains the chunk canonical text (we enforce this earlier)
    Returns True if example passes filters.
    """
    if example.get("skip"):
        return False
    if not example.get("question") or not example.get("answer_span"):
        return False
    return True

# ---------- Orchestration ----------
def build_dataset_from_policy(policy_path: str,
                              out_path: str,
                              call_llm_fn=call_llm,
                              model_kwargs: Dict = None,
                              paraphrase_count: int = 6,
                              negatives_per_rule: int = 3,
                              max_chunks: int = None,
                              questions_per_chunk: int = 1):
    """
    Main orchestration: parse policy, loop chunks, call teacher LLM to produce canonical Q/A,
    augment with paraphrases and negatives, and write JSONL to out_path.

    Args:
        questions_per_chunk: Number of diverse questions to generate per policy chunk.
    """
    model_kwargs = model_kwargs or {"model": "gpt-4o-mini", "temperature": 0.0, "max_tokens": 512}
    with open(policy_path, "r", encoding="utf-8") as f:
        policy_text = f.read()

    print("Splitting policy into chunks...")
    chunks = split_policy_into_chunks(policy_text)
    if max_chunks:
        chunks = chunks[:max_chunks]
    print(f"Found {len(chunks)} chunks")
    print(f"Generating {questions_per_chunk} question(s) per chunk...")

    results = []
    for chunk in tqdm(chunks):
        # 1) Generate Q&A(s) - single or multiple based on questions_per_chunk
        if questions_per_chunk == 1:
            # Use original single-question function
            try:
                canon = generate_canonical_qa_for_chunk(chunk, call_llm_fn, model_kwargs)
            except Exception as e:
                print(f"LLM error for chunk {chunk['chunk_id']}: {e}")
                continue

            if not basic_filter_example(canon):
                print(f"Skipping chunk {chunk['chunk_id']} due to filter")
                continue

            qa_list = [canon]
        else:
            # Use multi-question function
            try:
                qa_list = generate_multiple_qa_for_chunk(chunk, call_llm_fn, model_kwargs, num_questions=questions_per_chunk)
            except Exception as e:
                print(f"LLM error for chunk {chunk['chunk_id']}: {e}")
                continue

            if not qa_list:
                print(f"Skipping chunk {chunk['chunk_id']} - no valid questions generated")
                continue

        # 2) Generate negatives once per chunk (shared across all questions for this chunk)
        try:
            negs = generate_negatives(chunk["text"], call_llm_fn, n=negatives_per_rule, model_kwargs=model_kwargs)
        except Exception as e:
            print(f"Negative gen failed for {chunk['chunk_id']}: {e}")
            negs = []

        # 3) Build example objects for each Q&A
        for q_idx, canon in enumerate(qa_list):
            # Generate paraphrases for each question
            try:
                paras = generate_paraphrases(canon["question"], call_llm_fn, n=paraphrase_count, model_kwargs=model_kwargs)
            except Exception as e:
                print(f"Paraphrase generation failed for {chunk['chunk_id']}.q{q_idx+1:03d}: {e}")
                paras = []

            example = {
                "id": f"{chunk['chunk_id']}.q{q_idx+1:03d}",
                "policy_chunks": [chunk],
                "scenario": canon["question"],
                "gold_rule_spans": [chunk["chunk_id"]],
                "gold_rule_text": chunk["text"],
                "gold_decision_rationale": canon.get("rationale", "") + " " + canon.get("user_facing", ""),
                "paraphrases": paras,
                "negatives": [{"bad_span": n.strip(), "severity": "high"} for n in negs],
                "severity": "high"
            }
            results.append(example)

    # Save JSONL (append mode to preserve previous data)
    with open(out_path, "a", encoding="utf-8") as out_f:
        for ex in results:
            out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} examples to {out_path}")

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="Path to policy markdown")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit chunks for testing")
    parser.add_argument("--paraphrase-count", type=int, default=6)
    parser.add_argument("--negatives-per-rule", type=int, default=3)
    parser.add_argument("--questions-per-chunk", type=int, default=1,
                        help="Number of diverse questions to generate per policy chunk (default: 1)")
    args = parser.parse_args()
    model_kwargs = {"model": "gpt-4o", "temperature": 0.0, "max_tokens": 512}
    build_dataset_from_policy(policy_path=args.policy,
                              out_path=args.out,
                              call_llm_fn=call_llm,
                              model_kwargs=model_kwargs,
                              paraphrase_count=args.paraphrase_count,
                              negatives_per_rule=args.negatives_per_rule,
                              max_chunks=args.max_chunks,
                              questions_per_chunk=args.questions_per_chunk)

if __name__ == "__main__":
    main()