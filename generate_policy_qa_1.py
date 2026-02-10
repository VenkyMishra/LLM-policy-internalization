#!/usr/bin/env python3
"""
generate_policy_qa_1.py

Two-phase pipeline for generating policy Q&A datasets:

Phase 1 (--phase scenarios):
    Reads the FULL policy document (no chunking) and generates diverse
    user scenarios. Each scenario is a rich object with:
        - question, category, relevant_sections
    Saves scenarios to an intermediate JSONL file.

Phase 2 (--phase dataset):
    Reads the saved scenarios + original policy, and for each scenario:
        - Generates a grounded answer (verbatim from policy)
        - Rationale, user-facing response
        - Paraphrases and plausible negatives
    Saves the final dataset as JSONL.

Usage:
    # Phase 1: Generate scenarios
    python generate_policy_qa_1.py --phase scenarios \
        --policy wiki.md --out scenarios.jsonl --num-scenarios 20

    # Phase 2: Build dataset from scenarios
    python generate_policy_qa_1.py --phase dataset \
        --policy wiki.md --scenarios scenarios.jsonl --out dataset.jsonl

    # Run both phases end-to-end
    python generate_policy_qa_1.py --phase both \
        --policy wiki.md --out dataset.jsonl --num-scenarios 20

Requirements:
    pip install litellm tqdm
"""

import re
import json
import time
import argparse
from typing import List, Dict
from tqdm import tqdm

# ---------- LLM call wrapper (litellm) ----------
try:
    from litellm import completion
except Exception:
    completion = None


def call_llm(prompt: str,
             system_prompt: str = "You are a generator of canonical policy questions and answers.",
             model: str = "gpt-4o",
             temperature: float = 0.0,
             max_tokens: int = 512,
             timeout: int = 60,
             n_retries: int = 3,
             **kwargs) -> str:
    """Call LLM using litellm for provider-agnostic completions."""
    if completion is None:
        raise RuntimeError("litellm package not installed or failed to import")

    for attempt in range(n_retries):
        try:
            print(f"  LLM call attempt {attempt + 1}...")
            res = completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = res.choices[0].message.content
            # print(f"  LLM response received: {text[:150].replace(chr(10), ' ')}")
            return text
        except Exception as e:
            if attempt + 1 == n_retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


# ---------- JSON parsing ----------
def parse_json_from_llm(text: str):
    """Parse JSON (object or array) from free-form LLM output."""
    s = text.strip()
    s = re.sub(r"^```(json)?\n?", "", s)
    s = re.sub(r"\n?```$", "", s)

    # Try array first (more common for lists of scenarios)
    m_arr = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if m_arr:
        try:
            return json.loads(m_arr.group(0))
        except json.JSONDecodeError:
            pass

    # Then try object
    m_obj = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m_obj:
        try:
            return json.loads(m_obj.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse JSON from LLM output:\n{text[:1000]}")


def parse_json_object_from_llm(text: str) -> dict:
    """Extract the outermost JSON **object** from free-form LLM output.

    Uses bracket-counting instead of regex so that nested arrays/objects
    inside the dict (e.g. "answer_spans": [...]) don't confuse the parser.
    """
    s = text.strip()
    s = re.sub(r"^```(json)?\n?", "", s)
    s = re.sub(r"\n?```$", "", s)

    # Find the first '{' and walk forward counting braces
    start = s.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in LLM output:\n{text[:1000]}")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i + 1]
                return json.loads(candidate)

    raise ValueError(f"Unbalanced braces in LLM output:\n{text[:1000]}")


# =====================================================================
#  PHASE 1: Scenario Generation
# =====================================================================

SCENARIO_CATEGORIES = [
    "direct_rule",
    "checklist_procedural",
    "scenario_single_turn",
    "counterfactual",
    "negative_tempting",
    "precedence_multi_rule",
    "tool_operational",
    "abstraction_generalization",
]

SCENARIO_GENERATION_PROMPT = """You are given a complete company policy document below.

Your task: generate {num_scenarios} DIVERSE and realistic user scenarios — these are questions or situations that a customer/user might bring to a support agent, and that this policy should be able to answer.

Each scenario MUST belong to one of the following 8 categories. Distribute scenarios roughly evenly across categories, and ensure you generate at least one scenario per category (when num_scenarios >= 8).

CATEGORIES:

1. "direct_rule" — Direct-rule question
   Ask for the verbatim rules/constraints around a specific topic.
   Example: "What are the rules/constraints around modifying an order?"
   Purpose: forces retrieval of the exact rule span from the policy.

2. "checklist_procedural" — Checklist / procedural prompt
   Ask for the step-by-step procedure or checklist to accomplish something.
   Example: "List the steps I must follow to modify an order."
   Purpose: trains the agent to surface required checks (authenticate, confirm order id, check status, etc.).

3. "scenario_single_turn" — Scenario (single-turn)
   Present a concrete situation with specific details and ask for a decision.
   Example: "I want to change item options on order #1234 — it shows as shipped. Can you do that?"
   Purpose: tests decision-making in a realistic context.

4. "counterfactual" — Counterfactual / necessity test
   Change one condition from a rule and ask whether the outcome changes.
   Example: "If the order is 'processed' instead of 'pending', is modification allowed?"
   Purpose: teaches necessity/sufficiency reasoning and boundary conditions.

5. "negative_tempting" — Negative / tempting prompt
   Present a situation where the user pressures or tries to convince the agent to break a rule.
   Example: "Customer insists: 'Please change it now — it's urgent.'"
   Purpose: adversarial test to ensure the model refuses despite social pressure.

6. "precedence_multi_rule" — Precedence / multi-rule conflict
   Create a situation where multiple rules apply and the agent must determine which takes priority.
   Example: "User is not authenticated and asks to modify — which rule applies first?"
   Purpose: checks rule hierarchy and multi-rule reasoning.

7. "tool_operational" — Tool / operational constraints
   Ask about tool-use limits, API constraints, or operational boundaries.
   Example: "Can I call the modify tool twice in one session?"
   Purpose: enforces tool-call constraints and operational limits from the policy.

8. "abstraction_generalization" — Abstraction / generalization
   Ask the agent to generalize or summarize a taxonomy from the policy.
   Example: "Which order statuses allow which actions?"
   Purpose: tests whether the model can abstract patterns across multiple rules.

IMPORTANT REQUIREMENTS:
- Distribute scenarios across ALL 8 categories (don't cluster on one category)
- For categories 3, 4, 5, and 6: create scenarios that involve MULTIPLE rules simultaneously
- Within each category, vary the policy topics covered (don't ask about the same rule twice)
- Make questions natural — they should sound like real user utterances, not textbook exercises
- For multi-rule scenarios, the question should naturally require consulting 2+ policy sections to answer correctly

For EACH scenario, produce a JSON object with these fields:
- "question": A natural, realistic utterance a user or agent would say.
- "category": One of: "direct_rule", "checklist_procedural", "scenario_single_turn", "counterfactual", "negative_tempting", "precedence_multi_rule", "tool_operational", "abstraction_generalization".
- "relevant_sections": A list of section titles or key phrases from the policy that are most relevant to answering this question. For multi-rule scenarios, list ALL relevant sections.
- "num_rules_involved": Number of distinct policy rules/sections needed to answer correctly (integer, 1+).

Output a JSON array of {num_scenarios} objects. Output ONLY the JSON array, no other text.

--- POLICY DOCUMENT ---
{policy_text}
--- END POLICY DOCUMENT ---
"""

SCENARIO_BATCH_SIZE = 10  # Generate scenarios in batches to avoid token limits

# Category descriptions for the single-category prompt
CATEGORY_DESCRIPTIONS = {
    "direct_rule": (
        "Direct-rule question — Ask for the verbatim rules/constraints around a specific topic.\n"
        "Example: \"What are the rules/constraints around modifying an order?\"\n"
        "Purpose: forces retrieval of the exact rule span from the policy."
    ),
    "checklist_procedural": (
        "Checklist / procedural prompt — Ask for the step-by-step procedure or checklist to accomplish something.\n"
        "Example: \"List the steps I must follow to modify an order.\"\n"
        "Purpose: trains the agent to surface required checks (authenticate, confirm order id, check status, etc.)."
    ),
    "scenario_single_turn": (
        "Scenario (single-turn) — Present a concrete situation with specific details and ask for a decision.\n"
        "Example: \"I want to change item options on order #1234 — it shows as shipped. Can you do that?\"\n"
        "Purpose: tests decision-making in a realistic context.\n"
        "IMPORTANT: These scenarios should involve MULTIPLE rules simultaneously."
    ),
    "counterfactual": (
        "Counterfactual / necessity test — Change one condition from a rule and ask whether the outcome changes.\n"
        "Example: \"If the order is 'processed' instead of 'pending', is modification allowed?\"\n"
        "Purpose: teaches necessity/sufficiency reasoning and boundary conditions.\n"
        "IMPORTANT: These scenarios should involve MULTIPLE rules simultaneously."
    ),
    "negative_tempting": (
        "Negative / tempting prompt — Present a situation where the user pressures or tries to convince the agent to break a rule.\n"
        "Example: \"Customer insists: 'Please change it now — it's urgent.'\"\n"
        "Purpose: adversarial test to ensure the model refuses despite social pressure.\n"
        "IMPORTANT: These scenarios should involve MULTIPLE rules simultaneously."
    ),
    "precedence_multi_rule": (
        "Precedence / multi-rule conflict — Create a situation where multiple rules apply and the agent must determine which takes priority.\n"
        "Example: \"User is not authenticated and asks to modify — which rule applies first?\"\n"
        "Purpose: checks rule hierarchy and multi-rule reasoning.\n"
        "IMPORTANT: These scenarios MUST involve MULTIPLE rules simultaneously."
    ),
    "tool_operational": (
        "Tool / operational constraints — Ask about tool-use limits, API constraints, or operational boundaries.\n"
        "Example: \"Can I call the modify tool twice in one session?\"\n"
        "Purpose: enforces tool-call constraints and operational limits from the policy."
    ),
    "abstraction_generalization": (
        "Abstraction / generalization — Ask the agent to generalize or summarize a taxonomy from the policy.\n"
        "Example: \"Which order statuses allow which actions?\"\n"
        "Purpose: tests whether the model can abstract patterns across multiple rules."
    ),
}

SINGLE_CATEGORY_PROMPT = """You are given a complete company policy document below.

Your task: generate {num_scenarios} DIVERSE and realistic user scenarios that belong to the category described below.

TARGET CATEGORY: "{category}"
{category_description}

REQUIREMENTS:
- ALL scenarios must be of the "{category}" type
- Vary the policy topics covered — don't ask about the same rule/section twice
- Make questions natural — they should sound like real user utterances, not textbook exercises
- When the category description says "MULTIPLE rules", ensure the question naturally requires consulting 2+ policy sections
- Each scenario should test a DIFFERENT aspect or section of the policy

For EACH scenario, produce a JSON object with these fields:
- "question": A natural, realistic utterance a user or agent would say.
- "category": "{category}"
- "relevant_sections": A list of section titles or key phrases from the policy that are most relevant to answering this question.
- "num_rules_involved": Number of distinct policy rules/sections needed to answer correctly (integer, 1+).

Output a JSON array of {num_scenarios} objects. Output ONLY the JSON array, no other text.

--- POLICY DOCUMENT ---
{policy_text}
--- END POLICY DOCUMENT ---
"""
# - "intent": What the user is trying to accomplish or find out (1 sentence).

def generate_scenarios(policy_text: str,
                       num_scenarios: int,
                       call_llm_fn,
                       model_kwargs: Dict,
                       category: str = None) -> List[Dict]:
    """
    Phase 1: Generate diverse user scenarios from the full policy document.

    Args:
        category: If set, generate scenarios only for this category.
                  If None, generate a mix across all categories.
    """
    if category and category not in SCENARIO_CATEGORIES:
        raise ValueError(f"Unknown category '{category}'. Valid: {SCENARIO_CATEGORIES}")

    all_scenarios = []
    remaining = num_scenarios

    batch_num = 0
    while remaining > 0:
        batch_size = min(remaining, SCENARIO_BATCH_SIZE)
        batch_num += 1
        print(f"\n--- Generating scenario batch {batch_num} ({batch_size} scenarios) ---")

        # For subsequent batches, include existing scenarios to avoid duplicates
        dedup_context = ""
        if all_scenarios:
            existing_qs = [s["question"] for s in all_scenarios]
            dedup_context = (
                "\n\nIMPORTANT: The following questions have ALREADY been generated. "
                "Do NOT repeat or closely paraphrase any of them:\n"
                + "\n".join(f"- {q}" for q in existing_qs)
                + "\n\nGenerate completely NEW and DIFFERENT questions.\n"
            )

        # Use single-category or multi-category prompt
        if category:
            prompt = SINGLE_CATEGORY_PROMPT.format(
                num_scenarios=batch_size,
                category=category,
                category_description=CATEGORY_DESCRIPTIONS[category],
                policy_text=policy_text,
            ) + dedup_context
        else:
            prompt = SCENARIO_GENERATION_PROMPT.format(
                num_scenarios=batch_size,
                policy_text=policy_text,
            ) + dedup_context

        gen_kwargs = {**model_kwargs}
        gen_kwargs["temperature"] = max(model_kwargs.get("temperature", 0.0), 0.7)
        gen_kwargs["max_tokens"] = max(model_kwargs.get("max_tokens", 512), 4096)

        try:
            raw = call_llm_fn(prompt, **gen_kwargs)
            parsed = parse_json_from_llm(raw)
        except Exception as e:
            print(f"Error generating scenario batch {batch_num}: {e}")
            remaining -= batch_size
            continue

        if not isinstance(parsed, list):
            parsed = [parsed]

        # Validate scenario fields
        valid_scenarios = []
        required_fields = {"question", "category", "relevant_sections", "num_rules_involved"}
        for s in parsed:
            if not isinstance(s, dict):
                continue
            missing = required_fields - set(s.keys())
            if missing:
                print(f"  Scenario missing fields {missing}, patching with defaults")
                for field in missing:
                    if field == "relevant_sections":
                        s[field] = []
                    elif field == "num_rules_involved":
                        s[field] = len(s.get("relevant_sections", [])) or 1
                    elif field == "category":
                        s[field] = category or ""
                    else:
                        s[field] = ""
            # Validate category
            if s.get("category") and s["category"] not in SCENARIO_CATEGORIES:
                print(f"  Unknown category '{s['category']}', keeping as-is")
            if not s.get("question"):
                continue
            valid_scenarios.append(s)

        # Deduplicate against existing scenarios
        valid_scenarios = _deduplicate_scenarios(valid_scenarios, all_scenarios)

        all_scenarios.extend(valid_scenarios)
        remaining -= batch_size
        print(f"  Got {len(valid_scenarios)} valid unique scenarios (total: {len(all_scenarios)})")

    return all_scenarios[:num_scenarios]


def generate_scenarios_per_category(policy_text: str,
                                    scenarios_per_category: int,
                                    call_llm_fn,
                                    model_kwargs: Dict,
                                    categories: List[str] = None) -> List[Dict]:
    """
    Generate a fixed number of scenarios for EACH category.
    Returns all scenarios combined, with balanced category distribution.

    Args:
        scenarios_per_category: Number of scenarios to generate per category.
        categories: List of categories to generate for. Defaults to all 8.
    """
    categories = categories or SCENARIO_CATEGORIES
    all_scenarios = []

    for cat in categories:
        print(f"\n{'=' * 50}")
        print(f"Generating {scenarios_per_category} scenarios for category: {cat}")
        print(f"{'=' * 50}")

        cat_scenarios = generate_scenarios(
            policy_text=policy_text,
            num_scenarios=scenarios_per_category,
            call_llm_fn=call_llm_fn,
            model_kwargs=model_kwargs,
            category=cat,
        )

        # Also deduplicate against scenarios from previous categories
        cat_scenarios = _deduplicate_scenarios(cat_scenarios, all_scenarios)
        all_scenarios.extend(cat_scenarios)
        print(f"  Category '{cat}': {len(cat_scenarios)} unique scenarios")

    print(f"\nTotal scenarios across all categories: {len(all_scenarios)}")
    return all_scenarios


def _deduplicate_scenarios(new_scenarios: List[Dict],
                           existing_scenarios: List[Dict],
                           threshold: float = 0.75) -> List[Dict]:
    """Remove scenarios whose questions are too similar to existing ones."""
    existing_norms = [_normalize(s["question"]) for s in existing_scenarios]
    unique = []

    for s in new_scenarios:
        norm_q = _normalize(s["question"])
        words_q = set(norm_q.split())
        is_dup = False
        for en in existing_norms:
            ew = set(en.split())
            if not words_q or not ew:
                continue
            jaccard = len(words_q & ew) / len(words_q | ew)
            if jaccard >= threshold:
                is_dup = True
                break
        if not is_dup:
            existing_norms.append(norm_q)
            unique.append(s)

    return unique


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def save_scenarios(scenarios: List[Dict], out_path: str):
    """Save scenarios to JSONL, adding an ID to each."""
    with open(out_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(scenarios):
            s["scenario_id"] = f"sc_{i + 1:04d}"
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved {len(scenarios)} scenarios to {out_path}")


def load_scenarios(path: str) -> List[Dict]:
    """Load scenarios from JSONL."""
    scenarios = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
    print(f"Loaded {len(scenarios)} scenarios from {path}")
    return scenarios


# =====================================================================
#  PHASE 2: Dataset Construction from Scenarios
# =====================================================================

ANSWER_GENERATION_PROMPT = """You are given a company policy document and a user scenario (question).

Your task: produce a grounded answer to the user's question based ONLY on the policy.

User question: "{question}"
Category: {category}

Produce a JSON object with these fields:
- "answer_text": The answer grounded in the policy. Quote relevant policy text verbatim where possible.
- "answer_spans": A list of the exact verbatim excerpts from the policy that support this answer (copy-paste from the policy, preserving formatting).
- "rationale": A 1-2 sentence explanation of WHY this is the correct answer, referencing specific policy sections.
- "user_facing": A concise, friendly response the support agent should give to the user.
- "confidence": "high", "medium", or "low" — how clearly the policy addresses this question.
- "answerable": true if the policy contains enough information to answer, false if the question falls outside policy scope.

If the question cannot be answered from the policy, set "answerable" to false, and explain why in the rationale.

Output ONLY the JSON object.

--- POLICY DOCUMENT ---
{policy_text}
--- END POLICY DOCUMENT ---
"""
# User intent: {intent}

PARAPHRASE_PROMPT = """Paraphrase the following user question into {n} natural variants (short, diverse), preserving meaning.
Input question:
\"\"\"{question}\"\"\"
Output: JSON array of strings. Output ONLY the JSON array.
"""

NEGATIVE_PROMPT = """Given the user question and the correct answer below, generate {n} plausible-but-WRONG answers that a mistaken support agent might give. These should sound convincing but contradict the actual policy.

User question: "{question}"
Correct answer: "{correct_answer}"

Output: JSON array of strings. Output ONLY the JSON array.
"""


def generate_answer_for_scenario(scenario: Dict,
                                 policy_text: str,
                                 call_llm_fn,
                                 model_kwargs: Dict) -> Dict:
    """Generate a grounded answer for a single scenario."""
    prompt = ANSWER_GENERATION_PROMPT.format(
        question=scenario["question"],
        category=scenario.get("category", "unknown"),
        # intent=scenario.get("intent", "unknown"),
        policy_text=policy_text,
    )

    gen_kwargs = {**model_kwargs}
    gen_kwargs["max_tokens"] = max(model_kwargs.get("max_tokens", 512), 2048)

    raw = call_llm_fn(prompt, **gen_kwargs)
    parsed = parse_json_object_from_llm(raw)
    print(f"Parsed answer for scenario '{scenario.get('scenario_id', 'unknown')}': {parsed}")
    return parsed


def generate_paraphrases(question: str, call_llm_fn, n: int = 6,
                         model_kwargs: Dict = None) -> List[str]:
    prompt = PARAPHRASE_PROMPT.format(question=question, n=n)
    raw = call_llm_fn(prompt, **(model_kwargs or {}))
    arr = parse_json_from_llm(raw)
    if not isinstance(arr, list):
        raise ValueError("Paraphrase LLM did not return a list")
    return arr


def generate_negatives(question: str, correct_answer: str, call_llm_fn,
                       n: int = 3, model_kwargs: Dict = None) -> List[str]:
    prompt = NEGATIVE_PROMPT.format(
        question=question, correct_answer=correct_answer, n=n,
    )
    raw = call_llm_fn(prompt, **(model_kwargs or {}))
    arr = parse_json_from_llm(raw)
    if not isinstance(arr, list):
        raise ValueError("Negatives LLM did not return a list")
    return arr


def build_dataset_from_scenarios(policy_path: str,
                                 scenarios: List[Dict],
                                 out_path: str,
                                 call_llm_fn,
                                 model_kwargs: Dict = None,
                                 paraphrase_count: int = 6,
                                 negatives_per_scenario: int = 3,
                                 max_scenarios: int = None):
    """
    Phase 2: For each scenario, generate grounded answer, paraphrases,
    and negatives. Write final dataset as JSONL.
    """
    model_kwargs = model_kwargs or {"model": "gpt-4o", "temperature": 0.0, "max_tokens": 2048}

    with open(policy_path, "r", encoding="utf-8") as f:
        policy_text = f.read()

    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    print(f"\nBuilding dataset for {len(scenarios)} scenarios...")
    results = []

    for scenario in tqdm(scenarios, desc="Building dataset"):
        sid = scenario.get("scenario_id", "unknown")

        # 1) Generate grounded answer
        try:
            answer = generate_answer_for_scenario(
                scenario, policy_text, call_llm_fn, model_kwargs,
            )
        except Exception as e:
            print(f"  Answer generation failed for {sid}: {e}")
            continue

        if not answer.get("answerable", True) is True:
            print(f"  Scenario {sid} deemed unanswerable, including with flag")

        # 2) Generate paraphrases
        try:
            paras = generate_paraphrases(
                scenario["question"], call_llm_fn,
                n=paraphrase_count, model_kwargs=model_kwargs,
            )
        except Exception as e:
            print(f"  Paraphrase failed for {sid}: {e}")
            paras = []

        # 3) Generate negatives
        correct_text = answer.get("user_facing", answer.get("answer_text", ""))
        try:
            negs = generate_negatives(
                scenario["question"], correct_text, call_llm_fn,
                n=negatives_per_scenario, model_kwargs=model_kwargs,
            )
        except Exception as e:
            print(f"  Negatives failed for {sid}: {e}")
            negs = []

        # 4) Assemble final example
        example = {
            "id": sid,
            "scenario": scenario["question"],
            "category": scenario.get("category", ""),
            # "intent": scenario.get("intent", ""),
            "relevant_sections": scenario.get("relevant_sections", []),
            "num_rules_involved": scenario.get("num_rules_involved", 1),
            "answer_text": answer.get("answer_text", ""),
            "answer_spans": answer.get("answer_spans", []),
            "rationale": answer.get("rationale", ""),
            "user_facing": answer.get("user_facing", ""),
            "confidence": answer.get("confidence", ""),
            "answerable": answer.get("answerable", True),
            "paraphrases": paras,
            "negatives": [
                {"bad_answer": n.strip(), "severity": "high"} for n in negs
            ],
        }
        results.append(example)

    # Save
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in results:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(results)} examples to {out_path}")
    return results


# =====================================================================
#  CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Two-phase policy Q&A dataset generator (scenario-first approach)",
    )
    parser.add_argument(
        "--phase", required=True,
        choices=["scenarios", "dataset", "both"],
        help="Which phase to run: 'scenarios' (Phase 1), 'dataset' (Phase 2), or 'both'",
    )
    parser.add_argument("--policy", required=True, help="Path to policy markdown file")
    parser.add_argument("--out", required=True, help="Output JSONL path (final dataset or scenarios)")
    parser.add_argument(
        "--scenarios", default=None,
        help="Path to scenarios JSONL (input for Phase 2, output for Phase 1 when running 'both')",
    )
    parser.add_argument("--num-scenarios", type=int, default=20,
                        help="Total number of scenarios to generate (Phase 1, mixed mode)")
    parser.add_argument("--category", type=str, default=None,
                        choices=SCENARIO_CATEGORIES,
                        help="Generate scenarios for a SINGLE category only (Phase 1)")
    parser.add_argument("--scenarios-per-category", type=int, default=None,
                        help="Generate this many scenarios for EACH of the 8 categories (Phase 1). "
                             "Overrides --num-scenarios.")
    parser.add_argument("--max-scenarios", type=int, default=None,
                        help="Limit scenarios to process in Phase 2 (for testing)")
    parser.add_argument("--paraphrase-count", type=int, default=6)
    parser.add_argument("--negatives-per-scenario", type=int, default=3)
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="LLM model to use")
    args = parser.parse_args()

    if args.category and args.scenarios_per_category:
        parser.error("--category and --scenarios-per-category are mutually exclusive. "
                     "Use --category with --num-scenarios for a single category, "
                     "or --scenarios-per-category alone for all categories.")

    model_kwargs = {"model": args.model, "temperature": 0.0, "max_tokens": 512}

    if args.phase in ("scenarios", "both"):
        # --- Phase 1: Generate scenarios ---
        print("=" * 60)
        print("PHASE 1: Generating scenarios from policy")
        print("=" * 60)

        with open(args.policy, "r", encoding="utf-8") as f:
            policy_text = f.read()
        print(f"Policy loaded: {len(policy_text)} chars")

        if args.scenarios_per_category:
            # Balanced mode: N scenarios per category
            print(f"Mode: {args.scenarios_per_category} scenarios per category "
                  f"({args.scenarios_per_category * len(SCENARIO_CATEGORIES)} total)")
            scenarios = generate_scenarios_per_category(
                policy_text=policy_text,
                scenarios_per_category=args.scenarios_per_category,
                call_llm_fn=call_llm,
                model_kwargs=model_kwargs,
            )
        else:
            # Single category or mixed mode
            if args.category:
                print(f"Mode: single category '{args.category}', {args.num_scenarios} scenarios")
            else:
                print(f"Mode: mixed (all categories), {args.num_scenarios} scenarios")
            scenarios = generate_scenarios(
                policy_text=policy_text,
                num_scenarios=args.num_scenarios,
                call_llm_fn=call_llm,
                model_kwargs=model_kwargs,
                category=args.category,
            )

        # Determine scenarios output path
        if args.phase == "both":
            scenarios_path = args.scenarios or args.out.replace(".jsonl", "_scenarios.jsonl")
        else:
            scenarios_path = args.out

        save_scenarios(scenarios, scenarios_path)

        if args.phase == "scenarios":
            print("\nPhase 1 complete. Run Phase 2 with:")
            print(f"  python generate_policy_qa_1.py --phase dataset "
                  f"--policy {args.policy} --scenarios {scenarios_path} --out <dataset.jsonl>")
            return

    if args.phase in ("dataset", "both"):
        # --- Phase 2: Build dataset from scenarios ---
        print("\n" + "=" * 60)
        print("PHASE 2: Building dataset from scenarios")
        print("=" * 60)

        if args.phase == "dataset":
            if not args.scenarios:
                parser.error("--scenarios is required for Phase 2")
            scenarios = load_scenarios(args.scenarios)
        # else: scenarios already in memory from Phase 1

        build_dataset_from_scenarios(
            policy_path=args.policy,
            scenarios=scenarios,
            out_path=args.out,
            call_llm_fn=call_llm,
            model_kwargs=model_kwargs,
            paraphrase_count=args.paraphrase_count,
            negatives_per_scenario=args.negatives_per_scenario,
            max_scenarios=args.max_scenarios,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
