import requests
import pandas as pd
import json
import re
import time
import os
from tqdm import tqdm
from openai import OpenAI
from google import genai
import concurrent.futures

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

client = genai.Client(api_key="")

df = pd.read_csv("dataset.csv")

df.drop("label", axis=1)

prompt = """
You are an expert linguistic analyst specializing in sarcasm detection in Indian political headlines.

Your task is to classify the given headline as "sarcastic" or "non-sarcastic" using structured reasoning.

Follow these steps carefully. Sarcasm in headlines is often subtle and may rely on irony, mock praise, exaggeration, or contextual skepticism toward political claims.

When ambiguity exists, prefer identifying possible sarcasm rather than rejecting it outright. Even subtle signals of irony should be considered.

---------------------------
ANNOTATION PRINCIPLES
---------------------------

• Sarcasm can exist even if only ONE rule is triggered.
• Rules may be triggered STRONGLY or WEAKLY.
• Subtle irony, skepticism, or mocking tone may indicate sarcasm even without obvious exaggeration.
• If a headline could reasonably be interpreted as ironic or mocking by a reader familiar with political discourse, it may be labeled sarcastic.
• Sarcasm can still be labeled even with moderate confidence. Use lower confidence values if sarcasm is subtle.

---------------------------
STEP 1
---------------------------
Determine the literal meaning of the headline.

---------------------------
STEP 2
---------------------------
Determine the implied or intended meaning (if different from the literal meaning).

---------------------------
STEP 3
---------------------------
Evaluate the headline using the following sarcasm rule categories.

---------------------------
SARCASM RULE CATEGORIES
---------------------------

A. Sentiment & Polarity Rules
1. Sentiment–Situation Mismatch:
   Positive tone describing negative events OR negative tone describing positive events.

2. Polarity Reversal:
   Surface sentiment opposite to implied target sentiment.

3. Mock Praise:
   Praise used to imply criticism.

---------------------------

B. Hyperbole & Exaggeration Rules
4. Extreme exaggeration or overgeneralization (e.g., “entire nation”, “everyone agrees”).

5. Absurd or impossible outcomes.

6. Semantic disproportion:
   Small issue framed as national triumph or disaster.

---------------------------

C. Logical & Causal Incongruity Rules
7. Absurd cause–effect relationship.

8. Policy–Outcome inversion:
   Harmful policy framed as beneficial.

9. Statistical manipulation humor:
   Redefining metrics to claim success.

---------------------------

D. Structural & Semantic Contradictions
10. Internal paradox or contradiction (e.g., “transparent corruption”).

11. Unexpected role reversal.

12. Understatement of severe crisis.

---------------------------

E. Linguistic Markers
13. Ironic intensifiers (e.g., “Wow”, “Historic”, “Masterstroke”) used in contradictory contexts.

14. Quotation mark skepticism
   (e.g., “development”, “transparency”).

15. Faux neutral journalism tone masking absurdity.

---------------------------

F. Political & Cultural Context (Indian Politics)
16. Election slogan reframing
   (manifesto promises referenced ironically).

17. Bureaucratic absurdity framed as innovation.

18. Widely debated controversies framed as achievements.

---------------------------

G. Subtle Irony
19. Headline wording that subtly mocks or questions political claims.

20. Language implying skepticism toward official narratives.

---------------------------

H. Contextual Political Sarcasm
21. Headline framing that implicitly critiques a political claim, policy, or public narrative even without explicit contradiction.

---------------------------
STEP 4
---------------------------
List all rule numbers that are triggered.

Include rules that are strongly OR weakly triggered.
Even partial or subtle rule matches should be included.

---------------------------
STEP 5
---------------------------
Assign the final label.

If ONE OR MORE rules are triggered, sarcasm may be present.

Use the following guideline:

Label "sarcastic" if:
• at least one rule is triggered AND
• the headline contains irony, exaggeration, mock praise, contradiction, or skepticism.

Otherwise label "non-sarcastic".

---------------------------
STEP 6
---------------------------
Assign a sarcasm confidence score between 0 and 1.

Guidelines:
0.3–0.5 → subtle or weak sarcasm
0.5–0.7 → moderate sarcasm
0.7–1.0 → strong or obvious sarcasm

---------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
---------------------------

{
  "literal_meaning": "...",
  "implied_meaning": "...",
  "triggered_rules": [rule_numbers],
  "reasoning_summary": "brief explanation in 2-3 sentences",
  "label": "sarcastic" or "non-sarcastic",
  "confidence": 0.0-1.0
}

Headline: "<INSERT HEADLINE HERE>"
"""

import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# Prompt Builder
# ----------------------------
def build_prompt(base_prompt, headline):
    return base_prompt.replace("<INSERT HEADLINE HERE>", headline)

# ----------------------------
# JSON Parser
# ----------------------------
def parse_response(response_text):
    try:
        return json.loads(response_text)
    except:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        json_str = response_text[start:end]
        return json.loads(json_str)

# ----------------------------
# API Call
# ----------------------------
def annotate_headline(headline, base_prompt, model_name="gemini-3-flash-preview"):

    full_prompt = build_prompt(base_prompt, headline)

    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt
    )

    response_text = response.text.strip()

    parsed = parse_response(response_text)

    return {
        "label": parsed.get("label"),
        "confidence": parsed.get("confidence")
    }

# ----------------------------
# Worker Function
# ----------------------------
def worker(idx, row, text_column, base_prompt):
    try:
        headline = row[text_column]
        result = annotate_headline(headline, base_prompt)
        return idx, result

    except Exception as e:
        print(f"Error at index {idx}: {e}")
        return idx, {"label": None, "confidence": None}

# ----------------------------
# Parallel Annotation Function
# ----------------------------
def annotate_dataframe_parallel(df, text_column, base_prompt, max_workers=50):

    results = [None] * len(df)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = [
            executor.submit(worker, idx, row, text_column, base_prompt)
            for idx, row in df.iterrows()
        ]

        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    result_df = pd.DataFrame(results)

    return pd.concat([df.reset_index(drop=True), result_df], axis=1)

# ----------------------------
# Run Annotation
# ----------------------------
annotated_df = annotate_dataframe_parallel(
    df.head(10000),
    text_column="headline",   # change if column name differs
    base_prompt=prompt,
    max_workers=50
)

print("Annotation Complete")
print(annotated_df.head())

annotated_df.to_csv("/content/drive/MyDrive/dataset_labeled.csv", index=False)