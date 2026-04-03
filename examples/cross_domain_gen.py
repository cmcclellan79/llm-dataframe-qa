"""
Cross-domain training data generation demo.
Generates question/Pandas-query pairs from metadata ONLY across three domains.
"""

import json
import re
import time
from anthropic import Anthropic

client = Anthropic()
MODEL = "claude-sonnet-4-20250514"

DOMAINS = {
    "medical": {
        "name": "Heart Disease Dataset",
        "description": "Clinical records for heart disease diagnosis including patient demographics, test results, and diagnosis outcomes.",
        "columns": ["age", "sex", "chest_pain_type", "resting_bp", "cholesterol", "fasting_blood_sugar", "rest_ecg", "max_heart_rate", "exercise_angina", "oldpeak", "slope", "num_major_vessels", "thalassemia", "target"],
        "dtypes": ["int", "str", "str", "float", "float", "bool", "str", "float", "bool", "float", "str", "int", "str", "int"],
        "column_descriptions": {
            "target": "0 = no heart disease, 1-4 = increasing severity",
            "chest_pain_type": "typical angina, atypical angina, non-anginal pain, asymptomatic",
            "oldpeak": "ST depression induced by exercise relative to rest",
            "num_major_vessels": "number of major vessels (0-3) colored by fluoroscopy",
        },
    },
    "automotive": {
        "name": "Automobile Dataset",
        "description": "Specifications and pricing of various automobiles including performance metrics, dimensions, and insurance risk ratings.",
        "columns": ["make", "fuel_type", "aspiration", "num_doors", "body_style", "drive_wheels", "engine_location", "wheelbase", "length", "width", "height", "curb_weight", "engine_type", "num_cylinders", "engine_size", "fuel_system", "horsepower", "city_mpg", "highway_mpg", "price"],
        "dtypes": ["str", "str", "str", "str", "str", "str", "str", "float", "float", "float", "float", "int", "str", "int", "int", "str", "int", "int", "int", "float"],
    },
    "sports": {
        "name": "NBA Player Stats",
        "description": "Season statistics for NBA players including scoring, assists, rebounds, and efficiency metrics.",
        "columns": ["player", "team", "position", "age", "games_played", "minutes_per_game", "points_per_game", "assists_per_game", "rebounds_per_game", "steals_per_game", "blocks_per_game", "field_goal_pct", "three_point_pct", "free_throw_pct", "turnovers_per_game", "player_efficiency_rating"],
        "dtypes": ["str", "str", "str", "int", "int", "float", "float", "float", "float", "float", "float", "float", "float", "float", "float", "float"],
    },
}

ROLES = ["data_scientist", "general_user", "data_owner"]

ROLE_DESCRIPTIONS = {
    "data_scientist": "Expert-level analytical questions requiring advanced Pandas operations like groupby, correlation, pivot tables, statistical analysis, and multi-step transformations.",
    "general_user": "Practical questions a non-technical person would ask, avoiding specific column names, focused on consumer-oriented insights.",
    "data_owner": "Business-oriented questions focused on trends, performance tracking, strategic insights, and decision-making.",
}


def generate_pairs(domain_key, role, num_pairs=5):
    domain = DOMAINS[domain_key]
    col_info = "\n".join(
        f"  {col} ({dt})" + (f" — {domain.get('column_descriptions', {}).get(col, '')}" if domain.get('column_descriptions', {}).get(col) else "")
        for col, dt in zip(domain["columns"], domain["dtypes"])
    )

    prompt = f"""You are given a dataframe. Generate {num_pairs} realistic question/Pandas-query pairs for the {role} role.

Dataset: {domain['name']}
Description: {domain.get('description', 'N/A')}

Columns:
{col_info}

Role characteristics: {ROLE_DESCRIPTIONS[role]}

Assumptions:
- Pandas imported as 'pd', dataframe is 'df', all strings lowercase.
- Store answer in 'result'. No comments, no imports.

Return ONLY a JSON array:
[{{"question": "...", "query": "result = ..."}}]"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)

    try:
        pairs = json.loads(raw)
    except json.JSONDecodeError:
        pairs = []

    for p in pairs:
        p["domain"] = domain_key
        p["role"] = role
        q = p.get("query", "").lower()
        if any(op in q for op in ["groupby", "corr", "pivot", "merge", "apply", "rolling", "crosstab"]):
            p["complexity"] = "data_analysis"
        elif any(op in q for op in [".sum()", ".count()", ".mean()", ".max()", ".min()", ".nunique()"]):
            p["complexity"] = "aggregation"
        else:
            p["complexity"] = "retrieval"

    return pairs, message.usage


def main():
    all_pairs = []
    total_input = 0
    total_output = 0

    print(f"\nGenerating cross-domain training data...")
    print(f"Domains: {list(DOMAINS.keys())}")
    print(f"Roles: {ROLES}")
    print(f"Pairs per combo: 5")
    print(f"Expected total: {len(DOMAINS) * len(ROLES) * 5}\n")

    for domain_key in DOMAINS:
        for role in ROLES:
            label = f"{domain_key}/{role}"
            print(f"  {label}...", end=" ", flush=True)
            start = time.time()
            pairs, usage = generate_pairs(domain_key, role, num_pairs=5)
            elapsed = time.time() - start
            total_input += usage.input_tokens
            total_output += usage.output_tokens
            all_pairs.extend(pairs)
            print(f"{len(pairs)} pairs ({elapsed:.1f}s)")

    # Build stats
    by_domain = {}
    by_role = {}
    by_complexity = {}
    for p in all_pairs:
        d = p.get("domain", "?")
        r = p.get("role", "?")
        c = p.get("complexity", "?")
        by_domain[d] = by_domain.get(d, 0) + 1
        by_role[r] = by_role.get(r, 0) + 1
        by_complexity[c] = by_complexity.get(c, 0) + 1

    output = {
        "metadata": {
            "model": MODEL,
            "domains": {k: {"name": v["name"], "num_columns": len(v["columns"])} for k, v in DOMAINS.items()},
            "total_pairs": len(all_pairs),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "by_domain": by_domain,
            "by_role": by_role,
            "by_complexity": by_complexity,
        },
        "pairs": all_pairs,
    }

    with open("results/cross_domain_training_data.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone! {len(all_pairs)} total pairs")
    print(f"Tokens used: {total_input + total_output:,}")
    print(f"Saved to results/cross_domain_training_data.json")


if __name__ == "__main__":
    main()
