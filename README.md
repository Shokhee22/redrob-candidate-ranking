# Redrob Intelligent Candidate Ranking System

## Overview
An intelligent candidate ranking engine built for the Redrob Hackathon.
Ranks 100,000 candidates against a Senior AI Engineer job description using
semantic embeddings, trust-weighted skill scoring, and behavioral signal modulation.

## Architecture

    candidates.jsonl
          |
          v
    [1] Honeypot Detection (rule-based consistency checks)
          | removes ~408 suspicious profiles
          v
    [2] Semantic Embedding (all-MiniLM-L6-v2)
          | embeds career history descriptions vs JD text
          v
    [3] Five-Component Scoring
          | semantic match (35%) + skills (25%) + title tier (20%)
          | + experience fit (15%) + education (5%)
          v
    [4] Behavioral Signal Multiplier (0.5x to 1.3x)
          | open_to_work, last_active_date, github_activity,
          | recruiter_response_rate, interview_completion_rate
          v
    [5] Ranked Top 100 CSV with reasoning

## Key Design Decisions

**Why career history embeddings, not skills list:**
The JD explicitly warns against keyword stuffers. We embed actual job description
text rather than self-reported skill tags, which are trivially inflatable.

**Why trust-weighted skills:**
Each matched skill is weighted by duration_months. A skill claimed with 0 months
use gets half credit at most. Directly counters the expert-with-zero-experience honeypot pattern.

**Why a multiplier, not additive signals:**
Behavioral signals modulate the base score rather than add to it. A great-on-paper
candidate who is inactive and unresponsive gets pulled down proportionally.

**Honeypot detection catches:**
- Experience duration mismatches over 2 years
- Multiple is_current true jobs simultaneously
- Expert proficiency with under 3 months duration
- Assessment scores contradicting claimed proficiency
- High skill count with very low profile completeness

## How to Run

    pip install -r requirements.txt
    python src/rank.py --candidates candidates.jsonl --out submission.csv

Runs in under 5 minutes on CPU with 16GB RAM. No network calls during ranking.

## Results

- Top candidates: Senior AI Engineer, Lead AI Engineer, Senior Applied Scientist
- All top 100 have AI/ML relevant titles
- Honeypots removed: 408
- Validator: PASSED

## Files

- src/rank.py — Main ranking script
- src/scoring_utils.py — Reusable scoring functions
- requirements.txt — Python dependencies
- output/submission.csv — Final ranked top 100
