#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranking System
Team: [YOUR TEAM NAME]

Usage:
    python rank.py --candidates candidates.jsonl --out submission.csv

Architecture:
    1. Honeypot detection (rule-based consistency checks)
    2. Semantic embedding (all-MiniLM-L6-v2) on career history text
    3. Five-component scoring: semantic match, skills, title tier, experience, education
    4. Behavioral signal multiplier from redrob_signals
    5. Top 100 ranked candidates output as CSV
"""

import argparse
import json
import csv
import numpy as np
from datetime import datetime, date
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ============================================================
# JOB DESCRIPTION — Senior AI Engineer at Redrob AI
# ============================================================
JD = {
    "title": "Senior AI Engineer",
    "company": "Redrob AI",
    "location_cities": ["pune", "noida"],
    "experience_min": 5,
    "experience_max": 9,
    "must_have_skills": [
        "embeddings", "retrieval", "ranking", "llm", "fine-tuning",
        "python", "machine learning", "deep learning", "nlp"
    ],
    "good_to_have_skills": [
        "rag", "vector search", "recommendation systems", "pytorch", "tensorflow",
        "huggingface", "transformers", "bert", "search", "reranking",
        "faiss", "pinecone", "weaviate", "milvus", "kafka", "spark"
    ]
}

JD_TEXT = """
Senior AI Engineer at an AI-native talent intelligence startup.
Building ranking, retrieval and matching systems for candidates and jobs.
Deep technical work in embeddings, semantic search, LLMs, fine-tuning,
recommendation systems, vector search, retrieval pipelines.
Shipped production ML systems. Scrappy startup engineering culture.
Python, PyTorch, transformers, FAISS, vector databases.
Hybrid retrieval, re-ranking, evaluation frameworks.
"""

# ============================================================
# SCORING FUNCTIONS
# ============================================================

def classify_title(title):
    t = title.lower().strip()
    tier_a = ["ml engineer","machine learning engineer","ai engineer","data scientist",
              "nlp engineer","research engineer","deep learning","computer vision engineer",
              "applied scientist","ai researcher","ml researcher","senior data scientist",
              "staff ml","principal ml","ai research","applied ml"]
    tier_b = ["data engineer","analytics engineer","backend engineer",
              "senior software engineer","software architect",
              "platform engineer","senior data engineer","ai product"]
    tier_c = ["software engineer","full stack","backend","developer",
              "cloud engineer","devops","frontend engineer","sre",
              "data analyst","bi engineer"]
    for kw in tier_a:
        if kw in t: return 1.0, "A"
    for kw in tier_b:
        if kw in t: return 0.6, "B"
    for kw in tier_c:
        if kw in t: return 0.2, "C"
    return 0.0, "D"

def is_honeypot(candidate):
    profile = candidate["profile"]
    career = candidate["career_history"]
    skills = candidate["skills"]
    signals = candidate["redrob_signals"]
    total_months = sum(job["duration_months"] for job in career)
    claimed_years = profile["years_of_experience"]
    if abs(claimed_years - total_months/12) > 2.0:
        return True
    current_jobs = [j for j in career if j["is_current"]]
    if len(current_jobs) > 1:
        return True
    for skill in skills:
        if skill["proficiency"] == "expert" and skill.get("duration_months", 0) < 3:
            return True
    assessment_scores = signals.get("skill_assessment_scores", {})
    for skill in skills:
        if skill["name"] in assessment_scores:
            assessed = assessment_scores[skill["name"]]
            if skill["proficiency"] == "expert" and assessed < 30:
                return True
            if skill["proficiency"] == "advanced" and assessed < 20:
                return True
    if signals["profile_completeness_score"] < 30 and len(skills) > 15:
        return True
    return False

def get_career_text(candidate):
    parts = []
    for job in candidate["career_history"]:
        if job.get("title"): parts.append(job["title"])
        if job.get("description"): parts.append(job["description"])
    return " ".join(parts)

def score_skills(candidate):
    skill_lookup = {s["name"].lower().strip(): s for s in candidate["skills"]}
    must_have = JD["must_have_skills"]
    good_to_have = JD["good_to_have_skills"]
    total_possible = (len(must_have) * 2) + (len(good_to_have) * 1)
    total_earned = 0
    for skill_name in must_have:
        matched = next((skill_lookup[k] for k in skill_lookup
                       if skill_name in k or k in skill_name), None)
        if matched:
            duration = min(matched.get("duration_months", 0), 48)
            dm = 0.5 + (0.5 * duration / 48)
            eb = 0.1 * (min(matched.get("endorsements", 0), 50) / 50)
            total_earned += 2.0 * dm + eb
    for skill_name in good_to_have:
        matched = next((skill_lookup[k] for k in skill_lookup
                       if skill_name in k or k in skill_name), None)
        if matched:
            duration = min(matched.get("duration_months", 0), 48)
            dm = 0.5 + (0.5 * duration / 48)
            total_earned += 1.0 * dm
    return min(total_earned / total_possible if total_possible > 0 else 0, 1.0)

def score_experience(candidate):
    years = candidate["profile"]["years_of_experience"]
    career = candidate["career_history"]
    if 5 <= years <= 9: base = 1.0
    elif 4 <= years < 5: base = 0.8
    elif 9 < years <= 12: base = 0.7
    elif 3 <= years < 4: base = 0.5
    elif 12 < years <= 15: base = 0.4
    else: base = 0.2
    small = ["1-10","11-50","51-200"]
    startup_bonus = 0.1 if any(j["company_size"] in small for j in career) else 0.0
    all_large = all(j["company_size"] in ["1001-5000","5001-10000","10001+"] for j in career)
    large_penalty = -0.05 if all_large else 0.0
    return max(0.0, min(1.0, base + startup_bonus + large_penalty))

def score_education(candidate):
    education = candidate["education"]
    if not education: return 0.3
    tier_scores = {"tier_1":1.0,"tier_2":0.75,"tier_3":0.5,"tier_4":0.25,"unknown":0.4}
    return max(tier_scores.get(edu.get("tier","unknown"), 0.4) for edu in education)

def compute_signal_multiplier(candidate):
    signals = candidate["redrob_signals"]
    multiplier = 1.0
    if signals["open_to_work_flag"]: multiplier += 0.1
    else: multiplier -= 0.05
    try:
        last_active = datetime.strptime(signals["last_active_date"], "%Y-%m-%d").date()
        days_inactive = (date.today() - last_active).days
        if days_inactive <= 30: multiplier += 0.1
        elif days_inactive <= 90: multiplier += 0.05
        elif days_inactive > 180: multiplier -= 0.1
    except: pass
    rr = signals["recruiter_response_rate"]
    if rr >= 0.7: multiplier += 0.1
    elif rr >= 0.4: multiplier += 0.05
    elif rr < 0.2: multiplier -= 0.05
    gh = signals["github_activity_score"]
    if gh == -1: multiplier -= 0.05
    elif gh >= 70: multiplier += 0.1
    elif gh >= 40: multiplier += 0.05
    elif gh < 20: multiplier -= 0.05
    ir = signals["interview_completion_rate"]
    if ir >= 0.8: multiplier += 0.05
    elif ir < 0.5: multiplier -= 0.05
    return max(0.5, min(1.3, multiplier))

def generate_reasoning(candidate, scores):
    title = candidate["profile"]["current_title"]
    years = candidate["profile"]["years_of_experience"]
    signals = candidate["redrob_signals"]
    jd_skills = [s.lower() for s in JD["must_have_skills"] + JD["good_to_have_skills"]]
    matched = [s["name"] for s in candidate["skills"]
               if any(jk in s["name"].lower() or s["name"].lower() in jk
                      for jk in jd_skills)][:3]
    skills_str = ", ".join(matched) if matched else "general technical skills"
    reasoning = (f"{title} with {years} years of experience; "
                f"career history semantically aligned with AI/ML engineering; "
                f"matched skills include {skills_str}.")
    if signals["open_to_work_flag"] and signals["github_activity_score"] > 60:
        reasoning += f" Active on GitHub ({signals['github_activity_score']:.0f}/100)."
    elif not signals["open_to_work_flag"]:
        reasoning += " Note: not currently marked open to work."
    return reasoning

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print("Loading candidates...")
    with open(args.candidates, "r") as f:
        all_candidates = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(all_candidates)} candidates")

    print("Filtering honeypots...")
    candidates = [c for c in all_candidates if not is_honeypot(c)]
    print(f"  Clean pool: {len(candidates)} candidates")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding JD...")
    jd_embedding = model.encode(JD_TEXT)

    print("Embedding candidate career histories...")
    career_texts = [get_career_text(c) for c in candidates]
    career_embeddings = model.encode(career_texts, batch_size=256, show_progress_bar=True)

    print("Computing similarities...")
    similarities = cosine_similarity(jd_embedding.reshape(1,-1), career_embeddings)[0]

    print("Scoring all candidates...")
    all_scores = []
    for idx, c in enumerate(candidates):
        title_score = classify_title(c["profile"]["current_title"])[0]
        skills_score = score_skills(c)
        exp_score = score_experience(c)
        edu_score = score_education(c)
        sem = (float(similarities[idx]) + 1) / 2
        base = (0.35*sem + 0.25*skills_score + 0.20*title_score +
                0.15*exp_score + 0.05*edu_score)
        multiplier = compute_signal_multiplier(c)
        final = min(base * multiplier, 1.0)
        all_scores.append((final, idx, c))

    all_scores.sort(key=lambda x: x[0], reverse=True)
    top_100 = all_scores[:100]

    print(f"Writing top 100 to {args.out}...")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, idx, c) in enumerate(top_100, 1):
            reasoning = generate_reasoning(c, {})
            writer.writerow([c["candidate_id"], rank, round(score, 4), reasoning])

    print(f"Done. Top candidate: {top_100[0][2]['candidate_id']} "
          f"({top_100[0][2]['profile']['current_title']})")

if __name__ == "__main__":
    main()
