import json

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
    ],
    "ideal_career_signals": [
        "shipped ranking system", "built recommendation system",
        "production ml system", "retrieval system",
        "embedding pipeline", "startup experience", "founding team"
    ]
}

def classify_title(title):
    t = title.lower().strip()
    tier_a = ["ml engineer","machine learning engineer","ai engineer","data scientist",
              "nlp engineer","research engineer","deep learning","computer vision engineer",
              "applied scientist","ai researcher","ml researcher","senior data scientist",
              "staff ml","principal ml","ai research"]
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
        return True, f"Experience mismatch: claims {claimed_years}yrs but jobs add up to {total_months/12:.1f}yrs"
    current_jobs = [j for j in career if j["is_current"]]
    if len(current_jobs) > 1:
        return True, f"Multiple current jobs: {len(current_jobs)} marked is_current=True"
    for skill in skills:
        if skill["proficiency"] == "expert" and skill.get("duration_months", 0) < 3:
            return True, f"Expert claimed for {skill['name']} with {skill.get('duration_months',0)} months"
    assessment_scores = signals.get("skill_assessment_scores", {})
    for skill in skills:
        if skill["name"] in assessment_scores:
            assessed = assessment_scores[skill["name"]]
            if skill["proficiency"] == "expert" and assessed < 30:
                return True, f"Expert in {skill['name']} but scored {assessed}/100"
            if skill["proficiency"] == "advanced" and assessed < 20:
                return True, f"Advanced in {skill['name']} but scored {assessed}/100"
    if signals["profile_completeness_score"] < 30 and len(skills) > 15:
        return True, f"{len(skills)} skills but only {signals['profile_completeness_score']}% complete"
    return False, "clean"
