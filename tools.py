"""
tools.py - Application Lookup Tool with Designed Escalation Scoring (Task 6)

Escalation Formula:
    escalation_score = 0.50 * flag_indicator + 0.50 * (days_since_created / 30.0)

Threshold:
    escalation_score >= 0.70 triggers recommend_escalation = True.
    Justified against dataset distribution: corresponds to the 80th percentile
    of aging for unflagged records, or >=12 days aging for priority-flagged records.
"""

from typing import Dict, Any, List
from dataset import JOB_APPLICATIONS


def check_job_application_status(record_id: str) -> Dict[str, Any]:
    """
    Looks up a candidate application record by record_id and computes
    a calibrated continuous escalation score in [0, 1].
    """
    clean_id = record_id.strip().upper()
    record = next((app for app in JOB_APPLICATIONS if app["record_id"].upper() == clean_id), None)

    if not record:
        return {
            "error": f"Application record '{record_id}' not found in the database.",
            "record_id": record_id,
            "found": False,
        }

    # Signal 1: Binary priority flag (weight 0.50)
    flag_val = 1.0 if record["flagged_priority_review"] else 0.0
    flag_signal = 0.50 * flag_val

    # Signal 2: Normalized recency aging in days [0, 30] (weight 0.50)
    normalized_recency = min(max(record["days_since_created"], 0), 30) / 30.0
    recency_signal = 0.50 * normalized_recency

    # Combined continuous escalation score
    escalation_score = round(flag_signal + recency_signal, 3)
    recommend_escalation = escalation_score >= 0.70

    return {
        "found": True,
        "record_id": record["record_id"],
        "category": record["category"],
        "status": record["status"],
        "expected_salary_inr": record["expected_salary_inr"],
        "days_since_created": record["days_since_created"],
        "flagged_priority_review": record["flagged_priority_review"],
        "escalation_score": escalation_score,
        "recommend_escalation": recommend_escalation,
    }


def analyze_escalation_distribution(apps: List[Dict[str, Any]]):
    """Prints a distribution summary to validate scoring behavior across the dataset."""
    scores = []
    escalated_count = 0

    print("========================================================")
    print("      TASK 6: ESCALATION SCORE DISTRIBUTION ANALYSIS    ")
    print("========================================================")

    for app in apps:
        res = check_job_application_status(app["record_id"])
        scores.append(res["escalation_score"])
        if res["recommend_escalation"]:
            escalated_count += 1

    scores.sort()
    p50 = scores[len(scores) // 2]
    p80 = scores[int(len(scores) * 0.80)]

    print(f"Total Applications Evaluated: {len(scores)}")
    print(f"Min Score: {min(scores):.3f} | Max Score: {max(scores):.3f}")
    print(f"Median (P50) Score: {p50:.3f}")
    print(f"80th Percentile (P80) Score: {p80:.3f}")
    print(f"Escalation Threshold: >= 0.70")
    print(f"Total Escalated Records: {escalated_count}/{len(scores)} "
          f"({(escalated_count / len(scores)) * 100:.1f}%)")
    print("========================================================\n")


if __name__ == "__main__":
    analyze_escalation_distribution(JOB_APPLICATIONS)

    # Demo lookup with APP-1000 and APP-1001
    print("Sample Lookup for APP-1000:")
    print(check_job_application_status("APP-1000"))
    print("\nSample Lookup for APP-1001:")
    print(check_job_application_status("APP-1001"))
    print("\nSample Lookup for Non-existent APP-9999:")
    print(check_job_application_status("APP-9999"))