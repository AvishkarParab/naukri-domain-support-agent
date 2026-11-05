"""
knowledge_base.py - 12 Required Domain Support Policy Documents
Each document satisfies the 2-5 sentence requirement and covers one mandatory topic.
"""

from typing import List, Dict

KNOWLEDGE_BASE_DOCS: List[Dict[str, str]] = [
    {
        "id": "doc_01_eligibility",
        "topic": "job-application eligibility criteria",
        "content": (
            "Candidates applying for open positions on Naukri must possess an accredited degree "
            "or documented professional experience matching role requirements. "
            "All applicants must maintain valid, active legal authorization to work in the hiring jurisdiction. "
            "Submission of falsified employment records or inaccurate qualification claims results in permanent disqualification."
        ),
    },
    {
        "id": "doc_02_scheduling",
        "topic": "interview-scheduling process",
        "content": (
            "Interviews are scheduled through automated candidate portal invites coordinated with hiring panel availability. "
            "Interview rescheduling requests must be submitted at least 24 hours prior to the confirmed calendar slot. "
            "Failure to attend two consecutive scheduled evaluation sessions without prior notification leads to application withdrawal."
        ),
    },
    {
        "id": "doc_03_negotiation",
        "topic": "offer-negotiation policy",
        "content": (
            "Formal compensation offers are calculated strictly within structured departmental salary bands and assessed competency tiers. "
            "Candidates may submit written counter-proposals within five working days of receiving their initial offer letter. "
            "Total discretionary revisions to base compensation cannot exceed 15% above the baseline offer and require executive approval."
        ),
    },
    {
        "id": "doc_04_bgv",
        "topic": "background-verification process",
        "content": (
            "Background verification begins automatically upon written offer acceptance by the selected applicant. "
            "Screening procedures encompass verification of previous employment dates, educational credentials, and criminal background checks. "
            "Unexplained employment variances exceeding 30 days require formal candidate justification or risk immediate offer revocation."
        ),
    },
    {
        "id": "doc_05_notice_period",
        "topic": "notice-period policy",
        "content": (
            "Standard employment agreements stipulate a mandatory ninety-day notice period prior to voluntary release. "
            "Early exit buyouts remain subject to operational discretion and departmental leadership budget authorization. "
            "Employees serving probationary terms are subject to a thirty-day notice period across all job functions."
        ),
    },
    {
        "id": "doc_06_referral_bonus",
        "topic": "referral-bonus policy",
        "content": (
            "Full-time employees referring external candidates are eligible for structured referral bonuses upon candidate hiring. "
            "Referral payments are disbursed in two equal installments after the referee completes 90 and 180 calendar days of service. "
            "Referrals for leadership roles and lateral contractor conversions follow customized compensation schedules."
        ),
    },
    {
        "id": "doc_07_internal_transfer",
        "topic": "internal-transfer eligibility",
        "content": (
            "Employees are eligible for internal mobility transfers only after finishing twelve consecutive months in their present role. "
            "The applicant must hold a performance rating of Meets Expectations or above over the preceding review cycle. "
            "Transfer execution requires formal concurrence from both the releasing manager and the receiving team lead."
        ),
    },
    {
        "id": "doc_08_probation",
        "topic": "probation-period policy",
        "content": (
            "All incoming personnel undergo an initial probationary review window lasting six continuous months. "
            "A mid-probation milestone check occurs at ninety days to evaluate performance trajectory and organizational alignment. "
            "Substandard performance during probation may result in a single ninety-day extension or separation without cause."
        ),
    },
    {
        "id": "doc_09_remote_work",
        "topic": "remote-work eligibility",
        "content": (
            "Remote work privileges depend on department requirements, project demands, and consistent employee performance metrics. "
            "Eligible staff must operate within approved geographical regions featuring secure, enterprise-grade network connectivity. "
            "Management reserves the operational right to recall remote personnel to designated office facilities with 14 days advance notice."
        ),
    },
    {
        "id": "doc_10_diversity",
        "topic": "diversity-hiring guidelines",
        "content": (
            "Hiring committees must maintain balanced gender and background representation throughout candidate evaluation cycles. "
            "Candidate shortlists for senior and leadership positions must include at least two qualified individuals from underrepresented backgrounds. "
            "All active interviewers must refresh their unconscious bias training credentials bi-annually."
        ),
    },
    {
        "id": "doc_11_exit_interview",
        "topic": "exit-interview process",
        "content": (
            "Departing team members participate in a confidential exit interview managed by human resources personnel during their final week. "
            "The employee must return all corporate property and obtain clearance sign-offs prior to receiving their final experience documentation. "
            "Aggregated exit interview metrics are compiled quarterly to assist leadership in ongoing retention strategy planning."
        ),
    },
    {
        "id": "doc_12_retention",
        "topic": "applicant-data-retention policy",
        "content": (
            "Candidate profiles, submitted documentation, and evaluation notes are retained within the database for twenty-four months. "
            "Job seekers retain the statutory right to request expedited deletion of their personal records via written privacy inquiry. "
            "Expired application records are systematically scrubbed and purged in accordance with statutory privacy compliance standards."
        ),
    },
]


def validate_knowledge_base(docs: List[Dict[str, str]]) -> bool:
    print(f"Total Policy Documents: {len(docs)} (Constraint: >= 12)")
    assert len(docs) >= 12, "Knowledge base contains fewer than 12 documents!"

    required_topics = [
        "job-application eligibility criteria",
        "interview-scheduling process",
        "offer-negotiation policy",
        "background-verification process",
        "notice-period policy",
        "referral-bonus policy",
        "internal-transfer eligibility",
        "probation-period policy",
        "remote-work eligibility",
        "diversity-hiring guidelines",
        "exit-interview process",
        "applicant-data-retention policy",
    ]

    existing_topics = [d["topic"] for d in docs]
    for topic in required_topics:
        assert topic in existing_topics, f"Missing required topic: {topic}"

    for d in docs:
        sentences = [s.strip() for s in d["content"].split(".") if s.strip()]
        sentence_count = len(sentences)
        print(f"  - [{d['id']}] {d['topic']}: {sentence_count} sentences")
        assert 2 <= sentence_count <= 5, f"Doc {d['id']} sentence count out of bounds (2-5): {sentence_count}"

    print("\nKnowledge base validation passed all criteria.")
    return True


if __name__ == "__main__":
    validate_knowledge_base(KNOWLEDGE_BASE_DOCS)