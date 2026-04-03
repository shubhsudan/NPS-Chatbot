"""
test_questions.py — 25 gold-standard NPS evaluation questions with ground truths.

Ground truths are concise factual statements drawn directly from PFRDA regulations.
They are used by RAGAS context_recall (did the retrieved chunks cover the answer?)
and context_precision (were the retrieved chunks relevant?).

Coverage:
  - Withdrawal rules (partial, full, premature)
  - Annuity rules
  - Tax treatment (80CCD)
  - Tier I vs Tier II
  - PRAN registration
  - Death/nomination
  - Government vs private sector NPS
  - Fund management and portability
"""

TEST_CASES = [
    # ── Withdrawal ────────────────────────────────────────────────
    {
        "question": "Can I withdraw my entire NPS corpus before retirement at age 60?",
        "ground_truth": (
            "Premature exit before age 60 is allowed after completing 10 years in NPS. "
            "However, the subscriber must use at least 80% of the corpus to purchase an annuity; "
            "only up to 20% can be withdrawn as a lump sum. "
            "If the total corpus is less than ₹2.5 lakh, the entire amount can be withdrawn."
        ),
    },
    {
        "question": "What lump sum amount can I withdraw from NPS at the age of 60?",
        "ground_truth": (
            "At superannuation (age 60), a subscriber can withdraw up to 60% of the NPS corpus "
            "as a tax-free lump sum. The remaining 40% must be used to purchase an annuity. "
            "If the total corpus is less than ₹5 lakh, the entire amount can be withdrawn without "
            "buying an annuity."
        ),
    },
    {
        "question": "What are the conditions for partial withdrawal from NPS Tier I?",
        "ground_truth": (
            "Partial withdrawal from NPS Tier I is permitted after 3 years of subscription. "
            "The amount is limited to 25% of the subscriber's own contributions (excluding employer contributions). "
            "Allowed purposes include: higher education or marriage of children, purchase or construction "
            "of a residential house, treatment of specified critical illnesses, and for persons with "
            "disability. A maximum of 3 partial withdrawals are allowed during the entire tenure."
        ),
    },
    {
        "question": "What happens to my NPS account if I die before reaching age 60?",
        "ground_truth": (
            "On the death of an NPS subscriber before age 60, the entire accumulated corpus is paid "
            "to the nominee or legal heir as a lump sum. The nominee is not required to purchase an annuity. "
            "If no nomination is registered, the corpus is distributed as per succession laws."
        ),
    },

    # ── Annuity ───────────────────────────────────────────────────
    {
        "question": "What is the minimum annuity purchase requirement when exiting NPS at age 60?",
        "ground_truth": (
            "At normal exit (age 60), at least 40% of the NPS corpus must be used to purchase a "
            "life annuity from an Annuity Service Provider (ASP) empanelled by PFRDA. "
            "The remaining 60% can be withdrawn as a lump sum. If the total corpus is ₹5 lakh or less, "
            "the subscriber can withdraw the entire amount without buying an annuity."
        ),
    },
    {
        "question": "Which annuity service providers are empanelled by PFRDA for NPS?",
        "ground_truth": (
            "PFRDA empanels life insurance companies as Annuity Service Providers (ASPs) for NPS. "
            "Empanelled ASPs include LIC of India, SBI Life, HDFC Life, ICICI Prudential Life, "
            "Bajaj Allianz Life, and others approved by PFRDA from time to time. "
            "Subscribers can choose any empanelled ASP and annuity plan that suits them."
        ),
    },

    # ── Tax ───────────────────────────────────────────────────────
    {
        "question": "How is NPS contribution taxed under Section 80CCD(1)?",
        "ground_truth": (
            "Under Section 80CCD(1) of the Income Tax Act, a subscriber's own NPS contribution "
            "is deductible up to 10% of salary (basic + DA) for salaried individuals, "
            "or 20% of gross total income for self-employed persons. "
            "This deduction is within the overall ₹1.5 lakh limit of Section 80C."
        ),
    },
    {
        "question": "What is the additional tax deduction available under Section 80CCD(1B) for NPS?",
        "ground_truth": (
            "Section 80CCD(1B) provides an additional deduction of up to ₹50,000 for contributions "
            "to NPS Tier I. This deduction is over and above the ₹1.5 lakh limit under Section 80C "
            "and 80CCD(1), making the total potential NPS-related deduction ₹2 lakh per year."
        ),
    },
    {
        "question": "Is the employer's NPS contribution taxable in the hands of the employee?",
        "ground_truth": (
            "Employer contributions to NPS are deductible under Section 80CCD(2) of the Income Tax Act. "
            "For private sector employees, the deduction is up to 10% of salary (basic + DA). "
            "For central government employees, the employer contribution deduction limit is 14% of salary. "
            "This deduction is over and above the Section 80C and 80CCD(1B) limits."
        ),
    },
    {
        "question": "Is the lump sum withdrawal from NPS at retirement taxable?",
        "ground_truth": (
            "The lump sum withdrawal of up to 60% of the NPS corpus at retirement (age 60) is completely "
            "tax-free under Section 10(12A) of the Income Tax Act. The annuity income received after "
            "purchasing an annuity is taxable as income from other sources in the year of receipt."
        ),
    },

    # ── Tier I vs Tier II ─────────────────────────────────────────
    {
        "question": "What is the difference between NPS Tier I and Tier II accounts?",
        "ground_truth": (
            "NPS Tier I is a mandatory pension account with restricted withdrawals — the corpus is "
            "locked in until age 60 (with limited partial withdrawals). It qualifies for tax deductions. "
            "NPS Tier II is a voluntary savings account with no lock-in — money can be withdrawn "
            "freely at any time. Tier II does not offer tax benefits for most subscribers "
            "(except central government employees under a specified scheme)."
        ),
    },
    {
        "question": "Is there a minimum contribution requirement for NPS Tier II?",
        "ground_truth": (
            "NPS Tier II requires a minimum contribution of ₹250 per deposit and a minimum of "
            "₹250 at the time of account opening. There is no minimum annual contribution requirement "
            "for Tier II (unlike Tier I, which requires ₹1,000 per year to keep the account active). "
            "A Tier I account must be active before opening a Tier II account."
        ),
    },

    # ── PRAN Registration ─────────────────────────────────────────
    {
        "question": "How do I register for a PRAN number online through eNPS?",
        "ground_truth": (
            "To register for PRAN online through eNPS (enps.nsdl.com): visit the eNPS portal, "
            "select 'National Pension System', choose the appropriate account type (individual/corporate), "
            "fill in personal details, upload KYC documents (Aadhaar, PAN, bank details), "
            "make the initial contribution online, and submit. A PRAN is generated instantly. "
            "Aadhaar-based e-KYC allows paperless registration."
        ),
    },
    {
        "question": "What documents are required to open an NPS account?",
        "ground_truth": (
            "To open an NPS account, the following documents are typically required: "
            "PAN card, Aadhaar card (for e-KYC), a passport-size photograph, cancelled cheque or "
            "bank passbook for bank details, and a filled subscriber registration form. "
            "For NRI subscribers, additional documents like passport and foreign address proof are needed."
        ),
    },

    # ── Government / Private sector ───────────────────────────────
    {
        "question": "Is NPS mandatory for central government employees?",
        "ground_truth": (
            "Yes, NPS is mandatory for all central government employees who joined service on or "
            "after 1 January 2004. These employees are covered under the National Pension System "
            "(Government sector). The government contributes 14% of the employee's basic pay and DA, "
            "while the employee contributes 10%. State government employees are also covered under "
            "NPS as most states have adopted the scheme."
        ),
    },
    {
        "question": "Can private sector employees or self-employed individuals join NPS?",
        "ground_truth": (
            "Yes, NPS is open to all Indian citizens aged 18–70, including private sector employees "
            "and self-employed individuals, under the 'All Citizens Model' (NPS-Lite or corporate NPS). "
            "They can open a Tier I account voluntarily through a Point of Presence (PoP) or online "
            "via eNPS. There is no employer contribution for individual subscribers outside corporate NPS."
        ),
    },

    # ── Contributions & Fund Management ──────────────────────────
    {
        "question": "What is the minimum annual contribution required to keep an NPS Tier I account active?",
        "ground_truth": (
            "The minimum contribution for NPS Tier I is ₹500 per contribution and ₹1,000 per financial year "
            "to keep the account active. If the minimum annual contribution is not made, the account is "
            "frozen and cannot be operated until it is reactivated by paying the due contributions "
            "along with a penalty."
        ),
    },
    {
        "question": "How can I change my NPS fund manager or investment choice?",
        "ground_truth": (
            "NPS subscribers can change their Pension Fund Manager (PFM) once per financial year. "
            "The change can be made online through the NPS CRA portal or through the PoP. "
            "Subscribers can also change the asset allocation (Active Choice or Auto Choice) "
            "and the proportion of equity (E), corporate bonds (C), and government securities (G) "
            "once per financial year for each asset class."
        ),
    },
    {
        "question": "What are the investment choices available under NPS?",
        "ground_truth": (
            "NPS offers two investment choices: Active Choice and Auto Choice (Lifecycle Fund). "
            "Under Active Choice, subscribers allocate their contributions across Equity (E, max 75%), "
            "Corporate Bonds (C), Government Securities (G), and Alternative Assets (A, max 5%). "
            "Under Auto Choice, allocation automatically shifts from higher to lower equity as the "
            "subscriber ages. Three lifecycle fund options exist: Aggressive (LC75), Moderate (LC50), "
            "and Conservative (LC25)."
        ),
    },

    # ── PRAN portability ─────────────────────────────────────────
    {
        "question": "Is PRAN portable if I change jobs or move to a different city?",
        "ground_truth": (
            "Yes, PRAN (Permanent Retirement Account Number) is fully portable across jobs, "
            "sectors, and locations in India. A subscriber retains the same PRAN when switching "
            "employers, moving from government to private sector, or changing residence. "
            "The subscriber needs to submit a transfer request through the new employer or PoP."
        ),
    },

    # ── NRI ───────────────────────────────────────────────────────
    {
        "question": "Can Non-Resident Indians (NRIs) open an NPS account?",
        "ground_truth": (
            "Yes, NRIs who are Indian citizens aged 18–70 can open an NPS account. "
            "Contributions must be made from an NRE or NRO bank account. However, if the subscriber "
            "loses Indian citizenship or becomes a Person of Indian Origin (PIO), the NPS account "
            "will be closed and the corpus will be paid out. OCI card holders are not eligible."
        ),
    },

    # ── Premature exit ────────────────────────────────────────────
    {
        "question": "What is the rule for premature exit from NPS if I have subscribed for less than 10 years?",
        "ground_truth": (
            "If a subscriber exits NPS before completing 10 years of subscription, the entire "
            "corpus must be used to purchase an annuity — no lump sum withdrawal is allowed. "
            "If the total corpus is less than ₹1 lakh, the subscriber can withdraw the entire "
            "amount without buying an annuity."
        ),
    },

    # ── Charges ───────────────────────────────────────────────────
    {
        "question": "What charges are deducted from my NPS account?",
        "ground_truth": (
            "NPS has very low charges: the Central Recordkeeping Agency (CRA) charges a nominal fee "
            "for account maintenance; the Pension Fund Manager (PFM) charges an investment management "
            "fee (currently capped at 0.09% per annum on assets under management); the Point of Presence "
            "(PoP) charges a one-time registration fee and per-transaction charges. "
            "These charges are among the lowest in the Indian pension and mutual fund industry."
        ),
    },

    # ── Nomination ────────────────────────────────────────────────
    {
        "question": "How do I update or add a nominee to my NPS account?",
        "ground_truth": (
            "Nomination in NPS can be updated online through the CRA portal (NSDL or Karvy) "
            "by logging in with the PRAN and password, navigating to the 'Update Nominee' section, "
            "and entering nominee details. Up to 3 nominees can be added with defined percentage shares. "
            "Offline updates can be done by submitting a nomination form to the PoP."
        ),
    },

    # ── Superannuation deferral ───────────────────────────────────
    {
        "question": "Can I continue contributing to NPS after the age of 60?",
        "ground_truth": (
            "Yes, subscribers can defer exit and continue contributing to NPS beyond age 60 up to "
            "age 75. The account continues to earn market-linked returns during this deferral period. "
            "The subscriber can also choose to defer only the annuity purchase while making partial "
            "lump sum withdrawals. Deferral must be opted for before the normal exit date."
        ),
    },
]
