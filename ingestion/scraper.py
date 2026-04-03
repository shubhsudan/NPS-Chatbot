"""
scraper.py — Scrapes HTML pages and PDFs from official NPS/PFRDA sources.

Outputs a list of Document objects saved as JSON to data/raw/documents.json
Each document has: { "text": str, "metadata": { "source_url": str, "title": str, "doc_type": str } }
"""

import json
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup
import pdfplumber

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; NPS-Chatbot-Scraper/1.0; "
        "Academic research project, RIT; contact ss2401@rit.edu)"
    )
}

REQUEST_DELAY = 1.5  # seconds between requests — be polite to government servers

# Official NPS/PFRDA sources (verified working URLs)
HTML_SOURCES = [
    {"url": "https://www.npstrust.org.in/faqs",                       "title": "NPS Trust FAQs",                    "doc_type": "faq"},
    {"url": "https://www.npstrust.org.in/about-nps",                  "title": "About NPS",                         "doc_type": "general"},
    {"url": "https://www.npstrust.org.in/eligibility",                "title": "NPS Eligibility",                   "doc_type": "general"},
    {"url": "https://www.npstrust.org.in/benefits-of-nps",            "title": "NPS Tax Benefits",                  "doc_type": "tax"},
    {"url": "https://www.npstrust.org.in/charges-under-nps",          "title": "NPS Charges",                       "doc_type": "charges"},
    {"url": "https://www.npstrust.org.in/partial-withdrawal",         "title": "NPS Partial Withdrawal",            "doc_type": "withdrawal"},
    {"url": "https://www.npstrust.org.in/normal-exit",                "title": "NPS Normal Exit at 60",             "doc_type": "withdrawal"},
    {"url": "https://www.npstrust.org.in/pre-mature-exit",            "title": "NPS Premature Exit",                "doc_type": "withdrawal"},
    {"url": "https://www.npstrust.org.in/deferment",                  "title": "NPS Deferment after 60",            "doc_type": "withdrawal"},
    {"url": "https://www.npstrust.org.in/unfortunate-death-subscriber","title": "NPS Death of Subscriber",          "doc_type": "withdrawal"},
    {"url": "https://www.npstrust.org.in/open-an-nps-account",        "title": "How to Open NPS Account",          "doc_type": "registration"},
    {"url": "https://www.npstrust.org.in/activate-tier-ii",           "title": "NPS Tier II Activation",           "doc_type": "general"},
    {"url": "https://www.npstrust.org.in/about-nps-vatsalya",         "title": "NPS Vatsalya",                     "doc_type": "general"},
    {"url": "https://www.npstrust.org.in/about-apy",                  "title": "Atal Pension Yojana (APY)",        "doc_type": "general"},
    {"url": "https://npstrust.org.in/circulars",                       "title": "NPS Trust Circulars",             "doc_type": "circular"},
    {"url": "https://npstrust.org.in/annual-reports",                  "title": "NPS Trust Annual Reports",        "doc_type": "general"},
    {"url": "https://npstrust.org.in/index.php/tenders",               "title": "NPS Trust Tenders",               "doc_type": "general"},
    {"url": "https://npstrust.org.in/index.php/rti-0",                 "title": "NPS Trust Right to Information",  "doc_type": "general"},
]

# Official FAQ PDFs directly from NPS Trust — English only
_NPS_BASE = "https://www.npstrust.org.in"
DIRECT_PDF_SOURCES = [
    {"url": f"{_NPS_BASE}/sites/default/files/inline-images/03-Revised-FAQs-for-Central-Government-Sector-CG-and-Central-Autonomous-Bodies-CABs.pdf",
     "title": "FAQs — Central Government Sector", "doc_type": "faq"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-images/04-Revised-FAQs-for-for-State-Government-Sector-SG-and-State-Autonomous-Bodies-SABs.pdf",
     "title": "FAQs — State Government Sector", "doc_type": "faq"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-images/05-All-Citz-Mdl-Faq.pdf",
     "title": "FAQs — All Citizens Model", "doc_type": "faq"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-images/06-FAQs-for-Exit-from-National-Pension-System-by-citizens-including-corporate-sector-subscribers.pdf",
     "title": "FAQs — Exit from NPS (Citizens & Corporate)", "doc_type": "faq"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-images/07-Corp-FAQ.pdf",
     "title": "FAQs — Corporate Sector", "doc_type": "faq"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-images/13-NRI-FAQ.pdf",
     "title": "FAQs — NRI Subscribers", "doc_type": "faq"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-images/15-FAQ-eNPS.pdf",
     "title": "FAQs — eNPS Online Registration", "doc_type": "faq"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-images/17-FAQs-RA-Individual-09012020.pdf",
     "title": "FAQs — Retirement Adviser (Individual)", "doc_type": "faq"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-files/FAQs_Ombudsman_updated.pdf",
     "title": "FAQs — NPS Ombudsman", "doc_type": "faq"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-images/20-Exit-FAQs-CG-CAB.pdf",
     "title": "Exit FAQs — Central Government", "doc_type": "withdrawal"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-images/21-Exit-FAQs-SG-SAB.pdf",
     "title": "Exit FAQs — State Government", "doc_type": "withdrawal"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-images/22-Exit-FAQs-All-Citizen-Model.pdf",
     "title": "Exit FAQs — All Citizens Model", "doc_type": "withdrawal"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-images/23-Exit-FAQs-Corporate.pdf",
     "title": "Exit FAQs — Corporate Sector", "doc_type": "withdrawal"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-files/Partial_Withdrawal_FAQs_0.pdf",
     "title": "FAQs — Partial Withdrawal", "doc_type": "withdrawal"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-files/Final_APY_FAQs_English_28-11-23.pdf",
     "title": "FAQs — Atal Pension Yojana (APY)", "doc_type": "faq"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-files/NPS_Vatsalya_FAQS.pdf",
     "title": "FAQs — NPS Vatsalya", "doc_type": "faq"},
    {"url": f"{_NPS_BASE}/sites/default/files/inline-files/FAQonUPS13092025.pdf",
     "title": "FAQs — Unified Pension Scheme (UPS)", "doc_type": "faq"},

    # -----------------------------------------------------------------------
    # Official Acts & Regulations — npstrust.org.in/act-and-regulations
    # All 51 PDFs across pages 0-5
    # -----------------------------------------------------------------------
    # Page 0
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/exits-and-withdrawals-under-the-national-pension-system-regulations-2015-last-amended-on-16-december-2025.pdf",
     "title": "Exits & Withdrawals Regulations 2015 (amended Dec 2025)", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/GazettePFRDAOperationalisationofUPSNPSAmendmentRegulations2025.pdf",
     "title": "PFRDA — Operationalisation of UPS/NPS Amendment Regulations 2025", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/GazettePFRDAExitandWithdrawalstheNPSAmendmentRegulations2025.pdf",
     "title": "PFRDA — Exit and Withdrawals NPS Amendment Regulations 2025", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/RA21022024.pdf",
     "title": "PFRDA Retirement Adviser Amendment Regulations 2024", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFM2.pdf",
     "title": "PFRDA Pension Fund Amendment Regulations 2024", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/operationalUPS_0.pdf",
     "title": "Unified Pension Scheme (UPS) — Operational Guidelines", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/TrusteeRegulationsAmendment2023.pdf",
     "title": "NPS Trust Regulations Amendment 2023", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/CustodianRegulationsAmendment2023.pdf",
     "title": "PFRDA Custodian Regulations Amendment 2023", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRegulationAmendment2023_0.pdf",
     "title": "PFRDA Pension Fund Regulation Amendment 2023", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Redressal_of_Subscriber_Grievance_Second_Amendment_Regulations_Notification_2023.pdf",
     "title": "PFRDA Subscriber Grievance Second Amendment Regulations 2023", "doc_type": "regulation"},
    # Page 1
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/NPS_Trust_Regulations_Second_Amendment_Gazette_notification_2023.pdf",
     "title": "NPS Trust Regulations Second Amendment 2023", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/SEPF_regulation.pdf",
     "title": "PFRDA SEPF Regulations", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/POP_regulation.pdf",
     "title": "PFRDA Point of Presence (POP) Regulations", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA%20%28NATIONAL%20PENSION%20SYSTEM%20TRUST%29%20%28AMENDMENT%29%20REGULATIONS%2C%202023.pdf",
     "title": "PFRDA NPS Trust Amendment Regulations 2023", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/Notification_Amendment_in_Atal_Pension_Yojana_APY_Change_in_Eligibility_Criteria.pdf",
     "title": "APY — Amendment in Eligibility Criteria Notification", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Grievance_Redressal_Amendment_Regulations_2022.pdf",
     "title": "PFRDA Grievance Redressal Amendment Regulations 2022", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Custodian_of_Securities_Amendment_Regulations_2021.pdf",
     "title": "PFRDA Custodian of Securities Amendment Regulations 2021", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Pension_Fund_Sixth_Amendment_Regulations_2021.pdf",
     "title": "PFRDA Pension Fund Sixth Amendment Regulations 2021", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Point_Of_Presence_Amendment_Regulations_2021.pdf",
     "title": "PFRDA Point of Presence Amendment Regulations 2021", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Central_Record_keeping_Agency_Amendment_Regulations_2021.pdf",
     "title": "PFRDA CRA Amendment Regulations 2021", "doc_type": "regulation"},
    # Page 2
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Exits_And_Withdrawals_Under_the_National_Pension_System_Amendment_Regulations_2021.pdf",
     "title": "PFRDA Exits & Withdrawals Amendment Regulations 2021", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_National_Pension_System_Trust_Amendment_Regulations_2021.pdf",
     "title": "PFRDA NPS Trust Amendment Regulations 2021", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Pension_Fund_Fifth_Amendment_Regulations_2021.pdf",
     "title": "PFRDA Pension Fund Fifth Amendment Regulations 2021", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Pension_Fund_Fourth_Amendment_Regulation_2021.pdf",
     "title": "PFRDA Pension Fund Fourth Amendment Regulations 2021", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Act_2013.pdf",
     "title": "PFRDA Act 2013 — Primary Legislation", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Exits_And_Withdrawals_Under_the_National_Pension_System_Amendment_Regulations_2020.pdf",
     "title": "PFRDA Exits & Withdrawals Amendment Regulations 2020", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_National_Pension_System_Trust_Second_Amendment_Regulations_2020.pdf",
     "title": "PFRDA NPS Trust Second Amendment Regulations 2020", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Central_RecordKeeping_Agency_Second_Amendment_Regulations_2020.pdf",
     "title": "PFRDA CRA Second Amendment Regulations 2020", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/Pension_Fund_Regulatory_and_Development_Authority_Pension_Fund_Third_Amendment_Regulations_2020.pdf",
     "title": "PFRDA Pension Fund Third Amendment Regulations 2020", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/Pension_Fund_Regulatory_and_Development_Authority_Pension_Fund_Second_Amendment_Regulations_2020.pdf",
     "title": "PFRDA Pension Fund Second Amendment Regulations 2020", "doc_type": "regulation"},
    # Page 3
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_National_Pension_System_Trust_First_Amendment_Regulations_2019.pdf",
     "title": "PFRDA NPS Trust First Amendment Regulations 2019", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/Exit_5th_amendment.pdf",
     "title": "PFRDA Exits & Withdrawals Fifth Amendment Regulations", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Central_Recordkeeping_Agency_First_Amendment_Regulations_2018.pdf",
     "title": "PFRDA CRA First Amendment Regulations 2018", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/POP_Regulations_2018.pdf",
     "title": "PFRDA POP Regulations 2018", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/exit_4th_amendment.pdf",
     "title": "PFRDA Exits & Withdrawals Fourth Amendment Regulations", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Exit_Regulation_Third_Amendment.pdf",
     "title": "PFRDA Exits & Withdrawals Third Amendment Regulations", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/Second_Amendment_of_Exit_regulation.pdf",
     "title": "PFRDA Exits & Withdrawals Second Amendment Regulations", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_PFReg2015.pdf",
     "title": "PFRDA Pension Fund Regulations 2015", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_AggreReg_1Amend.pdf",
     "title": "PFRDA Aggregator Regulations First Amendment", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_ExitReg2015.pdf",
     "title": "PFRDA Exits & Withdrawals Regulations 2015 (Base)", "doc_type": "regulation"},
    # Page 4
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Trustee_Bank_Regulations_2015_0.pdf",
     "title": "PFRDA Trustee Bank Regulations 2015", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_CustReg.pdf",
     "title": "PFRDA Custodian Regulations 2015", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Retirement_Adviser_Regulations_2016.pdf",
     "title": "PFRDA Retirement Adviser Regulations 2016", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_POP_Regulations_2015.pdf",
     "title": "PFRDA Point of Presence Regulations 2015", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_PFReg_1amend.pdf",
     "title": "PFRDA Pension Fund Regulations First Amendment", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_POP_Regulations_1Amend.pdf",
     "title": "PFRDA POP Regulations First Amendment", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_NPSTReg2015.pdf",
     "title": "PFRDA NPS Trust Regulations 2015", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_AggreReg2015.pdf",
     "title": "PFRDA Aggregator Regulations 2015", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_Subscriber_Grievance_Regulations_2015.pdf",
     "title": "PFRDA Subscriber Grievance Regulations 2015", "doc_type": "regulation"},
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/PFRDA_CRARegulations2015.pdf",
     "title": "PFRDA CRA Regulations 2015", "doc_type": "regulation"},
    # Page 5
    {"url": "https://npstrust.org.in/sites/default/files/act-and-regulations-documents/Notification%20-%20Amendment%20in%20Atal%20Pension%20Yojana%20%28APY%29%20-%20Change%20in%20Eligibility%20Criteria.pdf",
     "title": "APY — Eligibility Criteria Amendment Notification 2022", "doc_type": "regulation"},

    # -----------------------------------------------------------------------
    # Annual Reports — last 2 years
    # -----------------------------------------------------------------------
    {"url": "https://npstrust.org.in/sites/default/files/annual-reports/Annual_Report_of_NPS_2024-Final.pdf",
     "title": "NPS Trust Annual Report 2023-24", "doc_type": "general"},
    {"url": "https://npstrust.org.in/sites/default/files/annual-reports/V16-Annual_Report_of_NPS_2023.pdf",
     "title": "NPS Trust Annual Report 2022-23", "doc_type": "general"},

    # -----------------------------------------------------------------------
    # RTI Disclosures — Rules, Regulations, Norms (skip salary/directory)
    # -----------------------------------------------------------------------
    {"url": "https://npstrust.org.in/sites/default/files/inline-images/4ibiv-min.pdf",
     "title": "RTI — Norms Set by NPS Trust", "doc_type": "general"},
    {"url": "https://npstrust.org.in/sites/default/files/inline-images/4ibv-min.pdf",
     "title": "RTI — Rules, Regulations, Instructions and Manuals", "doc_type": "general"},
    {"url": "https://npstrust.org.in/sites/default/files/inline-images/4ibvi-min.pdf",
     "title": "RTI — Categories of Documents held by NPS Trust", "doc_type": "general"},
    {"url": "https://npstrust.org.in/sites/default/files/inline-files/RTI-Disclosures-FY2023-24.pdf",
     "title": "RTI Disclosures FY2023-24", "doc_type": "general"},
]

# PFRDA listing pages — text scraped even if PDFs are JS-rendered
PDF_LISTING_SOURCES = [
    {
        "url": "https://www.pfrda.org.in/web/pfrda/regulatory-framework/circulars/active-circulars",
        "title": "PFRDA Active Circulars",
        "doc_type": "circular",
        "base_url": "https://www.pfrda.org.in",
    },
    {
        "url": "https://www.pfrda.org.in/web/pfrda/regulatory-framework/guidelines",
        "title": "PFRDA Guidelines",
        "doc_type": "guideline",
        "base_url": "https://www.pfrda.org.in",
    },
]

MAX_PDFS_PER_SOURCE = 10  # cap per listing page to stay within free-tier storage


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    text: str
    source_url: str
    title: str
    doc_type: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = 20) -> Optional[requests.Response]:
    """GET with retries and rate limiting."""
    for attempt in range(3):
        try:
            time.sleep(REQUEST_DELAY)
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            print(f"  [attempt {attempt + 1}/3] Error fetching {url}: {e}")
            if attempt < 2:
                time.sleep(3)
    return None


def _clean_text(text: str) -> str:
    """Normalise whitespace and remove junk characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # drop non-ASCII (noise from govt sites)
    return text.strip()


# ---------------------------------------------------------------------------
# Scrapers
# ---------------------------------------------------------------------------

def scrape_html_page(source: dict) -> Optional[Document]:
    """Scrape visible text from an HTML page."""
    print(f"Scraping HTML: {source['url']}")
    resp = _get(source["url"])
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove navigation, footer, scripts, styles
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Prefer <main> or <article> content; fall back to <body>
    content_el = soup.find("main") or soup.find("article") or soup.find("body")
    raw_text = content_el.get_text(separator=" ") if content_el else soup.get_text(separator=" ")
    text = _clean_text(raw_text)

    if len(text) < 100:
        print(f"  WARNING: very short text ({len(text)} chars) — page may require JS")
        return None

    return Document(
        text=text,
        source_url=source["url"],
        title=source["title"],
        doc_type=source["doc_type"],
    )


def _extract_pdf_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Find all PDF hrefs in a page."""
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            if href.startswith("http"):
                links.append(href)
            else:
                # Relative URL
                href = href.lstrip("/")
                links.append(f"{base_url}/{href}")
    return links


def scrape_pdf_from_url(pdf_url: str, title: str, doc_type: str) -> Optional[Document]:
    """Download a PDF and extract its text with pdfplumber."""
    print(f"  Downloading PDF: {pdf_url}")
    resp = _get(pdf_url)
    if resp is None:
        return None

    # Save PDF locally so we can open it with pdfplumber
    safe_name = re.sub(r"[^\w]", "_", pdf_url.split("/")[-1])[:80]
    pdf_path = RAW_DATA_DIR / safe_name
    pdf_path.write_bytes(resp.content)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
        text = _clean_text(" ".join(pages_text))
    except Exception as e:
        print(f"  ERROR reading PDF {pdf_url}: {e}")
        return None

    if len(text) < 100:
        print(f"  WARNING: PDF yielded very short text — skipping")
        return None

    return Document(
        text=text,
        source_url=pdf_url,
        title=title,
        doc_type=doc_type,
    )


def scrape_pdf_listing_page(source: dict) -> list[Document]:
    """Scrape page text + follow PDF links from a PFRDA listing page."""
    print(f"Scraping PDF listing page: {source['url']}")
    docs = []

    resp = _get(source["url"])
    if resp is None:
        return docs

    soup = BeautifulSoup(resp.text, "html.parser")

    # Also capture text from the listing page itself
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    page_text = _clean_text(soup.get_text(separator=" "))
    if len(page_text) >= 100:
        docs.append(Document(
            text=page_text,
            source_url=source["url"],
            title=source["title"],
            doc_type=source["doc_type"],
        ))

    # Follow PDF links
    pdf_links = _extract_pdf_links(soup, source["base_url"])
    print(f"  Found {len(pdf_links)} PDF links — fetching up to {MAX_PDFS_PER_SOURCE}")
    for pdf_url in pdf_links[:MAX_PDFS_PER_SOURCE]:
        doc = scrape_pdf_from_url(
            pdf_url,
            title=f"{source['title']} — {pdf_url.split('/')[-1]}",
            doc_type=source["doc_type"],
        )
        if doc:
            docs.append(doc)

    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_HINDI_MARKERS = re.compile(r"hindi|hind", re.IGNORECASE)


def scrape_local_pdfs(folder: Path) -> list[Document]:
    """
    Read all English PDFs from data/local_docs/.
    Hindi files (detected by filename) are skipped automatically.
    """
    docs = []
    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        return docs

    print(f"\nReading local PDFs from {folder}...")
    for pdf_path in pdf_files:
        if _HINDI_MARKERS.search(pdf_path.name):
            print(f"  SKIP (Hindi): {pdf_path.name}")
            continue
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_text = [p.extract_text() for p in pdf.pages if p.extract_text()]
            text = _clean_text(" ".join(pages_text))
            if len(text) < 100:
                print(f"  SKIP (too short): {pdf_path.name}")
                continue
            title = pdf_path.stem.replace("_", " ").replace("-", " ").title()
            docs.append(Document(
                text=text,
                source_url=f"local://{pdf_path.name}",
                title=title,
                doc_type="faq",
            ))
            print(f"  OK ({len(text):,} chars): {pdf_path.name}")
        except Exception as e:
            print(f"  ERROR: {pdf_path.name}: {e}")
    return docs


LOCAL_DOCS_DIR = Path(__file__).parent.parent / "data" / "local_docs"


def run_scraper() -> list[Document]:
    """Run all scrapers and return the combined document list."""
    all_docs: list[Document] = []

    # HTML pages
    for source in HTML_SOURCES:
        doc = scrape_html_page(source)
        if doc:
            all_docs.append(doc)

    # Direct PDF downloads (FAQ PDFs + official Acts & Regulations)
    print(f"\nDownloading {len(DIRECT_PDF_SOURCES)} direct PDFs (FAQs + Regulations)...")
    for source in DIRECT_PDF_SOURCES:
        doc = scrape_pdf_from_url(source["url"], source["title"], source["doc_type"])
        if doc:
            all_docs.append(doc)

    # Local PDFs — all PDFs in data/local_docs/ (official FAQs + any regulation docs)
    LOCAL_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    local_docs = scrape_local_pdfs(LOCAL_DOCS_DIR)
    all_docs.extend(local_docs)

    # PDF listing pages (PFRDA — captures page text + any discoverable PDFs)
    for source in PDF_LISTING_SOURCES:
        docs = scrape_pdf_listing_page(source)
        all_docs.extend(docs)

    print(f"\nTotal documents scraped: {len(all_docs)}")

    # Persist to disk
    output_path = RAW_DATA_DIR / "documents.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in all_docs], f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")

    return all_docs


if __name__ == "__main__":
    run_scraper()
