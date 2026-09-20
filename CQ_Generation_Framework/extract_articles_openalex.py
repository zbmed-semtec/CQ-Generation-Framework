import re
import json
import spacy
import datetime
import io
import requests
import tiktoken
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import pandas as pd
from newspaper import Article
from serpapi import GoogleSearch
from langdetect import detect
from urllib.parse import urlparse
from pathlib import Path
from utils import load_environment_variables, initialize_clients

load_environment_variables()
deployment_name, serpapi_api_key = initialize_clients()

# ========== Setup & Helpers ==========
MIN_TEXT_CHARS = 1500
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

nlp = spacy.load("en_core_web_sm")

def lemmatized_tokens(text: str, max_chars: int = 8000) -> set:
    doc = nlp(text[:max_chars].lower())
    return {token.lemma_ for token in doc if token.is_alpha and not token.is_stop}

def estimate_tokens(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def normalize_paragraphs(text: str) -> list[str]:
    t = text.replace("\r\n", "\n")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    paras = [p.strip() for p in re.split(r"\n{2,}", t) if p.strip()]
    return paras

# ========== Domain config ==========
PUBLISHER_SITES = ["site:springer.com"]
APPROVED_DOMAINS = ["springer.com"]

def is_english(text: str, min_chars: int = 300) -> bool:
    try:
        sample = text if len(text) <= 2000 else text[:2000]
        if len(sample) < min_chars:
            return True
        return detect(sample) == "en"
    except Exception:
        return True

# ========== Load domain information from JSON ==========
BASE_DIR = Path(__file__).resolve().parent

def load_domain_config(
    config_path: str = "json_input/domain-info.json"
) -> Dict:
    json_path = BASE_DIR / config_path

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
        return {}

DOMAIN_CONFIG = load_domain_config()

MAIN_DOMAIN_NAME = DOMAIN_CONFIG.get("MAIN_DOMAIN_NAME", "Unknown Domain")
TOPIC_TERMS = DOMAIN_CONFIG.get("TOPIC_TERMS", [])
FILTER_KEYWORDS = DOMAIN_CONFIG.get("FILTER_KEYWORDS", [])
MAIN_DOMAIN_WORDS = DOMAIN_CONFIG.get("MAIN_DOMAIN_WORDS", [])
COMPOUND_GENERAL_TERMS = DOMAIN_CONFIG.get("COMPOUND_GENERAL_TERMS", [])
ONTOLOGY_COVERAGE_AREAS = DOMAIN_CONFIG.get("ONTOLOGY_COVERAGE_AREAS", [])

all_items = []
for key, value in DOMAIN_CONFIG.items():
    all_items.append(key)
    if isinstance(value, list):
        all_items.extend(value)
    else:
        all_items.append(str(value))
scope_text = "\n".join(all_items)

# ========== Save Article Summary ==========
def save_article_summary(
        articles: List[Dict],
        token_count: int,
        output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("ARTICLE COLLECTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Articles Collected: {len(articles)}\n")
        f.write(f"Total Input Tokens: {token_count}\n")
        f.write(
            f"Generated on: "
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write("ARTICLES DETAILS:\n")
        f.write("-" * 30 + "\n\n")
        for i, article in enumerate(articles, 1):
            f.write(f"Article {i}:\n")
            f.write(f"  Title: {article['title']}\n")
            f.write(f"  URL: {article['url']}\n")
            f.write(f"  Source: {article.get('source', 'Unknown')}\n")
            f.write(f"  DOI: {article.get('doi') or 'N/A'}\n")
            f.write(f"  Text Length: {len(article['text'])} characters\n")
            lower_text = article['text'].lower()
            keyword_count = sum(
                1 for k in FILTER_KEYWORDS if k in lower_text)
            f.write(f"  Filter Keywords Found: {keyword_count}\n\n")

# ========== Fetching Articles ==========
def build_scholar_queries() -> List[str]:
    queries = []
    for site in PUBLISHER_SITES:
        for topic in TOPIC_TERMS:
            q = (f'("{topic}") '
                 f'({COMPOUND_GENERAL_TERMS[0]} OR '
                 f'{COMPOUND_GENERAL_TERMS[1]}) {site}')
            queries.append(q)
    for site in PUBLISHER_SITES:
        queries.append(
            f'{COMPOUND_GENERAL_TERMS[0]} OR '
            f'{COMPOUND_GENERAL_TERMS[1]} {site} (pdf OR "open access")'
        )
    return queries[:21]

def scholar_search(query: str, start: int = 0) -> List[Dict]:
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": serpapi_api_key,
        "start": start,
        "num": 10,
        "hl": "en",
    }
    search = GoogleSearch(params)
    result = search.get_dict()
    return result.get("organic_results", []) or []


def openalex_search(query: str, per_page: int = 25, pages: int = 2) -> List[Dict]:
    """
    Search OpenAlex for works matching the query.

    OpenAlex is used as a second discovery source. It provides metadata
    and open-access locations; the actual full text is downloaded later
    by the common PDF/HTML extraction pipeline.
    """
    url = "https://api.openalex.org/works"
    results = []

    for page in range(1, pages + 1):
        params = {
            "search": query,
            "per-page": per_page,
            "page": page,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[x] OpenAlex search failed: {e}")
            break

        page_results = data.get("results", []) or []
        results.extend(page_results)

        if len(page_results) < per_page:
            break

    return results


def get_openalex_url(work: Dict) -> Optional[str]:
    """
    Return the best available full-text URL from an OpenAlex work.

    Prefer a PDF URL; otherwise use an open-access/primary landing page.
    """
    best_oa = work.get("best_oa_location") or {}
    primary = work.get("primary_location") or {}

    for location in (best_oa, primary):
        if location.get("pdf_url"):
            return location["pdf_url"]

    for location in (best_oa, primary):
        if location.get("landing_page_url"):
            return location["landing_page_url"]

    return None


def get_openalex_doi(work: Dict) -> Optional[str]:
    doi = work.get("doi")
    if doi:
        return doi.lower().replace("https://doi.org/", "").strip()

    ids = work.get("ids") or {}
    doi = ids.get("doi")
    if doi:
        return doi.lower().replace("https://doi.org/", "").strip()

    return None


def get_openalex_year(work: Dict) -> Optional[int]:
    year = work.get("publication_year")
    if isinstance(year, int):
        return year

    return None


def build_openalex_queries() -> List[str]:
    """
    Build OpenAlex queries from the same topic/general terms used by
    Google Scholar, but without the Springer site restriction.
    """
    queries = []

    for topic in TOPIC_TERMS:
        q = (
            f'"{topic}" '
            f'({COMPOUND_GENERAL_TERMS[0]} OR '
            f'{COMPOUND_GENERAL_TERMS[1]})'
        )
        queries.append(q)

    # Add a broader query so OpenAlex can discover relevant works
    # that do not contain an exact TOPIC_TERM phrase.
    queries.append(
        f'({COMPOUND_GENERAL_TERMS[0]} OR '
        f'{COMPOUND_GENERAL_TERMS[1]})'
    )

    return queries[:21]

def get_pdf_url_from_result(res: Dict) -> Optional[str]:
    resources = res.get("resources") or []
    for r in resources:
        if r.get("file_format", "").lower() == "pdf" and r.get("link"):
            return r["link"]
    link = res.get("link")
    if link and link.lower().endswith(".pdf"):
        return link
    return None

# ========== Content extraction ==========
def try_download(
        url: str,
        timeout: int = 25) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(
            url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code == 200:
            return r.content, (r.headers.get("Content-Type") or "").lower()
    except Exception as e:
        print(f"[x] Download failed: {url} ({e})")
    return None, None

def looks_like_pdf(
        content: Optional[bytes],
        content_type: Optional[str]) -> bool:
    if content_type and "application/pdf" in content_type:
        return True
    if content and content[:4] == b"%PDF":
        return True
    return False

def extract_pdf_text(pdf_bytes: bytes) -> str:
    from pdfminer.high_level import extract_text
    try:
        text = extract_text(io.BytesIO(pdf_bytes))
        return text or ""
    except Exception as e:
        print(f"[x] PDF parse failed: {e}")
        return ""

ALLOW_HTML_FALLBACK = True

def extract_article_text_from_url(url: str) -> Tuple[str, str]:
    content, ctype = try_download(url)
    if content and looks_like_pdf(content, ctype) and len(content) > 2048:
        text = extract_pdf_text(content).strip()
        if not text:
            print(f"[!] Skipped PDF (empty after parse): {url}")
            return "", ""
        if len(text) < MIN_TEXT_CHARS:
            print(
                f"[!] Skipped PDF (too short: {len(text)} chars): {url}")
            return "", ""
        return "", text
    if not ALLOW_HTML_FALLBACK:
        print(f"[!] Skipped non-PDF or invalid PDF: {url}")
        return "", ""
    try:
        art = Article(url)
        art.download()
        art.parse()
        title = (art.title or "").strip()
        text = art.text or ""
        if len(text) >= MIN_TEXT_CHARS:
            print(f"[~] Using fallback HTML: {url}")
            return title, text
        else:
            print(
                f"[!] Skipped HTML (too short: {len(text)} chars): {url}")
    except Exception as e:
        print(f"[x] Newspaper parse failed: {url} ({e})")
    return "", ""

# ========== Pipeline: fetch articles ==========
def fetch_fulltext_articles(required_count: int = 30) -> List[Dict]:
    """
    Collect relevant full-text articles from two discovery sources:

        Google Scholar -> Springer only
        OpenAlex       -> Open-access / available locations

    Both sources then use the same:
        deduplication -> PDF/HTML extraction -> content filters

    Deduplication is performed using normalized URLs and DOI where
    available.
    """
    collected: List[Dict] = []
    seen_urls = set()
    seen_dois = set()
    covered_terms = set()

    scholar_queries = build_scholar_queries()
    openalex_queries = build_openalex_queries()
    current_year = datetime.datetime.now().year

    def normalize_url(url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        return url.split("#")[0].rstrip("/").lower()

    def already_seen(url: Optional[str], doi: Optional[str]) -> bool:
        normalized = normalize_url(url)

        if normalized and normalized in seen_urls:
            return True

        if doi and doi in seen_dois:
            return True

        return False

    def mark_seen(url: Optional[str], doi: Optional[str]) -> None:
        normalized = normalize_url(url)

        if normalized:
            seen_urls.add(normalized)

        if doi:
            seen_dois.add(doi)

    def process_candidate(
        target_url: str,
        title: str,
        text: str,
        source: str,
        doi: Optional[str],
        fallback_title: str = "",
    ) -> bool:
        """
        Apply the existing content filters to a downloaded article.
        Returns True when the article is added.
        """
        nonlocal collected, covered_terms

        if not text or len(text) <= MIN_TEXT_CHARS:
            print(f"[FILTER] Too short/no text: {target_url}")
            return False

        lower = text.lower()

        if not any(term.lower() in lower for term in MAIN_DOMAIN_WORDS):
            print(f"[FILTER] No MAIN_DOMAIN_WORDS match: {target_url}")
            return False

        keyword_count = sum(
            1 for k in FILTER_KEYWORDS if k.lower() in lower
        )

        if keyword_count < 5:
            print(
                f"[FILTER] Fewer than 5 FILTER_KEYWORDS "
                f"({keyword_count}): {target_url}"
            )
            return False

        if not is_english(text):
            print(f"[FILTER] Non-English: {target_url}")
            return False

        article_lemmas = lemmatized_tokens(text)
        matched_terms = []

        for term in TOPIC_TERMS:
            term_lemmas = lemmatized_tokens(term)

            if term_lemmas and all(
                t in article_lemmas
                for t in term_lemmas
            ):
                matched_terms.append(term)

        if len(matched_terms) < 3:
            print(
                f"[FILTER] Fewer than 3 TOPIC_TERMS "
                f"({len(matched_terms)}): {target_url}"
            )
            return False

        covered_terms.update(matched_terms)

        final_title = title or fallback_title or "Untitled Article"

        collected.append({
            "title": final_title,
            "url": target_url,
            "text": text,
            "source": source,
            "doi": doi,
        })

        print(
            f"[+] Article added [{source}]: {final_title} "
            f"({target_url}) - Terms matched: {matched_terms}"
        )

        return True

    def process_scholar_result(r: Dict) -> None:
        if len(collected) >= required_count:
            return

        year = None
        pub_info = r.get("publication_info") or {}

        if isinstance(pub_info, dict):
            year = pub_info.get("year")

            if not year and "summary" in pub_info:
                m = re.search(
                    r"\b(19|20)\d{2}\b",
                    str(pub_info["summary"])
                )
                if m:
                    year = int(m.group(0))

        elif isinstance(pub_info, str):
            m = re.search(r"\b(19|20)\d{2}\b", pub_info)
            if m:
                year = int(m.group(0))

        if not year:
            for field in [
                r.get("title", ""),
                r.get("snippet", ""),
            ]:
                m = re.search(r"\b(19|20)\d{2}\b", str(field))
                if m:
                    year = int(m.group(0))
                    break

        if year and year < current_year - 15:
            print(
                f"[i] Skipped old article ({year}): "
                f"{r.get('title')}"
            )
            return

        pdf_url = get_pdf_url_from_result(r)
        target_url = pdf_url or r.get("link")

        if not target_url:
            return

        normalized = normalize_url(target_url)

        if already_seen(target_url, None):
            return

        domain = urlparse(target_url).netloc.lower()

        # Google Scholar remains restricted to Springer.
        if not any(
            domain.endswith(a)
            for a in APPROVED_DOMAINS
        ):
            print(
                f"[FILTER] Scholar URL not in approved Springer "
                f"domains: {target_url}"
            )
            return

        if "scopus.com" in target_url.lower():
            return

        mark_seen(target_url, None)

        print(f"\n[Scholar] {r.get('title')}")
        print(f"       URL: {target_url}")

        title, text = extract_article_text_from_url(target_url)

        # If the PDF was downloaded but did not provide a title,
        # try the Scholar landing page for the title only.
        if (
            not title
            and pdf_url
            and r.get("link")
            and r["link"] != target_url
        ):
            try:
                a2 = Article(r["link"])
                a2.download()
                a2.parse()
                title = (a2.title or "").strip()
            except Exception:
                pass

        process_candidate(
            target_url=target_url,
            title=title,
            text=text,
            source="Google Scholar",
            doi=None,
            fallback_title=r.get("title", ""),
        )

    def process_openalex_result(work: Dict) -> None:
        if len(collected) >= required_count:
            return

        year = get_openalex_year(work)

        if year and year < current_year - 15:
            print(
                f"[i] Skipped old OpenAlex article ({year}): "
                f"{work.get('title')}"
            )
            return

        target_url = get_openalex_url(work)
        doi = get_openalex_doi(work)

        if not target_url:
            print(
                f"[i] OpenAlex result has no usable full-text URL: "
                f"{work.get('title')}"
            )
            return

        if already_seen(target_url, doi):
            return

        # OpenAlex is intentionally not restricted to Springer here.
        # It is used as a broad second discovery source.
        mark_seen(target_url, doi)

        print(f"\n[OpenAlex] {work.get('title')}")
        print(f"       URL: {target_url}")
        print(f"       DOI: {doi}")

        title, text = extract_article_text_from_url(target_url)

        process_candidate(
            target_url=target_url,
            title=title,
            text=text,
            source="OpenAlex",
            doi=doi,
            fallback_title=work.get("title", ""),
        )

    # ================================================================
    # 1. GOOGLE SCHOLAR -> SPRINGER
    # ================================================================
    print("\n========== GOOGLE SCHOLAR / SPRINGER ==========\n")

    for qi, q in enumerate(scholar_queries):
        if len(collected) >= required_count:
            break

        print(
            f"[Scholar Q{qi + 1}/{len(scholar_queries)}] {q}"
        )

        for start in (0, 10, 20, 30):
            if len(collected) >= required_count:
                break

            results = scholar_search(q, start=start)

            if not results:
                print(
                    f"[i] No Scholar results for start={start}"
                )
                continue

            for r in results:
                if len(collected) >= required_count:
                    break

                process_scholar_result(r)

    # ================================================================
    # 2. OPENALEX -> BROAD / OPEN ACCESS DISCOVERY
    # ================================================================
    print("\n========== OPENALEX ==========\n")

    for qi, q in enumerate(openalex_queries):
        if len(collected) >= required_count:
            break

        print(
            f"[OpenAlex Q{qi + 1}/{len(openalex_queries)}] {q}"
        )

        results = openalex_search(
            q,
            per_page=25,
            pages=2,
        )

        print(
            f"[i] OpenAlex returned {len(results)} results"
        )

        for work in results:
            if len(collected) >= required_count:
                break

            process_openalex_result(work)

    print(
        f"\n[i] Final collection: {len(collected)} articles"
    )
    print(
        f"[i] Final coverage: {len(covered_terms)} / "
        f"{len(TOPIC_TERMS)} topic terms"
    )

    if len(collected) >= required_count:
        print(
            f"[+] Required article count reached: "
            f"{required_count}"
        )

    return collected

# ========== Snippet filtering ==========
def filter_snippets(text: str, keywords: list[str]) -> list[str]:
    paras = normalize_paragraphs(text)
    out = []
    kws = [k.lower() for k in keywords]
    for p in paras:
        p_low = p.lower()
        if len(p) < 100:
            continue
        if any(k in p_low for k in kws):
            out.append(p)
    return out

# ========== Main ==========
def run_extraction() -> Path:
    """
    Fetch articles, extract snippets, save to Excel.
    Returns path to saved Excel file.
    """
    print("Fetching full-text scholarly articles...")
    articles = fetch_fulltext_articles(required_count=30)

    if not articles:
        print("No articles collected. Exiting.")
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"llm_input_springer_{timestamp}.xlsx"
    summary_path = OUTPUT_DIR / f"articles_summary_{timestamp}.txt"

    article_data = []
    snippet_table = {}
    for idx, a in enumerate(articles):
        print(f"\nArticle {idx + 1}: {a['title']}\nURL: {a['url']}")
        snippets = filter_snippets(a["text"], FILTER_KEYWORDS)
        article_data.append({"title": a["title"], "snippets": snippets})
        snippet_table[a["title"]] = snippets

    # Build payload for token count estimation
    max_per_article = 3
    BUDGET = 50000
    payload = ""
    max_len = max((len(a["snippets"]) for a in article_data), default=0)
    stop = False
    for i in range(max_len):
        for a in article_data:
            if i < len(a["snippets"]) and i < max_per_article:
                snippet = a["snippets"][i]
                if estimate_tokens(payload + "\n\n" + snippet) > BUDGET:
                    stop = True
                    break
                payload += "\n\n" + snippet
        if stop:
            break

    token_count = estimate_tokens(payload)
    print(f"Estimated input token count: {token_count}")
    save_article_summary(articles, token_count, summary_path)

    # Save snippets to Excel
    max_rows = max(
        (len(a["snippets"]) for a in article_data), default=0)
    snippet_df = pd.DataFrame()
    for a in article_data:
        padded = a["snippets"] + [""] * (max_rows - len(a["snippets"]))
        snippet_df[a["title"]] = padded

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        snippet_df.to_excel(
            writer, sheet_name="Snippets", index=False)

    print(f"\n[INFO] Snippets saved to: {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    run_extraction()