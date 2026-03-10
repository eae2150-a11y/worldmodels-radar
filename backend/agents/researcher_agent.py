import asyncio
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import anthropic
import httpx

from backend.agents.scoring import score_researcher


# ---------------------------------------------------------------------------
# Unicode folding for author name searches
# ---------------------------------------------------------------------------
_UNICODE_MAP = {
    "\u00df": "ss",
    "\u00f6": "oe",
    "\u00fc": "ue",
    "\u00e4": "ae",
    "\u00e9": "e",
    "\u00e8": "e",
    "\u00ea": "e",
    "\u00eb": "e",
    "\u00e0": "a",
    "\u00e2": "a",
    "\u00f4": "o",
    "\u00f9": "u",
    "\u00fb": "u",
    "\u00e7": "c",
    "\u00f1": "n",
}


def _fold_unicode(text: str) -> str:
    for src, dst in _UNICODE_MAP.items():
        text = text.replace(src, dst)
    return text


# ---------------------------------------------------------------------------
# Batching helper
# ---------------------------------------------------------------------------
async def batch_calls(
    items: list,
    func,
    batch_size: int = 2,
    delay: float = 2.0,
) -> list:
    results: list = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_results = await asyncio.gather(
            *[func(item) for item in batch], return_exceptions=True
        )
        results.extend(batch_results)
        if i + batch_size < len(items):
            await asyncio.sleep(delay)
    return results


# ---------------------------------------------------------------------------
# arXiv helpers
# ---------------------------------------------------------------------------
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_arxiv_authors(entry: ET.Element) -> List[str]:
    authors: List[str] = []
    for author_el in entry.findall("atom:author", ARXIV_NS):
        name_el = author_el.find("atom:name", ARXIV_NS)
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())
    return authors


def _parse_arxiv_entries(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    entries: List[Dict[str, Any]] = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        title_el = entry.find("atom:title", ARXIV_NS)
        title = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else ""
        authors = _parse_arxiv_authors(entry)
        abstract_el = entry.find("atom:summary", ARXIV_NS)
        abstract = abstract_el.text.strip().replace("\n", " ") if abstract_el is not None and abstract_el.text else ""
        entries.append({"title": title, "authors": authors, "abstract": abstract})
    return entries


# ---------------------------------------------------------------------------
# Semantic Scholar helpers
# ---------------------------------------------------------------------------
async def _s2_search_author(name: str, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    url = "https://api.semanticscholar.org/graph/v1/author/search"
    resp = await client.get(url, params={"query": name, "limit": 5}, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


async def _s2_get_author(author_id: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    fields = (
        "name,hIndex,citationCount,paperCount,"
        "papers.title,papers.year,papers.citationCount,"
        "papers.externalIds,papers.abstract"
    )
    url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}"
    resp = await client.get(url, params={"fields": fields}, timeout=30.0)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Theme extraction
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "in", "on", "to", "with",
    "from", "by", "is", "are", "was", "were", "at", "as", "it", "its",
    "via", "using", "based", "towards", "through", "into",
}


def _extract_themes(paper_titles: List[str], top_n: int = 5) -> List[str]:
    word_counts: Dict[str, int] = {}
    for title in paper_titles:
        words = re.findall(r"[a-z]{3,}", title.lower())
        for w in words:
            if w not in _STOPWORDS:
                word_counts[w] = word_counts.get(w, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


# ---------------------------------------------------------------------------
# Build a researcher profile dict from an S2 author record
# ---------------------------------------------------------------------------
def _build_profile(author_data: Dict[str, Any], include_full: bool = True) -> Dict[str, Any]:
    papers = author_data.get("papers") or []
    # Sort by citation count descending
    papers_sorted = sorted(papers, key=lambda p: (p.get("citationCount") or 0), reverse=True)
    top_papers_raw = papers_sorted[:5]

    top_papers: List[Dict[str, Any]] = []
    for p in top_papers_raw:
        arxiv_url: Optional[str] = None
        ext_ids = p.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv")
        if arxiv_id:
            arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
        top_papers.append({
            "title": p.get("title", ""),
            "year": p.get("year"),
            "citations": p.get("citationCount", 0),
            "arxivUrl": arxiv_url,
        })

    titles = [p.get("title", "") for p in papers_sorted[:20]]
    abstracts = [p.get("abstract", "") or "" for p in papers_sorted[:20]]
    themes = _extract_themes(titles)

    relevance = score_researcher(titles, themes, abstracts)

    author_id = author_data.get("authorId", "")
    profile: Dict[str, Any] = {
        "name": author_data.get("name", ""),
        "citations": author_data.get("citationCount", 0),
        "hIndex": author_data.get("hIndex", 0),
        "paperCount": author_data.get("paperCount", 0),
        "topPapers": top_papers,
        "relevanceScore": relevance,
        "themes": themes,
        "semanticScholarUrl": f"https://www.semanticscholar.org/author/{author_id}",
    }

    if not include_full:
        # Lighter-weight version: also include topPaper (single)
        if top_papers:
            profile["topPaper"] = top_papers[0]

    return profile


# ---------------------------------------------------------------------------
# Claude summary helpers
# ---------------------------------------------------------------------------
async def _generate_claude_summaries(profile: Dict[str, Any]) -> Dict[str, str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "whyMatters": "Claude API key not configured.",
            "researchSummary": "Claude API key not configured.",
        }

    client = anthropic.AsyncAnthropic(api_key=api_key)

    papers_text = "\n".join(
        f"- {p['title']} ({p.get('year', 'n/a')}, {p.get('citations', 0)} citations)"
        for p in profile.get("topPapers", [])
    )

    prompt = (
        f"Researcher: {profile['name']}\n"
        f"h-index: {profile.get('hIndex', 'unknown')}\n"
        f"Top papers:\n{papers_text}\n\n"
        "1. Write 2-3 sentences about why this researcher matters for physical world models research "
        "(3D scene understanding, neural rendering, embodied AI, spatial reasoning). "
        "Be specific about their contributions.\n"
        "2. Write a 1-2 sentence plain-language summary of what they work on.\n\n"
        "Format your response exactly as:\n"
        "WHY_MATTERS: <text>\n"
        "RESEARCH_SUMMARY: <text>"
    )

    try:
        message = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text

        why_matters = ""
        research_summary = ""
        for line in text.split("\n"):
            if line.startswith("WHY_MATTERS:"):
                why_matters = line[len("WHY_MATTERS:"):].strip()
            elif line.startswith("RESEARCH_SUMMARY:"):
                research_summary = line[len("RESEARCH_SUMMARY:"):].strip()

        return {
            "whyMatters": why_matters or text,
            "researchSummary": research_summary or text,
        }
    except Exception as exc:
        return {
            "whyMatters": f"Error generating summary: {exc}",
            "researchSummary": f"Error generating summary: {exc}",
        }


# ---------------------------------------------------------------------------
# Main lookup function
# ---------------------------------------------------------------------------
_ARXIV_ID_PATTERN = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)$", re.IGNORECASE)


async def lookup_researcher(query: str) -> Dict[str, Any]:
    """Look up a researcher by name or arXiv paper ID."""

    query = query.strip()
    arxiv_match = _ARXIV_ID_PATTERN.match(query)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        if arxiv_match:
            # ------- arXiv ID path -------
            arxiv_id = arxiv_match.group(1)
            arxiv_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
            resp = await client.get(arxiv_url, timeout=30.0)
            resp.raise_for_status()
            entries = _parse_arxiv_entries(resp.text)
            if not entries or not entries[0]["authors"]:
                return {"error": f"No paper found for arXiv ID {arxiv_id}"}
            primary_author = entries[0]["authors"][0]
        else:
            # ------- Name path -------
            primary_author = query

        # -- Name disambiguation --
        folded_name = _fold_unicode(primary_author)
        arxiv_search_url = (
            f"https://export.arxiv.org/api/query?"
            f'search_query=au:"{folded_name}"+AND+'
            f"(cat:cs.CV+OR+cat:cs.LG+OR+cat:cs.AI+OR+cat:cs.GR+OR+cat:cs.RO)"
            f"&max_results=5&sortBy=relevance"
        )
        resp = await client.get(arxiv_search_url, timeout=30.0)
        resp.raise_for_status()
        entries = _parse_arxiv_entries(resp.text)

        # Collect unique author names from arXiv results
        candidate_names: List[str] = []
        seen_lower: set = set()
        for entry in entries:
            for author in entry["authors"]:
                if author.lower() not in seen_lower:
                    seen_lower.add(author.lower())
                    candidate_names.append(author)

        if not candidate_names:
            # Fallback: just use the query name directly
            candidate_names = [primary_author]

        # Search S2 for each candidate name (batched)
        async def _search_s2(name: str) -> List[Dict[str, Any]]:
            try:
                return await _s2_search_author(name, client)
            except Exception:
                return []

        s2_search_results = await batch_calls(candidate_names, _search_s2, batch_size=2, delay=2.0)

        # Collect all unique S2 author IDs
        s2_author_ids: List[str] = []
        seen_ids: set = set()
        for result_list in s2_search_results:
            if isinstance(result_list, list):
                for item in result_list:
                    aid = item.get("authorId")
                    if aid and aid not in seen_ids:
                        seen_ids.add(aid)
                        s2_author_ids.append(aid)

        if not s2_author_ids:
            return {"error": f"No Semantic Scholar profile found for '{query}'"}

        # Get full author data (batched)
        async def _get_s2(aid: str) -> Optional[Dict[str, Any]]:
            try:
                return await _s2_get_author(aid, client)
            except Exception:
                return None

        full_authors = await batch_calls(s2_author_ids, _get_s2, batch_size=2, delay=2.0)

        # Filter out errors/None
        valid_authors = [a for a in full_authors if isinstance(a, dict) and a]

        if not valid_authors:
            return {"error": f"Could not retrieve author data for '{query}'"}

        # Sort by citation count descending
        valid_authors.sort(key=lambda a: a.get("citationCount", 0), reverse=True)

        # Auto-select logic
        best = valid_authors[0]
        best_citations = best.get("citationCount", 0)
        if len(valid_authors) > 1:
            second_citations = valid_authors[1].get("citationCount", 0)
            if best_citations >= 10 * max(second_citations, 1):
                pass  # auto-select best
        # Always use best (highest citations)

        profile = _build_profile(best, include_full=True)

        # Generate Claude summaries
        summaries = await _generate_claude_summaries(profile)
        profile["whyMatters"] = summaries["whyMatters"]
        profile["researchSummary"] = summaries["researchSummary"]

    return profile
