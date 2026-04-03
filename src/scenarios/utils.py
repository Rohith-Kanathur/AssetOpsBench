import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import logging

_log = logging.getLogger(__name__)

def fetch_arxiv_studies(search_queries: str | list[str], max_results_per_query: int = 2, metadata_out: dict = None) -> str:
    """Fetch relevant studies from ArXiv using their public API.
    
    If search_queries is a list, it will execute each search and combine unique results.
    """
    import time
    import ssl

    if isinstance(search_queries, str):
        queries = [search_queries]
    else:
        queries = search_queries

    base_url = "http://export.arxiv.org/api/query?"
    headers = {'User-Agent': 'AssetOpsBench/1.0 (mailto:admin@example.com)'}
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    if metadata_out is not None:
        metadata_out['queries'] = queries
        metadata_out['status_codes'] = []
        metadata_out['returned_entries'] = 0
        metadata_out['pdf_urls'] = []
        metadata_out['query_to_pdf'] = {}

    seen_ids = set()
    studies_text = []

    for i, q in enumerate(queries):
        # Respect ArXiv 3-second rule if doing multiple queries
        if i > 0:
            time.sleep(3.1)

        safe_query = urllib.parse.quote(q)
        url = f"{base_url}search_query={safe_query}&start=0&max_results={max_results_per_query}"
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
                status = response.getcode()
                if metadata_out is not None:
                    metadata_out['status_codes'].append(status)
                data = response.read()
            
            root = ET.fromstring(data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                arxiv_id = entry.find('atom:id', ns).text
                if arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)

                title = entry.find('atom:title', ns)
                summary = entry.find('atom:summary', ns)
                
                t_text = title.text.strip().replace('\n', ' ') if title is not None else "No Title"
                s_text = summary.text.strip().replace('\n', ' ') if summary is not None else "No Summary"
                
                pdf_url = None
                for link in entry.findall('atom:link', ns):
                    if link.attrib.get('title') == 'pdf' or link.attrib.get('type') == 'application/pdf':
                        pdf_url = link.attrib.get('href')
                        if pdf_url and not pdf_url.endswith('.pdf'):
                            pdf_url += '.pdf'
                        break
                        
                pdf_text = ""
                if pdf_url:
                    try:
                        import io
                        from pypdf import PdfReader
                        
                        # PDF extraction also has a 3-second rule
                        time.sleep(3.2)
                        
                        pdf_req = urllib.request.Request(pdf_url, headers=headers)
                        with urllib.request.urlopen(pdf_req, timeout=15, context=ctx) as pdf_resp:
                            pdf_bytes = pdf_resp.read()
                            
                        reader = PdfReader(io.BytesIO(pdf_bytes))
                        extracted = []
                        for j, page in enumerate(reader.pages):
                            if j > 4: break
                            page_text = page.extract_text()
                            if page_text:
                                extracted.append(page_text)
                                
                        pdf_text = "\n".join(extracted)
                        if len(pdf_text) > 10000:
                            pdf_text = pdf_text[:10000] + "\n...[TRUNCATED]"
                    except Exception as e:
                        _log.warning(f"Failed to fetch or parse PDF from {pdf_url}: {e}")
                        pdf_text = f"[PDF Extraction Failed: {e}]"
                
                if pdf_url and pdf_text and "[PDF Extraction Failed" not in pdf_text:
                    if metadata_out is not None:
                        metadata_out['pdf_urls'].append(pdf_url)
                        if q not in metadata_out['query_to_pdf']:
                            metadata_out['query_to_pdf'][q] = []
                        metadata_out['query_to_pdf'][q].append(pdf_url)
                
                studies_text.append(f"Title: {t_text}\nSummary: {s_text}\n\nPDF Content Extracted (First 5 pages):\n{pdf_text}")

        except urllib.error.HTTPError as e:
            if metadata_out is not None:
                metadata_out['status_codes'].append(e.code)
            _log.warning(f"HTTP Error {e.code} fetching ArXiv query '{q}': {e}")
        except Exception as e:
            _log.warning(f"Failed to fetch ArXiv query '{q}': {e}")

    if metadata_out is not None:
        metadata_out['returned_entries'] = len(studies_text)
        
    return "\n\n".join(studies_text) if studies_text else "No recent studies found via ArXiv."



def fetch_hf_fewshot(dataset_id: str = "ibm-research/AssetOpsBench", split: str = "scenarios", target_type: str = None, fallback_if_missing: bool = True) -> list[dict]:
    """Fetch few-shot scenarios from HuggingFace dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_id, split)
        
        examples = []
        if "train" in ds:
            train_ds = ds["train"]
        else:
            train_ds = ds
            
        for item in train_ds:
            if target_type:
                if str(item.get("type", "")).lower() == target_type.lower():
                    examples.append(item)
            else:
                examples.append(item)
                
            if len(examples) >= 3:
                break
                
        # Return only the relevant keys to keep context size manageable
        clean_examples = []
        for e in examples:
            clean_examples.append({
                "text": e.get("text", ""),
                "category": e.get("category", ""),
                "characteristic_form": e.get("characteristic_form", "")
            })
            
        return clean_examples
    except ImportError:
        _log.warning("HuggingFace 'datasets' library is not installed. Returning empty few-shots.")
        if fallback_if_missing:
            return [{"text": "Mock query", "category": "Knowledge Query", "characteristic_form": "Mock characteristic"}]
        return []
    except Exception as e:
        _log.warning(f"Failed to load HuggingFace dataset: {e}")
        if fallback_if_missing:
            return [{"text": "Mock query", "category": "Knowledge Query", "characteristic_form": f"Mock. Error: {e}"}]
        return []
