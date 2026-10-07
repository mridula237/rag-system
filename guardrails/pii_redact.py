import re, os, sys
from dataclasses import dataclass, field
from typing import Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    print("[pii_redact] Presidio not installed, using regex fallback.")

PRESIDIO_ENTITIES = ["PERSON","EMAIL_ADDRESS","PHONE_NUMBER","CREDIT_CARD","US_SSN","IP_ADDRESS","IBAN_CODE","URL"]
REPLACEMENTS = {"PERSON":"[NAME]","EMAIL_ADDRESS":"[EMAIL]","PHONE_NUMBER":"[PHONE]","CREDIT_CARD":"[CARD]","US_SSN":"[SSN]","IP_ADDRESS":"[IP]","IBAN_CODE":"[IBAN]","URL":"[URL]","DEFAULT":"[REDACTED]"}
REGEX_PATTERNS = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]"),
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b", "[CARD]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP]"),
]

@dataclass
class RedactResult:
    original: str
    redacted: str
    has_pii: bool
    pii_types_found: list = field(default_factory=list)
    redaction_count: int = 0
    engine_used: str = "none"

def _redact_with_presidio(text):
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    results = analyzer.analyze(text=text, entities=PRESIDIO_ENTITIES, language="en", score_threshold=0.3)
    if not results:
        return RedactResult(original=text, redacted=text, has_pii=False, engine_used="presidio")
    operators = {e: OperatorConfig("replace", {"new_value": REPLACEMENTS.get(e, "[REDACTED]")}) for e in set(r.entity_type for r in results)}
    anon = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
    return RedactResult(original=text, redacted=anon.text, has_pii=True,
                        pii_types_found=list(set(r.entity_type for r in results)),
                        redaction_count=len(results), engine_used="presidio")

def _redact_with_regex(text):
    redacted, found, count = text, [], 0
    for pattern, replacement in REGEX_PATTERNS:
        matches = re.findall(pattern, redacted)
        if matches:
            found.append(replacement.strip("[]")); count += len(matches)
            redacted = re.sub(pattern, replacement, redacted)
    return RedactResult(original=text, redacted=redacted, has_pii=count>0,
                        pii_types_found=found, redaction_count=count, engine_used="regex")

def redact_query(text):
    if not text or not text.strip():
        return RedactResult(original=text, redacted=text, has_pii=False)
    if PRESIDIO_AVAILABLE:
        try: return _redact_with_presidio(text)
        except Exception as e: print(f"[pii_redact] Presidio error: {e}")
    return _redact_with_regex(text)

def redact_response(text): return redact_query(text)

def sanitize_for_trace(obj):
    if isinstance(obj, str): return redact_query(obj).redacted
    if isinstance(obj, dict): return {k: sanitize_for_trace(v) for k, v in obj.items()}
    if isinstance(obj, list): return [sanitize_for_trace(i) for i in obj]
    return obj

if __name__ == "__main__":
    tests = [
        ("What are the rate limits?", False),
        ("My email is john.doe@example.com — reset my API key?", True),
        ("Call me at 415-555-1234 about upload limits.", True),
        ("My SSN is 123-45-6789.", True),
        ("What is the max file size?", False),
        ("Server at 192.168.1.100 is hitting the rate limit.", True),
    ]
    print(f"\n{'='*55}\nSTEP 7: PII REDACTION\nEngine: {'Presidio' if PRESIDIO_AVAILABLE else 'Regex'}\n{'='*55}")
    for text, expect in tests:
        r = redact_query(text)
        ok = "✓" if r.has_pii == expect else "✗"
        print(f"\n{ok} Original : {text}")
        print(f"   Redacted : {r.redacted}")
        print(f"   Status   : {'PII' if r.has_pii else 'clean'}  {r.pii_types_found}")
