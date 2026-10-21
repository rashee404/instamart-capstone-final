"""Part C - support-ticket triage with guardrails, redaction, retries, output validation, budget and routing."""
import json
import logging
import random
import re
import sys
import time

import pandas as pd

from common import OUT, load_tickets   # also puts the repository root on sys.path
import mock_llm
from mock_llm import TransientLLMError, est_tokens

CATEGORIES = ["billing", "delivery", "technical", "fraud"]
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05

# the three patterns and placeholders are fixed by the brief and applied in this order
REDACTIONS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
              ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
              ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

ZERO_SHOT = ('You are a support ticket router. CLASSIFY the customer ticket into exactly one category: '
             'billing, delivery, technical, fraud. Reply with JSON only, in the form '
             '{"category": "...", "reason": "..."}.')
FEW_SHOT = ZERO_SHOT + '''
EXAMPLES:
Ticket: The invoice shows a wrong amount for my order. -> {"category": "billing", "reason": "wrong bill"}
Ticket: Someone hacked my account and used my wallet balance. -> {"category": "fraud", "reason": "unauthorized use"}
Ticket: The app freezes when I open the cart. -> {"category": "technical", "reason": "app problem"}
Ticket: The rider has not arrived and my order is late. -> {"category": "delivery", "reason": "late order"}'''

log = logging.getLogger("triage")
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    fmt = logging.Formatter("%(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "retry_log.txt", mode="w")):
        h.setFormatter(fmt)
        log.addHandler(h)


# ------------------------------------------------------------------ redaction
def redact(text):
    for _, pat, ph in REDACTIONS:
        text = pat.sub(ph, text)
    return text


def count_redactions(texts):
    counts = {name: 0 for name, _, _ in REDACTIONS}
    for t in texts:
        for name, pat, ph in REDACTIONS:
            t, n = pat.subn(ph, t)
            counts[name] += n
    return counts


def contains_pii(text):
    return any(pat.search(text) for _, pat, _ in REDACTIONS)


# ------------------------------------------------------------------ resilient model call
class Stats:
    def __init__(self):
        self.transient_errors = 0
        self.waits = []
        self.prompts = []          # every message list actually sent (for the PII assertion)
        self.called_ids = set()


def call_model(ticket_id, system_prompt, user_text, stats):
    """One model call with up to MAX_RETRIES retries; exponential backoff with jitter."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
    for m in messages:  # guardrail: nothing sensitive may leave the building
        assert not contains_pii(m["content"]), f"PII in prompt for {ticket_id}"
    stats.prompts.append(messages)
    stats.called_ids.add(ticket_id)
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            stats.transient_errors += 1
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15)
            log.info(f"[retry] ticket={ticket_id} retry={attempt} wait={wait:.4f}s")
            stats.waits.append((attempt, wait))
            time.sleep(wait)


# ------------------------------------------------------------------ output validation
def validate_reply(text):
    """Return (status, category). Never guess: anything unclear goes to a human."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        found = {w for w in re.findall(r"[a-z]+", (text or "").lower()) if w in CATEGORIES}
        return ("classified_keyword", found.pop()) if len(found) == 1 else ("flagged_invalid_output", None)
    if isinstance(data, dict) and data.get("category") in CATEGORIES:
        return "classified_json", data["category"]
    return "flagged_invalid_output", None


# ------------------------------------------------------------------ routing
def load_segments():
    seg = pd.read_csv(OUT / "customer_segments.csv")
    return seg.set_index("customer_id")


def route(status, category, customer_id, seg):
    if status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "flagged_llm_error"):
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if customer_id in seg.index and bool(seg.loc[customer_id, "is_premium"]):
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


# ------------------------------------------------------------------ the pipeline
def run_pipeline(tickets, system_prompt, token_budget=None, label=""):
    mock_llm.reset()
    random.seed(42)                     # jitter is random; fix it for reproducibility
    log.info(f"--- run: {label} ---")
    seg, stats, total_tokens, rows = load_segments(), Stats(), 0, []
    for t in tickets.itertuples():
        status, category, used = None, None, 0
        if not str(t.text).strip():
            status = "skipped_empty"
        elif est_tokens(t.text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            status = "budget_exhausted"
        else:
            try:
                resp = call_model(t.ticket_id, system_prompt, redact(t.text), stats)
                used = resp.usage.total_tokens
                total_tokens += used
                status, category = validate_reply(resp.choices[0].message.content)
            except TransientLLMError:
                status = "flagged_llm_error"
        rt, pr = route(status, category, t.customer_id, seg)
        rows.append({"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": status,
                     "category": category or "",
                     "segment_name": seg.loc[t.customer_id, "segment_name"] if t.customer_id in seg.index else "",
                     "priority": pr, "route": rt, "tokens_used": used})
    res = pd.DataFrame(rows)
    # acceptance assertions
    blocked = set(res.loc[res.status.isin(["skipped_empty", "flagged_long"]), "ticket_id"])
    assert not (blocked & stats.called_ids), "an empty/long ticket reached the model"
    assert not any(contains_pii(m["content"]) for p in stats.prompts for m in p), "PII found in a prompt"
    assert (res.status != "flagged_llm_error").all(), "a ticket ended with a permanent error"
    return res, stats, total_tokens


def accuracy(res, truth):
    cls = res[res.status.isin(["classified_json", "classified_keyword"])].merge(truth, on="ticket_id")
    return cls, (cls["category"] == cls["true_category"]).mean(), int((cls["category"] == cls["true_category"]).sum())


if __name__ == "__main__":
    tickets = load_tickets()
    truth = tickets[["ticket_id", "true_category"]]
    print(f"tickets: {len(tickets)}")

    # task 1 + 2: guardrail counts and redaction report
    est = tickets["text"].map(est_tokens)
    print(f"empty: {int((tickets.text.str.strip() == '').sum())}; long (>{MAX_INPUT_TOKENS} est tokens): "
          f"{int((est > MAX_INPUT_TOKENS).sum())} (est tokens {sorted(set(est[est > MAX_INPUT_TOKENS]))})")
    rc = count_redactions(tickets["text"])
    print(f"redactions: {rc} total {sum(rc.values())}")
    left = sum(contains_pii(redact(t)) for t in tickets["text"])
    print(f"tickets still containing a pattern after redaction: {left}")
    assert left == 0

    # task 3: prompts
    assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
    assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT and '"fraud"' in FEW_SHOT
    print(f"prompt tokens: zero-shot {est_tokens(ZERO_SHOT)}, few-shot {est_tokens(FEW_SHOT)}")

    # tasks 4-6: full runs
    summary = {}
    for name, prompt in (("zero_shot", ZERO_SHOT), ("few_shot", FEW_SHOT)):
        res, stats, tok = run_pipeline(tickets, prompt, label=name)
        cls, acc, ok = accuracy(res, truth)
        summary[name] = (res, stats, tok, cls, acc, ok)
        waits_ok = all(BASE_DELAY * 2 ** n * 0.9 <= w <= BASE_DELAY * 2 ** n * 1.15 for n, w in stats.waits)
        print(f"\n[{name}] status counts: {res.status.value_counts().to_dict()}")
        print(f"[{name}] transient errors {stats.transient_errors}, retries logged {len(stats.waits)}, "
              f"all waits in range: {waits_ok}; model calls {len(stats.prompts)}")
        print(f"[{name}] classified {len(cls)}; accuracy {acc:.1%} ({ok}/{len(cls)}); total tokens {tok:,}; "
              f"tokens per classified ticket {tok / len(cls):.1f}")
    print("\nzero-shot confusion (rows = true, columns = predicted):")
    print(pd.crosstab(summary["zero_shot"][3]["true_category"], summary["zero_shot"][3]["category"]).to_string())

    # task 7a: budget demo (zero-shot)
    BUDGET = 6000
    demo, _, tok = run_pipeline(tickets, ZERO_SHOT, token_budget=BUDGET, label=f"budget demo {BUDGET}")
    demo.to_csv(OUT / "triage_budget_demo.csv", index=False)
    print(f"\nbudget demo: TOKEN_BUDGET={BUDGET}, tokens used {tok:,}, "
          f"budget_exhausted tickets: {int((demo.status == 'budget_exhausted').sum())}")
    assert (demo.status == "budget_exhausted").sum() >= 1

    # task 7b: final routed output from the few-shot run
    final, _, tok = run_pipeline(tickets, FEW_SHOT, token_budget=100000, label="final few-shot 100000")
    final.to_csv(OUT / "triage_results.csv", index=False)
    print(f"\nfinal: rows {len(final)}, tokens {tok:,}")
    print("status:", final.status.value_counts().to_dict())
    print("route x priority:\n" + final.groupby(["route", "priority"]).size().to_string())
    print("classified (rule-validated):", int(final.status.isin(['classified_json', 'classified_keyword']).sum()),
          "| to human review:", int((final.route == "human_review").sum()))
