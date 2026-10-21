"""Part D tasks 5-8 - tools, ReAct loop, safety limits, checkpoint/resume, and the scenario runner."""
import json
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd

from common import DB, OUT, RAW, STATE
import mock_llm
from d_rag import retrieve

LEDGER = OUT / "refund_ledger.jsonl"
RUNS = OUT / "agent_runs.jsonl"
DEMOS = OUT / "agent_demos.jsonl"
LATE_MINUTES, AUTO_LIMIT = 45, 1000

AGENT_SYSTEM = ("You are a refund AGENT for Instamart support. For each request: look up the order with get_order, "
                "check the refund rules with search_policy, compute the amount with calculate_refund, then call "
                "issue_refund (up to INR 1000) or escalate_to_human (above INR 1000). Finish with one line starting "
                "with REFUND:, ESCALATE:, DENIED:, NOT_FOUND: or NEEDS_INFO:.")


# ------------------------------------------------------------------ tools
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    dm = row[3]
    if dm is not None and float(dm).is_integer():
        dm = int(dm)
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": dm}


def search_policy(query):
    return {"results": retrieve(query)}


def calculate_refund(order_amount, delivery_minutes):
    """Late (> 45 min) completed order -> 100% of the amount; otherwise 0."""
    if delivery_minutes is None or delivery_minutes <= LATE_MINUTES:
        return {"refund_inr": 0.0}
    return {"refund_inr": round(float(order_amount), 2)}


def _ledger_lines():
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()] if LEDGER.exists() else []


def _record(tool, order_id, refund_inr):
    """Append one ledger line. Idempotent per (tool, order): a repeated call after a crash must not pay twice."""
    if not any(e["tool"] == tool and e["order_id"] == order_id for e in _ledger_lines()):
        with LEDGER.open("a") as f:
            f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "tool": tool, "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    if refund_inr > AUTO_LIMIT:   # hard rule enforced in code, whatever the model asks
        return {"error": f"refund above INR {AUTO_LIMIT} must be escalated to a human"}
    return _record("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _record("escalate_to_human", order_id, refund_inr)


def _spec(name, description, props, required):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}


TOOLS_SPEC = [
    _spec("get_order", "Look up an order by id and return its status, amount and delivery time.",
          {"order_id": {"type": "string", "description": "Order id such as ORD-100048"}}, ["order_id"]),
    _spec("search_policy", "Search the policy documents and return the most relevant passages.",
          {"query": {"type": "string", "description": "What to look up"}}, ["query"]),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}},
          ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ------------------------------------------------------------------ agent loop
class SimulatedCrash(RuntimeError):
    pass


def _checkpoint_path(run_id):
    return STATE / f"{run_id}.json"


def save_checkpoint(state):
    cp = {k: state[k] for k in ("run_id", "scenario_id", "user_message", "messages", "tokens_used", "llm_calls",
                                "steps", "tools_executed")}
    cp["step"] = len(state["tools_executed"])
    _checkpoint_path(state["run_id"]).write_text(json.dumps(cp, indent=1, default=str))


def _refund_from(steps):
    for s in reversed(steps):
        if s["tool"] in ("issue_refund", "escalate_to_human") and "refund_inr" in s["observation"]:
            return float(s["observation"]["refund_inr"])
    return 0.0


def _result(state, status, final):
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status,
            "final_answer": final, "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"], "refund_inr": _refund_from(state["steps"])}


def _fallback(state, status, reason):
    done = "; ".join(f"{s['tool']}({', '.join(f'{k}={v}' for k, v in s['arguments'].items() if k != 'query')}) -> "
                     f"{json.dumps(s['observation'])[:70]}" for s in state["steps"]) or "no tool was run"
    final = (f"{status}: {reason} Please handle this manually. Request: \"{state['user_message']}\". "
             f"Completed so far: {done}.")
    return _result(state, status, final)


def _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    t0 = time.monotonic()
    while True:
        if time.monotonic() - t0 >= time_limit_seconds:
            return _fallback(state, "FALLBACK_TIMEOUT", f"time limit of {time_limit_seconds}s reached.")
        if state["llm_calls"] >= max_iterations:
            return _fallback(state, "FALLBACK_MAX_ITERATIONS", f"no final answer after {max_iterations} model calls.")
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > token_limit:      # stop BEFORE executing the tool the model just asked for
            return _fallback(state, "FALLBACK_TOKEN_LIMIT",
                             f"token limit {token_limit} exceeded ({state['tokens_used']} used).")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content or ""
            return _result(state, final.split(":", 1)[0].strip(), final)
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                                                 for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                obs = TOOL_MAP[tc.name](**tc.arguments)
            except Exception as exc:                 # a tool failure becomes an observation, not a crash
                obs = {"error": f"{type(exc).__name__}: {exc}"}
            state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                      "content": json.dumps(obs)})
            state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
            state["tools_executed"].append(tc.name)
            save_checkpoint(state)                   # after every observation
            if crash_after_tools and len(state["tools_executed"]) == crash_after_tools:
                raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools} ({tc.name})")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    state = {"run_id": run_id or f"{scenario_id}-{int(time.time())}", "scenario_id": scenario_id,
             "user_message": user_message,
             "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "steps": [], "tools_executed": []}
    return _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools)


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    """Continue from the last checkpoint. Tools already executed are NOT run again (their results are in messages)."""
    state = json.loads(_checkpoint_path(run_id).read_text())
    state.pop("step", None)
    return _loop(state, max_iterations, token_limit, time_limit_seconds)


# ------------------------------------------------------------------ runner
def _dump(path, res, extra=None):
    row = {k: res[k] for k in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used")}
    row.update(extra or {})
    row["steps"] = res["steps"]
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    for p in (LEDGER, RUNS, DEMOS, *STATE.glob("*.json")):
        p.unlink(missing_ok=True)
    sc = pd.read_csv(RAW / "agent_scenarios.csv", keep_default_na=False)

    print("=== task 6: seven scenarios (default limits: max_iterations=6, token_limit=6000, time_limit=10s) ===")
    rows = []
    for s in sc.itertuples():
        r = run_agent(s.message, s.scenario_id, run_id=f"{s.scenario_id}-main")
        _dump(RUNS, r, {"refund_inr": r["refund_inr"]})
        want = s.expected_outcome
        ok_status = r["status"].startswith("FALLBACK_MAX_ITERATIONS") if want == "FALLBACK" else r["status"] == want
        ok_amt = abs(r["refund_inr"] - float(s.expected_refund_inr)) <= 0.01
        rows.append({"scenario": s.scenario_id, "status": r["status"], "refund_inr": f"{r['refund_inr']:.2f}",
                     "model_calls": r["llm_calls"], "tokens_used": r["tokens_used"],
                     "expected": want, "expected_inr": s.expected_refund_inr, "ok": ok_status and ok_amt})
        assert ok_status and ok_amt, rows[-1]
    print(pd.DataFrame(rows).to_string(index=False))
    s1 = next(r for r in rows if r["scenario"] == "S1"); s7 = next(r for r in rows if r["scenario"] == "S7")
    assert s1["model_calls"] == 5 and s7["model_calls"] == 6 and s7["status"] == "FALLBACK_MAX_ITERATIONS"

    S1 = sc[sc.scenario_id == "S1"].iloc[0]
    print("\n=== task 7: stop-safely demonstrations on S1 ===")
    r = run_agent(S1.message, "S1", token_limit=200, run_id="S1-token-limit-demo")
    _dump(DEMOS, r, {"demo": "token_limit=200"})
    print(f"token_limit=200      -> {r['status']} (llm_calls {r['llm_calls']}, tokens {r['tokens_used']}, "
          f"tools run {[s['tool'] for s in r['steps']]})\n  handoff: {r['final_answer']}")
    assert r["status"] == "FALLBACK_TOKEN_LIMIT"
    r = run_agent(S1.message, "S1", time_limit_seconds=0, run_id="S1-timeout-demo")
    _dump(DEMOS, r, {"demo": "time_limit_seconds=0"})
    print(f"time_limit_seconds=0 -> {r['status']} (llm_calls {r['llm_calls']})\n  handoff: {r['final_answer']}")
    assert r["status"] == "FALLBACK_TIMEOUT" and r["llm_calls"] == 0

    print("\n=== task 8: crash after the 2nd tool execution, then resume ===")
    try:
        run_agent(S1.message, "S1", run_id="S1-crash-demo", crash_after_tools=2)
        raise AssertionError("expected a simulated crash")
    except SimulatedCrash as exc:
        print("crashed:", exc)
    cp = json.loads(_checkpoint_path("S1-crash-demo").read_text())
    print(f"checkpoint on disk: step={cp['step']} tools_executed={cp['tools_executed']} tokens_used={cp['tokens_used']}")
    r = resume_agent("S1-crash-demo")
    _dump(DEMOS, r, {"demo": "crash_and_resume"})
    executed = json.loads(_checkpoint_path("S1-crash-demo").read_text())["tools_executed"]
    print(f"resumed -> {r['status']} refund INR {r['refund_inr']:.2f}; llm_calls {r['llm_calls']}; "
          f"tools executed across crash+resume: {executed}")
    assert executed == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert r["status"] == "REFUND" and abs(r["refund_inr"] - 128.25) < 0.01

    print("\nledger lines:")
    print(LEDGER.read_text())
    print(f"state/ checkpoints: {sorted(p.name for p in STATE.glob('*.json'))}")
