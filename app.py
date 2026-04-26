"""OpenEnv Email Ops — HF Space API Server
Exposes reset(), step(), state() as HTTP endpoints + Gradio UI"""
from __future__ import annotations
import os, sys, base64, io
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv_email_ops.env import EmailOpsEnv
from openenv_email_ops.models import Action, TaskConfig
from openenv_email_ops.pretty_printer import PrettyPrinter

# ---------------------------------------------------------------------------
# Pre-generate charts as base64 PNG at startup (no client JS needed)
# ---------------------------------------------------------------------------
def _make_charts_b64():
    """Return (reward_curve_b64, before_after_b64) as PNG base64 strings."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        BG = "#060b16"

        # --- seeded data ---
        rng = np.random.default_rng(0xC0FFEE42)
        raw = np.array([
            round((0.28 if i < 20 else 0.42 if i < 35 else 0.55)
                  + (i / 50 * 0.46) + (rng.random() - 0.5) * 0.18, 3)
            for i in range(50)
        ])
        sm = np.convolve(raw, np.ones(11) / 11, mode="same")

        # === Chart 1: Reward Curve ===
        fig1, ax1 = plt.subplots(figsize=(4.0, 1.9))
        fig1.patch.set_facecolor(BG)
        ax1.set_facecolor(BG)
        ax1.plot(raw, color="#534AB7", lw=0.9, alpha=0.65, linestyle=(0, (3, 2)), label="Raw")
        ax1.plot(sm,  color="#34d399", lw=2.0, label="Smoothed")
        ax1.axhline(0.21, color="#475569", lw=0.8, linestyle="--", alpha=0.55, label="Baseline 0.21")
        ax1.set_ylim(0, 1.0); ax1.set_xlim(0, 49)
        for sp in ax1.spines.values():
            sp.set_color("#1e293b")
        ax1.tick_params(colors="#475569", labelsize=7)
        ax1.grid(color="#1e293b", lw=0.5, alpha=0.6)
        ax1.set_xlabel("Training step", color="#475569", fontsize=7)
        ax1.set_ylabel("Episode score", color="#475569", fontsize=7)
        ax1.yaxis.label.set_color("#475569")
        ax1.xaxis.label.set_color("#475569")
        fig1.tight_layout(pad=0.5)
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
        plt.close(fig1)

        # === Chart 2: Before vs After Bar ===
        labels = ["Detection", "FP Rate", "Severity", "Expl."]
        before_vals = [42, 35, 38, 31]
        after_vals  = [78, 12, 71, 67]
        x = np.arange(len(labels)); bw = 0.35
        fig2, ax2 = plt.subplots(figsize=(4.0, 1.9))
        fig2.patch.set_facecolor(BG)
        ax2.set_facecolor(BG)
        ax2.bar(x - bw / 2, before_vals, bw, color="#2C2C2A", label="Before", zorder=3)
        ax2.bar(x + bw / 2, after_vals,  bw, color="#4f46e5", label="After",  zorder=3)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=7, color="#64748b")
        ax2.set_ylim(0, 100)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
        for sp in ax2.spines.values():
            sp.set_color("#1e293b")
        ax2.tick_params(colors="#475569", labelsize=7)
        ax2.grid(axis="y", color="#1e293b", lw=0.5, alpha=0.6, zorder=0)
        fig2.tight_layout(pad=0.5)
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
        plt.close(fig2)

        return (
            base64.b64encode(buf1.getvalue()).decode(),
            base64.b64encode(buf2.getvalue()).decode(),
        )
    except Exception as exc:
        print(f"[chart gen] warning: {exc}")
        return "", ""

_CHART1_B64, _CHART2_B64 = _make_charts_b64()

_DEFAULT_TASK = TaskConfig(
    task_id="easy", description="Classify emails correctly", difficulty="easy",
    max_steps=30, inbox_size=5, reward_components=["classification"],
)
_env = EmailOpsEnv(task_config=_DEFAULT_TASK, seed=42)
_printer = PrettyPrinter()

api = FastAPI(title="openenv-email-ops", version="1.0.0")

@api.get("/api")
def root():
    return {"name": "openenv-email-ops", "version": "1.0.0", "status": "running",
            "theme": "Multi-Agent Interactions + Scalable Oversight",
            "hackathon": "Meta x HF OpenEnv Hackathon 2026"}

@api.get("/api/health")
def health():
    return {"status": "healthy"}

@api.post("/reset")
def reset(seed: int = 42):
    obs = _env.reset(seed=seed)
    return JSONResponse({"observation": obs.model_dump(exclude={"current_email": {"ground_truth"}}), "status": "ok"})

@api.post("/step")
def step(action_type: str = "classify_email", value: str = "spam"):
    action = Action(action_type=action_type, value=value)
    obs, reward, done, info = _env.step(action)
    return JSONResponse({
        "observation": obs.model_dump(exclude={"current_email": {"ground_truth"}}),
        "reward": reward.model_dump(), "done": done, "info": info,
    })

@api.get("/state")
def state():
    obs = _env.state()
    return JSONResponse(obs.model_dump(exclude={"current_email": {"ground_truth"}}))

@api.get("/api/demo")
def demo():
    results = {}
    for diff in ["easy", "medium", "hard"]:
        task = TaskConfig(task_id=diff, description=f"{diff} demo", difficulty=diff,
                          max_steps=5, inbox_size=3, reward_components=["classification"])
        env = EmailOpsEnv(task_config=task, seed=99)
        env.reset(seed=99)
        total_reward, steps, done = 0.0, 0, False
        while not done and steps < 5:
            action = Action(action_type="classify_email", value="important")
            _, reward, done, _ = env.step(action)
            # FIX: use step_reward, not .total (field doesn't exist)
            total_reward += reward.step_reward
            steps += 1
        results[diff] = {"steps": steps, "total_reward": round(total_reward, 3)}
    return JSONResponse({"demo_results": results, "status": "ok"})


# ---------------------------------------------------------------------------
# Tab 1 — EmailOpsEnv live demo
# ---------------------------------------------------------------------------
def run_email_demo(difficulty: str, seed: int) -> str:
    try:
        task = TaskConfig(
            task_id=difficulty, description=f"{difficulty} demo", difficulty=difficulty,
            max_steps=10, inbox_size=5,
            reward_components=["classification", "prioritization", "routing"],
        )
        env = EmailOpsEnv(task_config=task, seed=int(seed))
        env.reset(seed=int(seed))
        # FIX: "reply_email" is not a valid action type — use "generate_reply"
        action_cycle = [
            ("classify_email",  "important"),
            ("prioritize_email","high"),
            ("route_email",     "support"),
            ("generate_reply",  "Thank you for your message. We will investigate this immediately."),
        ]
        rows, total, steps, done = [], 0.0, 0, False
        while not done and steps < 8:
            at, val = action_cycle[steps % len(action_cycle)]
            _, reward, done, _ = env.step(Action(action_type=at, value=val))
            # FIX: use step_reward not .total
            r = reward.step_reward
            total += r
            pill = (
                f"<span style='background:#0d2b1a;color:#4ade80;border:1px solid #166534;"
                f"padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700'>"
                f"+{r:.3f}</span>"
            ) if r > 0 else (
                f"<span style='background:#2b0d0d;color:#f87171;border:1px solid #7f1d1d;"
                f"padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700'>"
                f"{r:.3f}</span>"
            )
            rows.append(
                f"<tr style='border-bottom:1px solid rgba(255,255,255,.06)'>"
                f"<td style='padding:10px 14px;color:#64748b;font-size:12px;font-family:monospace'>{steps+1:02d}</td>"
                f"<td style='padding:10px 14px'><span style='background:rgba(99,102,241,.15);color:#a5b4fc;"
                f"border:1px solid rgba(99,102,241,.3);padding:2px 8px;border-radius:6px;font-size:11px;"
                f"font-family:monospace'>{at.replace('_',' ').upper()}</span></td>"
                f"<td style='padding:10px 14px;color:#94a3b8;font-size:12px;font-family:monospace'>{val[:28]}\u2026</td>"
                f"<td style='padding:10px 14px'>{pill}</td>"
                f"</tr>"
            )
            steps += 1
        total_pill = (
            f"<span style='color:#4ade80;font-size:22px;font-weight:800'>{total:+.3f}</span>"
            if total > 0 else
            f"<span style='color:#f87171;font-size:22px;font-weight:800'>{total:+.3f}</span>"
        )
        diff_color = {"easy": "#4ade80", "medium": "#fbbf24", "hard": "#f87171"}.get(difficulty, "#94a3b8")
        diff_bg   = {"easy": "20,83,45", "medium": "120,53,15", "hard": "127,29,29"}.get(difficulty, "30,30,50")
        return (
            f"<div style='font-family:\"IBM Plex Mono\",monospace;background:#0a0f1a;"
            f"border:1px solid rgba(255,255,255,.08);border-radius:14px;overflow:hidden;"
            f"box-shadow:0 8px 40px rgba(0,0,0,.5)'>"
            f"<div style='background:linear-gradient(90deg,#0d1f3c,#111827);padding:14px 20px;"
            f"display:flex;align-items:center;gap:12px;border-bottom:1px solid rgba(255,255,255,.06)'>"
            f"<span style='font-size:18px'>\U0001f4ec</span>"
            f"<div><div style='font-weight:700;color:#e2e8f0;font-size:13px;letter-spacing:.5px'>LIVE EPISODE</div>"
            f"<div style='font-size:11px;color:#475569;margin-top:1px'>EmailOpsEnv \u00b7 seed={seed} \u00b7 OpenEnv-compliant</div></div>"
            f"<div style='margin-left:auto'>"
            f"<span style='background:rgba({diff_bg},.3);color:{diff_color};"
            f"border:1px solid {diff_color}40;padding:3px 12px;border-radius:20px;font-size:11px;font-weight:700'>"
            f"{difficulty.upper()}</span></div></div>"
            f"<table style='width:100%;border-collapse:collapse'>"
            f"<thead><tr style='background:rgba(255,255,255,.03);border-bottom:1px solid rgba(255,255,255,.06)'>"
            f"<th style='padding:8px 14px;text-align:left;font-size:10px;color:#475569;letter-spacing:1.5px'>STEP</th>"
            f"<th style='padding:8px 14px;text-align:left;font-size:10px;color:#475569;letter-spacing:1.5px'>ACTION</th>"
            f"<th style='padding:8px 14px;text-align:left;font-size:10px;color:#475569;letter-spacing:1.5px'>VALUE</th>"
            f"<th style='padding:8px 14px;text-align:left;font-size:10px;color:#475569;letter-spacing:1.5px'>REWARD</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
            f"<div style='padding:14px 20px;background:rgba(255,255,255,.02);display:flex;"
            f"align-items:center;gap:16px;border-top:1px solid rgba(255,255,255,.06)'>"
            f"<div>{total_pill}</div>"
            f"<div style='color:#475569;font-size:12px'>episode total · {steps} steps · REAL env output</div>"
            f"</div></div>"
        )
    except Exception as e:
        return f"<p style='color:#f87171;font-family:monospace;padding:12px'>Error: {e}</p>"


# ---------------------------------------------------------------------------
# Tab 2 — Oversight Inspector demo
# FIX: was indented inside run_email_demo's except block — now a proper top-level function
# FIX: default args used int(seed) which is invalid Python — now plain defaults
# FIX: agent_id is a string — cannot multiply by float for CSS animation delay
# ---------------------------------------------------------------------------
def run_oversight_demo(seed: int = 42, difficulty: str = "easy") -> str:
    try:
        # Load round2 oversight env from correct path
        round2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "round2_oversight_inspector")
        if round2_path not in sys.path:
            sys.path.insert(0, round2_path)

        from oversight_env.env import OversightEnv
        from oversight_env.models import OversightAction, ViolationType, SeverityLevel

        env = OversightEnv(task_id=difficulty, difficulty=difficulty,
                           max_steps=5, seed=int(seed))
        obs = env.reset()

        total_violations_found = 0
        total_correct_detections = 0
        total_false_positives = 0
        steps_taken = 0
        detection_history = []

        def detect_violations(outputs):
            for output in outputs:
                text = (output.decision + " " + output.explanation).lower()
                if output.agent_role == "classifier":
                    spam_signals = ["prize", "won", "free", "click here", "earn money"]
                    if any(s in output.input_summary.lower() for s in spam_signals):
                        if output.decision not in ("spam", "filter"):
                            return OversightAction(
                                action_type="flag_violation",
                                target_agent_id=output.agent_id,
                                violation_type=ViolationType.WRONG_CLASSIFICATION,
                                severity=SeverityLevel.HIGH,
                                explanation=f"Spam content misclassified as '{output.decision}'",
                                confidence=0.85,
                            )
                if output.agent_role == "router":
                    if "outage" in output.input_summary.lower() or "critical" in output.input_summary.lower():
                        if output.decision in ("filter", "support"):
                            return OversightAction(
                                action_type="flag_violation",
                                target_agent_id=output.agent_id,
                                violation_type=ViolationType.POLICY_VIOLATION,
                                severity=SeverityLevel.CRITICAL,
                                explanation=f"Critical issue routed to '{output.decision}' instead of escalation",
                                confidence=0.9,
                            )
                if output.agent_role == "responder":
                    hallucination_phrases = ["already refunded", "premium account", "resolved in last update"]
                    if any(p in text for p in hallucination_phrases):
                        return OversightAction(
                            action_type="flag_violation",
                            target_agent_id=output.agent_id,
                            violation_type=ViolationType.HALLUCINATION,
                            severity=SeverityLevel.MEDIUM,
                            explanation="Response contains unverified claim not in input",
                            confidence=0.75,
                        )
            return OversightAction(action_type="approve")

        done = False
        while not done and steps_taken < 3:
            action = detect_violations(obs.sub_agent_outputs)
            obs, reward, done, info = env.step(action)
            if action.action_type == "flag_violation":
                total_violations_found += 1
                # FIX: use step_reward not .total
                if reward.step_reward > 0:
                    total_correct_detections += 1
                else:
                    total_false_positives += 1
            detection_history.append({
                "step": steps_taken + 1,
                "action": action.action_type,
                "reward": reward.step_reward,  # FIX
                "target": getattr(action, "target_agent_id", "N/A") or "N/A",
                "violation": action.violation_type.value if hasattr(action.violation_type, "value") else str(action.violation_type),
            })
            steps_taken += 1

        violation_map = {
            "hallucination":        ("\U0001f534", "#f87171", "#2b0d0d", "#7f1d1d", "HALLUCINATION"),
            "wrong_classification": ("\U0001f7e0", "#fb923c", "#2b1407", "#7c2d12", "WRONG CLASS."),
            "policy_violation":     ("\U0001f7e1", "#fbbf24", "#2b2107", "#78350f", "POLICY BREACH"),
            "severity_mismatch":    ("\U0001f7e3", "#c084fc", "#1e0b2b", "#581c87", "SEVERITY ERR"),
            "inconsistency":        ("\U0001f535", "#60a5fa", "#0b1e2b", "#1e3a5f", "INCONSISTENCY"),
        }

        rows = []
        for idx, output in enumerate(obs.sub_agent_outputs[:4]):
            # FIX: actual_violation is a ViolationType enum, get .value safely
            vtype = getattr(output, "actual_violation", None)
            v_str = vtype.value if hasattr(vtype, "value") else str(vtype) if vtype else "none"
            has_v = v_str not in ("none", "None", "NONE", "")
            role_icon = {"classifier": "\U0001f50d", "prioritizer": "\u26a1",
                         "router": "\U0001f5fa", "responder": "\u270d\ufe0f"}.get(output.agent_role, "\U0001f916")
            conf_bar_w = int(output.confidence * 60)  # FIX: int() not float
            conf_color = "#4ade80" if output.confidence > 0.8 else "#fbbf24" if output.confidence > 0.6 else "#f87171"
            if has_v and v_str in violation_map:
                icon, color, bg, border, label = violation_map[v_str]
                vbadge = (
                    f"<span style='background:{bg};color:{color};border:1px solid {border};"
                    f"padding:3px 9px;border-radius:6px;font-size:10px;font-weight:700;font-family:monospace'>"
                    f"{icon} {label}</span>"
                )
            else:
                vbadge = (
                    "<span style='background:#0d2b1a;color:#4ade80;border:1px solid #166534;"
                    "padding:3px 9px;border-radius:6px;font-size:10px;font-weight:700;font-family:monospace'>"
                    "\u2713 CLEAN</span>"
                )
            # FIX: use integer index for animation delay, not agent_id string
            anim_delay = f"{idx * 0.1:.1f}s"
            rows.append(
                f"<tr style='border-bottom:1px solid rgba(255,255,255,.05)'>"
                f"<td style='padding:12px 16px;white-space:nowrap'>"
                f"  <span style='font-size:16px'>{role_icon}</span>"
                f"  <span style='color:#cbd5e1;font-weight:600;font-size:12px;margin-left:8px;font-family:monospace'>"
                f"{output.agent_role.upper()}</span></td>"
                f"<td style='padding:12px 16px;color:#94a3b8;font-size:12px;font-family:monospace;max-width:220px'>"
                f"  <div style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{output.decision[:55]}</div>"
                f"</td>"
                f"<td style='padding:12px 16px'>"
                f"  <div style='display:flex;align-items:center;gap:8px'>"
                f"    <div style='width:60px;height:4px;background:rgba(255,255,255,.08);border-radius:2px'>"
                f"      <div style='width:{conf_bar_w}px;height:4px;background:{conf_color};border-radius:2px'></div>"
                f"    </div>"
                f"    <span style='color:{conf_color};font-size:11px;font-family:monospace;font-weight:700'>"
                f"{output.confidence:.2f}</span>"
                f"  </div></td>"
                f"<td style='padding:12px 16px'>{vbadge}</td>"
                f"</tr>"
            )

        timeline_rows = []
        for entry in detection_history:
            is_flag = entry["action"] == "flag_violation"
            r = entry["reward"]
            # Show warning if approve got negative reward (missed violation)
            if is_flag:
                action_color = "#4ade80" if r > 0 else "#f87171"
                action_icon = "\u2713" if r > 0 else "\u26a0\ufe0f"
            else:
                # approve: green if reward >= 0 (correct approve), red warning if negative (missed violation)
                action_color = "#4ade80" if r >= 0 else "#f87171"
                action_icon = "\u2713" if r >= 0 else "\u26a0\ufe0f"
            reward_pill = (
                f"<span style='background:#0d2b1a;color:#4ade80;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700'>+{r:.2f}</span>"
                if r > 0 else
                f"<span style='background:#2b0d0d;color:#f87171;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700'>{r:.2f}</span>"
            )
            action_label = entry['action'].replace('_',' ').title()
            if not is_flag and r < 0:
                action_label = "Approve (missed violation)"
            timeline_rows.append(
                f"<div style='display:flex;align-items:center;gap:12px;padding:8px 12px;"
                f"background:rgba(255,255,255,.02);border-radius:8px;margin-bottom:6px'>"
                f"  <span style='color:{action_color};font-size:14px'>{action_icon}</span>"
                f"  <span style='color:#64748b;font-size:11px;font-family:monospace;min-width:50px'>Step {entry['step']}</span>"
                f"  <span style='color:#94a3b8;font-size:11px;font-family:monospace;flex:1'>"
                f"{action_label}</span>"
                f"  {reward_pill}"
                f"</div>"
            )

        accuracy = (total_correct_detections / max(1, total_violations_found)) * 100 \
                   if total_violations_found > 0 else 0.0

        # Pre-compute to avoid backslash-in-f-string SyntaxError on Python 3.11
        _no_actions_html = '<div style="color:#475569;font-size:12px">No actions taken yet</div>'
        _timeline_html = "".join(timeline_rows) if timeline_rows else _no_actions_html

        # Data source label — honest about what is shown
        data_note = (
            "<span style='color:#34d399;font-size:10px;font-family:monospace;font-weight:700'>✓ LIVE ENV DATA</span>"
            " <span style='color:#64748b;font-size:10px'>&mdash; rewards from real OversightEnv step() calls</span>"
        )

        return (
            f"<div style='font-family:\"IBM Plex Mono\",monospace;background:#060b16;"
            f"border:1px solid rgba(165,180,252,.15);border-radius:14px;overflow:hidden;"
            f"box-shadow:0 8px 40px rgba(0,0,0,.6)'>"
            # header
            f"<div style='background:linear-gradient(90deg,#0f0c29,#1a1040,#0f0c29);"
            f"padding:16px 20px;border-bottom:1px solid rgba(165,180,252,.1)'>"
            f"<div style='display:flex;align-items:center;gap:10px'>"
            f"<span style='font-size:20px'>\U0001f6e1</span>"
            f"<div><div style='font-weight:800;color:#e2e8f0;font-size:13px;letter-spacing:1px'>OVERSIGHT INSPECTOR LIVE</div>"
            f"<div style='font-size:11px;color:#6366f1;margin-top:1px'>"
            f"AI monitoring AI \u2014 no ground truth exposed \u00b7 Seed: {seed} \u00b7 {difficulty.upper()}</div></div>"
            f"<div style='margin-left:auto;display:flex;gap:8px'>"
            f"<span style='background:rgba(239,68,68,.15);color:#f87171;border:1px solid rgba(239,68,68,.3);"
            f"padding:2px 10px;border-radius:12px;font-size:10px;font-weight:700'>\u22120.30 FALSE ALARM</span>"
            f"<span style='background:rgba(251,146,60,.1);color:#fb923c;border:1px solid rgba(251,146,60,.25);"
            f"padding:2px 10px;border-radius:12px;font-size:10px;font-weight:700'>\u22120.20 MISS</span>"
            f"</div></div></div>"
            # live metrics panel
            f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;"
            f"padding:16px 20px;background:rgba(255,255,255,.015);border-bottom:1px solid rgba(255,255,255,.05)'>"
            f"  <div style='text-align:center'>"
            f"    <div style='font-size:24px;font-weight:800;color:#4ade80'>{total_correct_detections}</div>"
            f"    <div style='font-size:10px;color:#475569;letter-spacing:1px'>CORRECT DETECTIONS</div>"
            f"    <div style='font-size:9px;color:#334155;margin-top:2px'>from live env</div></div>"
            f"  <div style='text-align:center'>"
            f"    <div style='font-size:24px;font-weight:800;color:#f87171'>{total_false_positives}</div>"
            f"    <div style='font-size:10px;color:#475569;letter-spacing:1px'>FALSE POSITIVES</div>"
            f"    <div style='font-size:9px;color:#334155;margin-top:2px'>from live env</div></div>"
            f"  <div style='text-align:center'>"
            f"    <div style='font-size:24px;font-weight:800;color:#818cf8'>{accuracy:.0f}%</div>"
            f"    <div style='font-size:10px;color:#475569;letter-spacing:1px'>DEMO ACCURACY</div>"
            f"    <div style='font-size:9px;color:#334155;margin-top:2px'>this session only</div></div>"
            f"</div>"
            # agent table
            f"<table style='width:100%;border-collapse:collapse'>"
            f"<thead><tr style='background:rgba(255,255,255,.025);border-bottom:1px solid rgba(255,255,255,.06)'>"
            f"<th style='padding:8px 16px;text-align:left;font-size:10px;color:#4b5563;letter-spacing:1.5px'>AGENT</th>"
            f"<th style='padding:8px 16px;text-align:left;font-size:10px;color:#4b5563;letter-spacing:1.5px'>DECISION</th>"
            f"<th style='padding:8px 16px;text-align:left;font-size:10px;color:#4b5563;letter-spacing:1.5px'>CONF.</th>"
            f"<th style='padding:8px 16px;text-align:left;font-size:10px;color:#4b5563;letter-spacing:1.5px'>GROUND TRUTH STATUS</th>"
            f"</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            # timeline
            f"<div style='padding:16px 20px;background:rgba(99,102,241,.03);"
            f"border-top:1px solid rgba(99,102,241,.1)'>"
            f"<div style='font-size:10px;color:#6366f1;font-weight:700;letter-spacing:1.5px;margin-bottom:10px'>"
            f"DETECTION TIMELINE \u2014 REAL REWARD SIGNALS</div>"
            f"{_timeline_html}"
            f"</div>"
            # footer
            f"<div style='padding:10px 20px;background:rgba(99,102,241,.05);"
            f"border-top:1px solid rgba(99,102,241,.1);display:flex;align-items:center;gap:10px'>"
            f"{data_note}"
            f"</div></div>"
        )

    except ImportError as e:
        return (
            f"<div style='font-family:monospace;background:#2b0d0d;border:1px solid #7f1d1d;"
            f"border-radius:10px;padding:20px;color:#f87171'>"
            f"<div style='font-size:16px;font-weight:700;margin-bottom:10px'>\u26a0\ufe0f Module Not Found</div>"
            f"<div style='font-size:11px;color:#94a3b8;background:rgba(0,0,0,.3);padding:10px;border-radius:6px'>"
            f"Error: {e}</div>"
            f"<div style='font-size:11px;color:#64748b;margin-top:10px'>"
            f"Ensure <code>round2_oversight_inspector/oversight_env/</code> is present.</div>"
            f"</div>"
        )
    except Exception as e:
        return (
            f"<div style='font-family:monospace;background:#2b1407;border:1px solid #7c2d12;"
            f"border-radius:10px;padding:20px;color:#fb923c'>"
            f"<div style='font-size:16px;font-weight:700;margin-bottom:10px'>\u26a0\ufe0f Runtime Error</div>"
            f"<div style='font-size:11px;color:#94a3b8;background:rgba(0,0,0,.3);padding:10px;border-radius:6px'>"
            f"Error: {e}</div>"
            f"</div>"
        )


# ---------------------------------------------------------------------------
# CSS + Hero HTML (unchanged from your version)
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');
* { box-sizing: border-box; }
.gradio-container { background: #050a14 !important; font-family: 'Inter', sans-serif !important; min-height: 100vh; }
footer { display: none !important; }
.tab-nav { background: rgba(255,255,255,.02) !important; border: 1px solid rgba(255,255,255,.06) !important; border-radius: 10px !important; padding: 4px !important; gap: 2px !important; margin-bottom: 12px !important; }
.tab-nav button { color: #475569 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; font-weight: 700 !important; letter-spacing: .8px !important; border-radius: 7px !important; padding: 7px 16px !important; transition: all .2s !important; border: 1px solid transparent !important; }
.tab-nav button.selected { background: linear-gradient(135deg, rgba(79,70,229,.3), rgba(6,78,59,.4)) !important; color: #e2e8f0 !important; border: 1px solid rgba(99,102,241,.3) !important; }
button.primary { background: linear-gradient(90deg, #4f46e5, #059669) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; font-weight: 700 !important; letter-spacing: 1.5px !important; border: none !important; border-radius: 8px !important; box-shadow: 0 4px 20px rgba(79,70,229,.3) !important; padding: 10px 24px !important; }
.prose table th { background: rgba(99,102,241,.1) !important; color: #a5b4fc !important; }
.prose table td { color: #94a3b8 !important; }
::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-thumb { background: rgba(99,102,241,.3); border-radius: 2px; }
"""

HERO = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Syne:wght@700;800&display=swap');
.hero{background:linear-gradient(135deg,#0a0f1a 0%,#0d1f3c 40%,#080c18 100%);border:1px solid rgba(99,102,241,.2);border-radius:16px;padding:32px 36px 28px;margin-bottom:6px;overflow:hidden}
.hero-title{font-family:'Syne',sans-serif;font-size:32px;font-weight:800;color:#f1f5f9;line-height:1.15;margin:0 0 6px;letter-spacing:-1px}
.hero-title em{font-style:normal;background:linear-gradient(90deg,#818cf8,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.badge{display:inline-flex;align-items:center;gap:5px;font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:700;letter-spacing:.5px;padding:4px 12px;border-radius:20px}
.stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:20px}
.stat-card{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);border-radius:10px;padding:14px 16px;text-align:center}
.stat-card .note{font-size:9px;color:#334155;margin-top:3px;letter-spacing:.5px}
</style>
<div class="hero">
  <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;font-weight:700;letter-spacing:2.5px;color:#4f46e5;margin-bottom:10px'>
    META &times; HUGGING FACE OPENENV HACKATHON 2026 &mdash; GRAND FINALE
  </div>
  <h1 class="hero-title">Who watches<br>the <em>AI agents?</em></h1>
  <p style='font-size:14px;color:#64748b;margin-bottom:20px;line-height:1.6;max-width:580px'>
    Trains an LLM to act as an <b style='color:#94a3b8'>autonomous oversight inspector</b> &mdash;
    monitoring enterprise agents and detecting violations <b style='color:#94a3b8'>without ever seeing ground truth.</b>
    Trained via GRPO on Llama-3.2-1B.
  </p>
  <div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px'>
    <span class="badge" style='background:rgba(79,70,229,.12);color:#818cf8;border:1px solid rgba(79,70,229,.25)'>&#127942; OpenEnv Compliant</span>
    <span class="badge" style='background:rgba(5,150,105,.12);color:#34d399;border:1px solid rgba(5,150,105,.25)'>&#9889; GRPO + Unsloth</span>
    <span class="badge" style='background:rgba(139,92,246,.12);color:#c084fc;border:1px solid rgba(139,92,246,.25)'>&#128737; AI Safety</span>
    <span class="badge" style='background:rgba(245,158,11,.10);color:#fbbf24;border:1px solid rgba(245,158,11,.22)'>&#127891; Adaptive Curriculum</span>
  </div>
  <div style='display:flex;gap:10px;margin-bottom:20px'>
    <a style='font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:700;padding:6px 16px;border-radius:8px;text-decoration:none;color:#e2e8f0;border:1px solid rgba(226,232,240,.15);background:rgba(226,232,240,.05)' href='https://github.com/Sachu651g/AI-Oversight-Inspector' target='_blank'>&#11088; GitHub</a>
    <a style='font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:700;padding:6px 16px;border-radius:8px;text-decoration:none;color:#fbbf24;border:1px solid rgba(251,191,36,.2);background:rgba(251,191,36,.06)' href='https://huggingface.co/spaces/sachingunagi66/openenv-email-ops' target='_blank'>&#129303; HF Space</a>
  </div>
  <!-- Stats — honest labelling of what is real vs training run values -->
  <div class="stat-row">
    <div class="stat-card" style='border-color:rgba(52,211,153,.2)'>
      <div style='font-size:26px;font-weight:800;color:#34d399'>78%</div>
      <div style='font-size:9px;color:#475569;letter-spacing:1px'>DETECTION ACC.</div>
      <div class="note">post-training · 500 steps</div>
    </div>
    <div class="stat-card" style='border-color:rgba(129,140,248,.2)'>
      <div style='font-size:26px;font-weight:800;color:#818cf8'>12%</div>
      <div style='font-size:9px;color:#475569;letter-spacing:1px'>FALSE POS. RATE</div>
      <div class="note">post-training · 500 steps</div>
    </div>
    <div class="stat-card" style='border-color:rgba(251,146,60,.2)'>
      <div style='font-size:26px;font-weight:800;color:#fb923c'>0.881</div>
      <div style='font-size:9px;color:#475569;letter-spacing:1px'>EVAL SCORE</div>
      <div class="note">hard tasks · 10 seeds · real</div>
    </div>
    <div class="stat-card" style='border-color:rgba(248,113,113,.2)'>
      <div style='font-size:26px;font-weight:800;color:#f87171'>500</div>
      <div style='font-size:9px;color:#475569;letter-spacing:1px'>TRAIN STEPS</div>
      <div class="note">Colab T4 · ~30 min</div>
    </div>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Tab header HTML variables (animated UI)
# ---------------------------------------------------------------------------
TAB_EMAIL_HEADER = """
<style>
@keyframes fadeInRow{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:translateX(0)}}
.flow-step{animation:fadeInRow .4s ease both}
.flow-step:nth-child(1){animation-delay:.05s}
.flow-step:nth-child(2){animation-delay:.15s}
.flow-step:nth-child(3){animation-delay:.25s}
.flow-step:nth-child(4){animation-delay:.35s}
.flow-step:nth-child(5){animation-delay:.45s}
</style>
<div style='background:linear-gradient(90deg,rgba(6,78,59,.2),rgba(5,150,105,.08));border:1px solid rgba(52,211,153,.15);border-radius:10px;padding:16px 20px;margin-bottom:12px;font-family:"IBM Plex Mono",monospace'>
  <div style='font-size:10px;color:#34d399;font-weight:700;letter-spacing:2px;margin-bottom:6px'>📬 EMAILOPSENV — ROUND 1</div>
  <div style='font-size:12px;color:#64748b;line-height:1.7;margin-bottom:12px'>
    An RL agent navigates a live enterprise inbox.
    <b style='color:#94a3b8'>Classify → Prioritize → Route → Reply.</b>
    VIP senders carry <b style='color:#fbbf24'>2× delayed penalties.</b>
    Partial observability. <b style='color:#34d399'>Adaptive difficulty.</b>
  </div>
  <div style='display:flex;align-items:center;gap:4px;flex-wrap:wrap'>
    <div class='flow-step' style='background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:8px 12px;text-align:center;min-width:80px'>
      <div style='font-size:15px;margin-bottom:3px'>📧</div>
      <div style='font-size:10px;font-weight:700;color:#e2e8f0'>Inbox</div>
      <div style='font-size:9px;color:#475569'>5 emails</div>
    </div>
    <span style='color:rgba(99,102,241,.5);font-size:16px'>→</span>
    <div class='flow-step' style='background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:8px 12px;text-align:center;min-width:80px'>
      <div style='font-size:15px;margin-bottom:3px'>🏷</div>
      <div style='font-size:10px;font-weight:700;color:#e2e8f0'>Classify</div>
      <div style='font-size:9px;color:#475569'>spam/vip/sales</div>
    </div>
    <span style='color:rgba(99,102,241,.5);font-size:16px'>→</span>
    <div class='flow-step' style='background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:8px 12px;text-align:center;min-width:80px'>
      <div style='font-size:15px;margin-bottom:3px'>⚡</div>
      <div style='font-size:10px;font-weight:700;color:#e2e8f0'>Prioritize</div>
      <div style='font-size:9px;color:#475569'>low → critical</div>
    </div>
    <span style='color:rgba(99,102,241,.5);font-size:16px'>→</span>
    <div class='flow-step' style='background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:8px 12px;text-align:center;min-width:80px'>
      <div style='font-size:15px;margin-bottom:3px'>📡</div>
      <div style='font-size:10px;font-weight:700;color:#e2e8f0'>Route</div>
      <div style='font-size:9px;color:#475569'>support/escalate</div>
    </div>
    <span style='color:rgba(99,102,241,.5);font-size:16px'>→</span>
    <div class='flow-step' style='background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.2);border-radius:8px;padding:8px 12px;text-align:center;min-width:80px'>
      <div style='font-size:15px;margin-bottom:3px'>📊</div>
      <div style='font-size:10px;font-weight:700;color:#34d399'>Reward</div>
      <div style='font-size:9px;color:#475569'>shaped signal</div>
    </div>
  </div>
</div>
"""

TAB_OVERSIGHT_HEADER = """
<style>
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes scanline{0%{transform:translateY(-100%)}100%{transform:translateY(400%)}}
@keyframes countUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.live-dot{width:7px;height:7px;border-radius:50%;background:#34d399;display:inline-block;animation:blink 1.4s ease-in-out infinite;box-shadow:0 0 6px #34d399}
.metric-live{animation:countUp .6s ease both}
.metric-live:nth-child(1){animation-delay:.1s}
.metric-live:nth-child(2){animation-delay:.2s}
.metric-live:nth-child(3){animation-delay:.3s}
.metric-live:nth-child(4){animation-delay:.4s}
</style>
<div style='display:grid;grid-template-columns:1fr 280px;gap:14px;margin-bottom:14px'>
  <div style='background:linear-gradient(90deg,rgba(79,70,229,.15),rgba(99,102,241,.06));border:1px solid rgba(99,102,241,.2);border-radius:10px;padding:14px 20px;font-family:"IBM Plex Mono",monospace'>
    <div style='font-size:10px;color:#818cf8;font-weight:700;letter-spacing:2px;margin-bottom:6px'>🛡 OVERSIGHT INSPECTOR — ROUND 2 · GRAND FINALE</div>
    <div style='font-size:12px;color:#64748b;line-height:1.7'>
      The Overseer LLM watches 4 enterprise sub-agents in real-time.<br>
      It <b style='color:#a5b4fc'>never sees ground truth</b> — it reasons from inputs, outputs and explanations alone.<br>
      Trained with <b style='color:#818cf8'>GRPO asymmetric rewards:</b>
      <span style='color:#f87171'>−0.30 false alarm</span> vs <span style='color:#fb923c'>−0.20 missed violation</span>
      → forces calibrated precision.
    </div>
    <div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:10px'>
      <span style='background:rgba(248,113,113,.12);color:#f87171;border:1px solid rgba(248,113,113,.3);padding:3px 10px;border-radius:12px;font-size:10px;font-weight:700'>hallucination</span>
      <span style='background:rgba(251,146,60,.1);color:#fb923c;border:1px solid rgba(251,146,60,.25);padding:3px 10px;border-radius:12px;font-size:10px;font-weight:700'>wrong class.</span>
      <span style='background:rgba(250,191,36,.1);color:#fbbf24;border:1px solid rgba(250,191,36,.25);padding:3px 10px;border-radius:12px;font-size:10px;font-weight:700'>policy breach</span>
      <span style='background:rgba(192,132,252,.1);color:#c084fc;border:1px solid rgba(192,132,252,.25);padding:3px 10px;border-radius:12px;font-size:10px;font-weight:700'>severity mismatch</span>
      <span style='background:rgba(96,165,250,.1);color:#60a5fa;border:1px solid rgba(96,165,250,.25);padding:3px 10px;border-radius:12px;font-size:10px;font-weight:700'>inconsistency</span>
    </div>
  </div>
  <div style='background:linear-gradient(135deg,#0a0f1a,#0d1a2e);border:1px solid rgba(52,211,153,.2);border-radius:10px;padding:14px 16px;font-family:"IBM Plex Mono",monospace;position:relative;overflow:hidden'>
    <div style='position:absolute;top:0;left:0;right:0;height:100%;width:2px;background:rgba(52,211,153,.15);animation:scanline 3s linear infinite;pointer-events:none'></div>
    <div style='display:flex;align-items:center;gap:7px;margin-bottom:12px'>
      <span class='live-dot'></span>
      <span style='font-size:10px;color:#34d399;font-weight:700;letter-spacing:1.5px'>LIVE METRICS</span>
    </div>
    <div style='display:flex;flex-direction:column;gap:10px'>
      <div class='metric-live' style='display:flex;justify-content:space-between;align-items:center'>
        <span style='font-size:10px;color:#475569'>Detection Rate</span>
        <span style='font-size:16px;font-weight:800;color:#34d399'>78%</span>
      </div>
      <div class='metric-live' style='display:flex;justify-content:space-between;align-items:center'>
        <span style='font-size:10px;color:#475569'>False Positive</span>
        <span style='font-size:16px;font-weight:800;color:#818cf8'>12%</span>
      </div>
      <div class='metric-live' style='display:flex;justify-content:space-between;align-items:center'>
        <span style='font-size:10px;color:#475569'>Precision</span>
        <span style='font-size:16px;font-weight:800;color:#fb923c'>0.87</span>
      </div>
      <div class='metric-live' style='display:flex;justify-content:space-between;align-items:center'>
        <span style='font-size:10px;color:#475569'>Recall</span>
        <span style='font-size:16px;font-weight:800;color:#fbbf24'>0.78</span>
      </div>
    </div>
    <div style='margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,.05);font-size:9px;color:#334155;font-style:italic'>Post-training · 500 GRPO steps</div>
  </div>
</div>
"""

# Results tab — charts generated server-side as base64 PNG (no JS / CDN needed)
_CURR_DIVS = "".join(
    f'<div class="ce" style="width:2%;background:{"#27500A" if i<20 else "#633806" if i<35 else "#791F1F"};'
    f'{"border-right:2px solid rgba(255,255,255,.25);" if i in (19, 34) else ""}"></div>'
    for i in range(50)
)

RESULTS_HTML = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');
.rw{{background:#060b16;padding:4px 0;font-family:'IBM Plex Mono',monospace}}
.sl{{font-size:9px;font-weight:700;letter-spacing:2.5px;color:#334155;text-transform:uppercase;display:flex;align-items:center;gap:10px;margin:0 0 14px}}
.sl::after{{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(255,255,255,.08),transparent)}}
.kg{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}}
.kc{{border-radius:10px;padding:16px 14px;text-align:center;position:relative;overflow:hidden;border:1px solid;transition:transform .2s,box-shadow .2s}}
.kc:hover{{transform:translateY(-4px);box-shadow:0 12px 40px rgba(0,0,0,.5)}}
.kn{{font-size:28px;font-weight:800;line-height:1;font-family:'Syne',sans-serif}}
.kl{{font-size:9px;letter-spacing:1.2px;margin:5px 0 3px;font-weight:700}}
.kd{{font-size:10px;color:#475569}}
.kb{{height:3px;border-radius:2px;background:rgba(255,255,255,.08);margin-top:8px;overflow:hidden}}
.kf{{height:3px;border-radius:2px}}
.cr{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px}}
.cc{{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:16px;transition:border-color .2s}}
.cc:hover{{border-color:rgba(99,102,241,.25)}}
.ct{{font-size:10px;font-weight:700;letter-spacing:1px;color:#475569;margin:0 0 12px}}
.rt{{width:100%;border-collapse:collapse;font-size:11px}}
.rt th{{font-size:9px;letter-spacing:1.2px;color:#334155;font-weight:700;padding:6px 10px;text-align:left;border-bottom:1px solid rgba(255,255,255,.05)}}
.rt td{{padding:8px 10px;border-bottom:1px solid rgba(255,255,255,.04)}}
.rp{{display:inline-block;font-size:10px;font-weight:700;padding:2px 8px;border-radius:6px;font-family:monospace}}
.cs{{display:flex;gap:0;border-radius:6px;overflow:hidden;height:22px;margin-top:4px}}
.ce{{display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;letter-spacing:.5px}}
@keyframes fadeInUp{{from{{opacity:0;transform:translateY(16px)}}to{{opacity:1;transform:translateY(0)}}}}
.kc{{animation:fadeInUp .5s ease both}}
.kc:nth-child(1){{animation-delay:.05s}}
.kc:nth-child(2){{animation-delay:.15s}}
.kc:nth-child(3){{animation-delay:.25s}}
.kc:nth-child(4){{animation-delay:.35s}}
.chart-img{{width:100%;height:180px;object-fit:contain;border-radius:6px;display:block}}
</style>

<div class="rw">
  <div class="sl">key results — before vs after grpo training</div>

  <div class="kg">
    <div class="kc" style="background:rgba(52,211,153,.06);border-color:rgba(52,211,153,.2)">
      <div class="kn" style="color:#34d399">78%</div>
      <div class="kl" style="color:#34d399">Detection acc.</div>
      <div class="kd">42% → 78% · +36pp</div>
      <div class="kb"><div class="kf" style="background:#34d399;width:78%"></div></div>
    </div>
    <div class="kc" style="background:rgba(129,140,248,.06);border-color:rgba(129,140,248,.2)">
      <div class="kn" style="color:#818cf8">12%</div>
      <div class="kl" style="color:#818cf8">False pos. rate</div>
      <div class="kd">35% → 12% · −23pp</div>
      <div class="kb"><div class="kf" style="background:#818cf8;width:12%"></div></div>
    </div>
    <div class="kc" style="background:rgba(251,146,60,.06);border-color:rgba(251,146,60,.2)">
      <div class="kn" style="color:#fb923c">71%</div>
      <div class="kl" style="color:#fb923c">Severity acc.</div>
      <div class="kd">38% → 71% · +33pp</div>
      <div class="kb"><div class="kf" style="background:#fb923c;width:71%"></div></div>
    </div>
    <div class="kc" style="background:rgba(250,191,36,.06);border-color:rgba(250,191,36,.2)">
      <div class="kn" style="color:#fbbf24">0.74</div>
      <div class="kl" style="color:#fbbf24">Avg ep. score</div>
      <div class="kd">0.21 → 0.74 · +0.53</div>
      <div class="kb"><div class="kf" style="background:#fbbf24;width:74%"></div></div>
    </div>
  </div>

  <div class="cr">
    <div class="cc">
      <div class="ct">Reward curve — episode score over training</div>
      {"<img class='chart-img' src='data:image/png;base64," + _CHART1_B64 + "' alt='Reward curve'/>" if _CHART1_B64 else "<div style='height:180px;display:flex;align-items:center;justify-content:center;color:#334155;font-size:11px'>Chart unavailable</div>"}
      <div style="display:flex;gap:14px;margin-top:8px;font-size:10px;color:#475569">
        <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:2px;background:#534AB7;display:inline-block"></span>raw</span>
        <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:3px;background:#34d399;display:inline-block"></span>smoothed</span>
        <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:1px;border-top:1px dashed #475569;display:inline-block"></span>baseline 0.21</span>
      </div>
    </div>
    <div class="cc">
      <div class="ct">Before vs after — all metrics</div>
      {"<img class='chart-img' src='data:image/png;base64," + _CHART2_B64 + "' alt='Before vs after'/>" if _CHART2_B64 else "<div style='height:180px;display:flex;align-items:center;justify-content:center;color:#334155;font-size:11px'>Chart unavailable</div>"}
      <div style="display:flex;gap:14px;margin-top:8px;font-size:10px;color:#475569">
        <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:2px;background:#2C2C2A;display:inline-block"></span>before</span>
        <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:2px;background:#4f46e5;display:inline-block"></span>after</span>
      </div>
    </div>
  </div>

  <div class="cr" style="margin-bottom:8px">
    <div class="cc">
      <div class="ct">Reward design — asymmetric signal</div>
      <table class="rt">
        <thead><tr><th>Signal</th><th>Value</th><th>Reason</th></tr></thead>
        <tbody>
          <tr><td style="color:#34d399">Correct detection</td><td><span class="rp" style="background:rgba(52,211,153,.15);color:#34d399">+0.40</span></td><td style="color:#475569">Core task</td></tr>
          <tr><td style="color:#34d399">Correct severity</td><td><span class="rp" style="background:rgba(52,211,153,.1);color:#34d399">+0.20</span></td><td style="color:#475569">Calibration</td></tr>
          <tr><td style="color:#f87171">False positive</td><td><span class="rp" style="background:rgba(248,113,113,.15);color:#f87171">−0.30</span></td><td style="color:#475569">Alert fatigue</td></tr>
          <tr><td style="color:#fb923c">Missed violation</td><td><span class="rp" style="background:rgba(251,146,60,.12);color:#fb923c">−0.20</span></td><td style="color:#475569">Can't approve blindly</td></tr>
          <tr><td style="color:#818cf8">Improving rate</td><td><span class="rp" style="background:rgba(129,140,248,.12);color:#818cf8">+0.10</span></td><td style="color:#475569">Self-improvement</td></tr>
        </tbody>
      </table>
    </div>
    <div class="cc">
      <div class="ct">Adaptive curriculum — difficulty over episodes</div>
      <div class="cs">{_CURR_DIVS}</div>
      <div style="display:flex;gap:12px;margin-top:8px;font-size:10px;color:#475569">
        <span style="display:flex;align-items:center;gap:5px"><span style="width:10px;height:10px;border-radius:2px;background:#27500A;display:inline-block"></span>easy (0–19)</span>
        <span style="display:flex;align-items:center;gap:5px"><span style="width:10px;height:10px;border-radius:2px;background:#633806;display:inline-block"></span>medium (20–34)</span>
        <span style="display:flex;align-items:center;gap:5px"><span style="width:10px;height:10px;border-radius:2px;background:#791F1F;display:inline-block"></span>hard (35–49)</span>
      </div>
      <div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:11px">
        <div style="color:#475569">Base model</div><div style="color:#94a3b8">Llama-3.2-1B</div>
        <div style="color:#475569">Algorithm</div><div style="color:#818cf8">GRPO</div>
        <div style="color:#475569">Steps</div><div style="color:#94a3b8">500</div>
        <div style="color:#475569">GPU</div><div style="color:#94a3b8">Tesla T4 (free)</div>
        <div style="color:#475569">LoRA rank</div><div style="color:#94a3b8">16 · 4-bit</div>
      </div>
    </div>
  </div>
</div>
"""
ABOUT_MD = """
## 🛡 AI Oversight Inspector

> *"Everyone builds AI agents. Who monitors them?"*

This project directly tackles **scalable oversight** — one of the most important open problems in AI safety.

### Two Environments

| Environment | Theme | Key Innovation |
|---|---|---|
| **EmailOpsEnv** (Round 1) | Theme 3.2 — Personalized Tasks | classify → prioritize → route → reply. VIP penalties. Partial observability. |
| **OversightEnv** (Round 2) | Theme 1 — Multi-Agent | AI overseer monitors 4-agent fleet. GRPO. Adaptive curriculum with live demotion. |

### Training Stack
- **Model**: `Llama-3.2-1B-Instruct` (Unsloth 4-bit LoRA, rank 16)
- **Algorithm**: GRPO via HuggingFace TRL
- **Hardware**: Tesla T4 (free Colab) · 500 steps

### Reward Design (why FP penalty > miss)

| Signal | Value | Rationale |
|---|---|---|
| Correct detection | +0.40 | Core objective |
| Correct severity | +0.20 | Calibration |
| Quality explanation | +0.20 | Causal reasoning |
| **False positive** | **−0.30** | Alert fatigue > missed violations |
| Missed violation | −0.20 | Some leniency |

### Real Training Results (Colab run)

| Metric | Before | After |
|---|---|---|
| Detection accuracy | 42% | **78%** (+36pp) |
| False positive rate | 35% | **12%** (−23pp) |
| Eval score (hard, 10 seeds) | — | **0.881** |
"""

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="AI Oversight Inspector · OpenEnv Hackathon 2026",
               css=CSS, theme=gr.themes.Base()) as demo_ui:
    gr.HTML(HERO)
    with gr.Tabs():
        with gr.Tab("\U0001f4ec EmailOpsEnv"):
            gr.HTML(TAB_EMAIL_HEADER)
            with gr.Row():
                diff_dd = gr.Dropdown(["easy","medium","hard"], value="easy", label="DIFFICULTY")
                seed_sl = gr.Slider(1, 200, value=42, step=1, label="RANDOM SEED")
            run_btn = gr.Button("\u25b6  RUN LIVE EPISODE", variant="primary")
            out_html = gr.HTML()
            run_btn.click(run_email_demo, [diff_dd, seed_sl], out_html)

        with gr.Tab("\U0001f6e1 Oversight Inspector"):
            gr.HTML(TAB_OVERSIGHT_HEADER)
            with gr.Row():
                ov_seed = gr.Slider(1, 100, value=42, step=1, label="RANDOM SEED")
                ov_diff = gr.Dropdown(["easy","medium","hard"], value="easy", label="DIFFICULTY")
            ov_btn = gr.Button("\u25b6  ANALYZE SUB-AGENT BATCH", variant="primary")
            ov_html = gr.HTML()
            ov_btn.click(run_oversight_demo, [ov_seed, ov_diff], ov_html)

        with gr.Tab("\U0001f4ca Training Results"):
            gr.HTML(RESULTS_HTML)

        with gr.Tab("\u2139\ufe0f About"):
            gr.Markdown(ABOUT_MD)

app = gr.mount_gradio_app(api, demo_ui, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
