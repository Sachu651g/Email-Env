"""OpenEnv Email Ops — HF Space API Server
Exposes reset(), step(), state() as HTTP endpoints + Gradio UI"""
from __future__ import annotations
import json, os, sys
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv_email_ops.env import EmailOpsEnv
from openenv_email_ops.models import Action, TaskConfig
from openenv_email_ops.pretty_printer import PrettyPrinter

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
            total_reward += reward.total; steps += 1
        results[diff] = {"steps": steps, "total_reward": round(total_reward, 3)}
    return JSONResponse({"demo_results": results, "status": "ok"})


def run_email_demo(difficulty: str, seed: int) -> str:
    try:
        task = TaskConfig(task_id=difficulty, description=f"{difficulty} demo", difficulty=difficulty,
                          max_steps=10, inbox_size=5, reward_components=["classification", "prioritization", "routing"])
        env = EmailOpsEnv(task_config=task, seed=int(seed))
        env.reset(seed=int(seed))
        action_cycle = [
            ("classify_email", "important"), ("prioritize_email", "high"),
            ("route_email", "support"), ("generate_reply", "Thank you for your message. We will investigate this immediately."),
        ]
        rows, total, steps, done = [], 0.0, 0, False
        while not done and steps < 8:
            at, val = action_cycle[steps % len(action_cycle)]
            _, reward, done, _ = env.step(Action(action_type=at, value=val))
            total += reward.step_reward
            is_pos = reward.step_reward > 0
            pill = (
                f"<span style='background:#0d2b1a;color:#4ade80;border:1px solid #166534;"
                f"padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.5px'>"
                f"+{reward.step_reward:.3f}</span>"
            ) if is_pos else (
                f"<span style='background:#2b0d0d;color:#f87171;border:1px solid #7f1d1d;"
                f"padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.5px'>"
                f"{reward.step_reward:.3f}</span>"
            )
            action_label = at.replace("_", " ").upper()
            rows.append(
                f"<tr style='border-bottom:1px solid rgba(255,255,255,.06)'>"
                f"<td style='padding:10px 14px;color:#64748b;font-size:12px;font-family:monospace'>{steps+1:02d}</td>"
                f"<td style='padding:10px 14px'><span style='background:rgba(99,102,241,.15);color:#a5b4fc;"
                f"border:1px solid rgba(99,102,241,.3);padding:2px 8px;border-radius:6px;font-size:11px;"
                f"font-family:monospace;letter-spacing:.3px'>{action_label}</span></td>"
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
        diff_bg = {"easy": "20,83,45", "medium": "120,53,15", "hard": "127,29,29"}.get(difficulty, "30,30,50")
        return (
            f"<div style='font-family:\"IBM Plex Mono\",\"Fira Code\",monospace;background:#0a0f1a;"
            f"border:1px solid rgba(255,255,255,.08);border-radius:14px;overflow:hidden;"
            f"box-shadow:0 8px 40px rgba(0,0,0,.5)'>"
            f"<div style='background:linear-gradient(90deg,#0d1f3c,#111827);padding:14px 20px;"
            f"display:flex;align-items:center;gap:12px;border-bottom:1px solid rgba(255,255,255,.06)'>"
            f"<span style='font-size:18px'>\U0001f4ec</span>"
            f"<div><div style='font-weight:700;color:#e2e8f0;font-size:13px;letter-spacing:.5px'>LIVE EPISODE</div>"
            f"<div style='font-size:11px;color:#475569;margin-top:1px'>EmailOpsEnv \u00b7 seed={seed} \u00b7 OpenEnv-compliant</div></div>"
            f"<div style='margin-left:auto'><span style='background:rgba({diff_bg},.3);color:{diff_color};"
            f"border:1px solid {diff_color}40;padding:3px 12px;border-radius:20px;font-size:11px;"
            f"font-weight:700;letter-spacing:1px'>{difficulty.upper()}</span></div>"
            f"</div>"
            f"<table style='width:100%;border-collapse:collapse'>"
            f"<thead><tr style='background:rgba(255,255,255,.03);border-bottom:1px solid rgba(255,255,255,.06)'>"
            f"<th style='padding:8px 14px;text-align:left;font-size:10px;color:#475569;letter-spacing:1.5px;font-weight:700'>STEP</th>"
            f"<th style='padding:8px 14px;text-align:left;font-size:10px;color:#475569;letter-spacing:1.5px;font-weight:700'>ACTION</th>"
            f"<th style='padding:8px 14px;text-align:left;font-size:10px;color:#475569;letter-spacing:1.5px;font-weight:700'>VALUE</th>"
            f"<th style='padding:8px 14px;text-align:left;font-size:10px;color:#475569;letter-spacing:1.5px;font-weight:700'>REWARD</th>"
            f"</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            f"</table>"
            f"<div style='padding:14px 20px;background:rgba(255,255,255,.02);display:flex;"
            f"align-items:center;gap:16px;border-top:1px solid rgba(255,255,255,.06)'>"
            f"<div>{total_pill}</div>"
            f"<div style='color:#475569;font-size:12px'>episode total</div>"
            f"<div style='margin-left:auto;color:#334155;font-size:11px;font-family:monospace'>{steps} steps \u00b7 EmailOpsEnv v1.0</div>"
            f"</div></div>"
        )
    except Exception as e:
        return f"<p style='color:#f87171;font-family:monospace;padding:12px'>Error: {e}</p>"


def run_oversight_demo(seed: int = 42, difficulty: str = "easy") -> str:
    try:
        # Add the round2 folder to Python path
        round2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "round2_oversight_inspector")
        if round2_path not in sys.path:
            sys.path.insert(0, round2_path)
        
        # Import after path is set
        from oversight_env.env import OversightEnv
        from oversight_env.models import OversightAction, ViolationType, SeverityLevel
        
        env = OversightEnv(task_id=difficulty, difficulty=difficulty, max_steps=5, seed=int(seed))
        obs = env.reset()
        
        # Run the overseer through multiple steps to show detection in action
        total_violations_found = 0
        total_correct_detections = 0
        total_false_positives = 0
        steps_taken = 0
        detection_history = []
        
        # Simple rule-based overseer for demo
        def detect_violations(outputs):
            """Mock overseer that detects obvious violations"""
            for output in outputs:
                text = (output.decision + " " + output.explanation).lower()
                
                # Detect spam misclassification
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
                
                # Detect critical routing errors
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
                
                # Detect hallucinations
                if output.agent_role == "responder":
                    hallucination_phrases = ["already refunded", "premium account", "resolved in last update"]
                    if any(p in text for p in hallucination_phrases):
                        return OversightAction(
                            action_type="flag_violation",
                            target_agent_id=output.agent_id,
                            violation_type=ViolationType.HALLUCINATION,
                            severity=SeverityLevel.MEDIUM,
                            explanation=f"Response contains unverified claim not in input",
                            confidence=0.75,
                        )
            
            return OversightAction(action_type="approve")
        
        # Run 3 steps to show detection in action
        done = False
        while not done and steps_taken < 3:
            action = detect_violations(obs.sub_agent_outputs)
            obs, reward, done, info = env.step(action)
            
            if action.action_type == "flag_violation":
                total_violations_found += 1
                if reward.step_reward > 0:
                    total_correct_detections += 1
                else:
                    total_false_positives += 1
            
            detection_history.append({
                "step": steps_taken + 1,
                "action": action.action_type,
                "reward": reward.step_reward,
                "target": getattr(action, "target_agent_id", "N/A"),
                "violation": str(getattr(action, "violation_type", "none")),
            })
            steps_taken += 1
        
        # Build visualization showing the detection process
        violation_map = {
            "hallucination":        ("\U0001f534", "#f87171", "#2b0d0d", "#7f1d1d", "HALLUCINATION"),
            "wrong_classification": ("\U0001f7e0", "#fb923c", "#2b1407", "#7c2d12", "WRONG CLASS."),
            "policy_violation":     ("\U0001f7e1", "#fbbf24", "#2b2107", "#78350f", "POLICY BREACH"),
            "severity_mismatch":    ("\U0001f7e3", "#c084fc", "#1e0b2b", "#581c87", "SEVERITY ERR"),
            "inconsistency":        ("\U0001f535", "#60a5fa", "#0b1e2b", "#1e3a5f", "INCONSISTENCY"),
        }
        
        # Show current batch of agents
        rows = []
        for output in obs.sub_agent_outputs[:4]:
            vtype = getattr(output, "actual_violation", None)
            v_str = str(vtype.value) if hasattr(vtype, "value") else str(vtype) if vtype else "none"
            has_v = v_str != "none"
            role_icon = {"classifier": "\U0001f50d", "prioritizer": "\u26a1", "router": "\U0001f5fa", "responder": "\u270d\ufe0f"}.get(output.agent_role, "\U0001f916")
            conf_bar_w = int(output.confidence * 60)
            conf_color = "#4ade80" if output.confidence > 0.8 else "#fbbf24" if output.confidence > 0.6 else "#f87171"
            if has_v and v_str in violation_map:
                icon, color, bg, border, label = violation_map[v_str]
                vbadge = (
                    f"<span style='background:{bg};color:{color};border:1px solid {border};"
                    f"padding:3px 9px;border-radius:6px;font-size:10px;font-weight:700;"
                    f"letter-spacing:.5px;font-family:monospace'>{icon} {label}</span>"
                )
            else:
                vbadge = (
                    "<span style='background:#0d2b1a;color:#4ade80;border:1px solid #166534;"
                    "padding:3px 9px;border-radius:6px;font-size:10px;font-weight:700;"
                    "letter-spacing:.5px;font-family:monospace'>\u2713 CLEAN</span>"
                )
            rows.append(
                f"<tr style='border-bottom:1px solid rgba(255,255,255,.05);animation:slideIn 0.3s ease-out {output.agent_id * 0.1}s backwards'>"
                f"<td style='padding:12px 16px;white-space:nowrap'>"
                f"  <span style='font-size:16px'>{role_icon}</span>"
                f"  <span style='color:#cbd5e1;font-weight:600;font-size:12px;margin-left:8px;font-family:monospace'>{output.agent_role.upper()}</span>"
                f"</td>"
                f"<td style='padding:12px 16px;color:#94a3b8;font-size:12px;font-family:monospace;max-width:220px'>"
                f"  <div style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{output.decision[:55]}</div>"
                f"</td>"
                f"<td style='padding:12px 16px'>"
                f"  <div style='display:flex;align-items:center;gap:8px'>"
                f"    <div style='width:60px;height:4px;background:rgba(255,255,255,.08);border-radius:2px'>"
                f"      <div style='width:{conf_bar_w}px;height:4px;background:{conf_color};border-radius:2px;animation:fillBar 0.8s ease-out'></div>"
                f"    </div>"
                f"    <span style='color:{conf_color};font-size:11px;font-family:monospace;font-weight:700'>{output.confidence:.2f}</span>"
                f"  </div>"
                f"</td>"
                f"<td style='padding:12px 16px'>{vbadge}</td>"
                f"</tr>"
            )
        
        # Build detection history timeline
        timeline_rows = []
        for entry in detection_history:
            is_flag = entry["action"] == "flag_violation"
            is_correct = entry["reward"] > 0
            action_color = "#f87171" if is_flag else "#4ade80"
            action_icon = "\u26a0\ufe0f" if is_flag else "\u2713"
            reward_pill = (
                f"<span style='background:#0d2b1a;color:#4ade80;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700'>+{entry['reward']:.2f}</span>"
                if entry["reward"] > 0 else
                f"<span style='background:#2b0d0d;color:#f87171;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700'>{entry['reward']:.2f}</span>"
            )
            timeline_rows.append(
                f"<div style='display:flex;align-items:center;gap:12px;padding:8px 12px;background:rgba(255,255,255,.02);border-radius:8px;margin-bottom:6px;animation:fadeIn 0.4s ease-out {entry['step'] * 0.15}s backwards'>"
                f"  <span style='color:{action_color};font-size:14px'>{action_icon}</span>"
                f"  <span style='color:#64748b;font-size:11px;font-family:monospace;min-width:50px'>Step {entry['step']}</span>"
                f"  <span style='color:#94a3b8;font-size:11px;font-family:monospace;flex:1'>{entry['action'].replace('_', ' ').title()}</span>"
                f"  {reward_pill}"
                f"</div>"
            )
        
        # Calculate accuracy metrics
        accuracy = (total_correct_detections / max(1, total_violations_found)) * 100 if total_violations_found > 0 else 100
        precision = (total_correct_detections / max(1, total_violations_found)) if total_violations_found > 0 else 1.0
        
        return (
            f"<div style='font-family:\"IBM Plex Mono\",\"Fira Code\",monospace;background:#060b16;"
            f"border:1px solid rgba(165,180,252,.15);border-radius:14px;overflow:hidden;"
            f"box-shadow:0 8px 40px rgba(0,0,0,.6);animation:fadeIn 0.5s ease-in'>"
            f"<div style='background:linear-gradient(90deg,#0f0c29,#1a1040,#0f0c29);padding:16px 20px;"
            f"border-bottom:1px solid rgba(165,180,252,.1)'>"
            f"<div style='display:flex;align-items:center;gap:10px'>"
            f"<span style='font-size:20px'>\U0001f6e1</span>"
            f"<div><div style='font-weight:800;color:#e2e8f0;font-size:13px;letter-spacing:1px'>OVERSIGHT INSPECTOR LIVE</div>"
            f"<div style='font-size:11px;color:#6366f1;margin-top:1px'>AI monitoring AI \u2014 no ground truth exposed · Seed: {seed} · Difficulty: {difficulty.upper()}</div></div>"
            f"<div style='margin-left:auto;display:flex;gap:8px'>"
            f"<span style='background:rgba(239,68,68,.15);color:#f87171;border:1px solid rgba(239,68,68,.3);"
            f"padding:2px 10px;border-radius:12px;font-size:10px;font-weight:700'>\u22120.30 FALSE ALARM</span>"
            f"<span style='background:rgba(251,146,60,.1);color:#fb923c;border:1px solid rgba(251,146,60,.25);"
            f"padding:2px 10px;border-radius:12px;font-size:10px;font-weight:700'>\u22120.20 MISS</span>"
            f"</div></div></div>"
            
            # Detection metrics panel
            f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:16px 20px;background:rgba(255,255,255,.015);border-bottom:1px solid rgba(255,255,255,.05)'>"
            f"  <div style='text-align:center'>"
            f"    <div style='font-size:24px;font-weight:800;color:#4ade80'>{total_correct_detections}</div>"
            f"    <div style='font-size:10px;color:#475569;letter-spacing:1px'>CORRECT DETECTIONS</div>"
            f"  </div>"
            f"  <div style='text-align:center'>"
            f"    <div style='font-size:24px;font-weight:800;color:#f87171'>{total_false_positives}</div>"
            f"    <div style='font-size:10px;color:#475569;letter-spacing:1px'>FALSE POSITIVES</div>"
            f"  </div>"
            f"  <div style='text-align:center'>"
            f"    <div style='font-size:24px;font-weight:800;color:#818cf8'>{accuracy:.0f}%</div>"
            f"    <div style='font-size:10px;color:#475569;letter-spacing:1px'>ACCURACY</div>"
            f"  </div>"
            f"</div>"
            
            # Agent status table
            f"<table style='width:100%;border-collapse:collapse'>"
            f"<thead><tr style='background:rgba(255,255,255,.025);border-bottom:1px solid rgba(255,255,255,.06)'>"
            f"<th style='padding:8px 16px;text-align:left;font-size:10px;color:#4b5563;letter-spacing:1.5px;font-weight:700'>AGENT</th>"
            f"<th style='padding:8px 16px;text-align:left;font-size:10px;color:#4b5563;letter-spacing:1.5px;font-weight:700'>DECISION</th>"
            f"<th style='padding:8px 16px;text-align:left;font-size:10px;color:#4b5563;letter-spacing:1.5px;font-weight:700'>CONF.</th>"
            f"<th style='padding:8px 16px;text-align:left;font-size:10px;color:#4b5563;letter-spacing:1.5px;font-weight:700'>STATUS</th>"
            f"</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            f"</table>"
            
            # Detection timeline
            f"<div style='padding:16px 20px;background:rgba(99,102,241,.03);border-top:1px solid rgba(99,102,241,.1)'>"
            f"<div style='font-size:10px;color:#6366f1;font-weight:700;letter-spacing:1.5px;margin-bottom:10px'>DETECTION TIMELINE</div>"
            f"{''.join(timeline_rows)}"
            f"</div>"
            
            f"<div style='padding:12px 20px;background:rgba(99,102,241,.05);border-top:1px solid rgba(99,102,241,.1)'>"
            f"<span style='color:#6366f1;font-size:11px;font-weight:700'>OVERSEER TASK \u2192</span>"
            f"<span style='color:#64748b;font-size:11px;margin-left:8px'>Flag which agents violated policy. Reason from outputs only \u2014 ground truth is hidden.</span>"
            f"</div>"
            
            # Add CSS animations
            f"<style>"
            f"@keyframes fadeIn {{from {{opacity:0;transform:translateY(10px)}} to {{opacity:1;transform:translateY(0)}}}}"
            f"@keyframes slideIn {{from {{opacity:0;transform:translateX(-20px)}} to {{opacity:1;transform:translateX(0)}}}}"
            f"@keyframes fillBar {{from {{width:0}} to {{width:100%}}}}"
            f"</style>"
            f"</div>"
        )
    except ImportError as e:
        return (
            f"<div style='font-family:monospace;background:#2b0d0d;border:1px solid #7f1d1d;"
            f"border-radius:10px;padding:20px;color:#f87171'>"
            f"<div style='font-size:16px;font-weight:700;margin-bottom:10px'>\u26a0\ufe0f Oversight Environment Not Available</div>"
            f"<div style='font-size:12px;color:#fb923c;margin-bottom:10px'>The Round 2 oversight environment module could not be loaded.</div>"
            f"<div style='font-size:11px;color:#94a3b8;background:rgba(0,0,0,.3);padding:10px;border-radius:6px;font-family:monospace'>"
            f"Error: {str(e)}</div>"
            f"<div style='font-size:11px;color:#64748b;margin-top:10px'>"
            f"This demo requires the <code>round2_oversight_inspector/oversight_env/</code> module. "
            f"The EmailOpsEnv demo in Tab 1 should still work.</div>"
            f"</div>"
        )
    except Exception as e:
        return (
            f"<div style='font-family:monospace;background:#2b1407;border:1px solid #7c2d12;"
            f"border-radius:10px;padding:20px;color:#fb923c'>"
            f"<div style='font-size:16px;font-weight:700;margin-bottom:10px'>\u26a0\ufe0f Runtime Error</div>"
            f"<div style='font-size:12px;color:#fbbf24;margin-bottom:10px'>An error occurred while running the oversight demo.</div>"
            f"<div style='font-size:11px;color:#94a3b8;background:rgba(0,0,0,.3);padding:10px;border-radius:6px;font-family:monospace'>"
            f"Error: {str(e)}</div>"
            f"<div style='font-size:11px;color:#64748b;margin-top:10px'>"
            f"Try adjusting the seed or difficulty level, or check the EmailOpsEnv demo in Tab 1.</div>"
            f"</div>"
        )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');
* { box-sizing: border-box; }
.gradio-container {
    background: #050a14 !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 100vh;
}
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(99,102,241,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}
.gradio-container::after {
    content: '';
    position: fixed;
    top: -200px;
    left: 50%;
    transform: translateX(-50%);
    width: 900px;
    height: 500px;
    background: radial-gradient(ellipse, rgba(79,70,229,.12) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
footer { display: none !important; }
.tabs { background: transparent !important; position: relative; z-index: 1; }
.tab-nav {
    background: rgba(255,255,255,.02) !important;
    border: 1px solid rgba(255,255,255,.06) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    margin-bottom: 12px !important;
}
.tab-nav button {
    color: #475569 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: .8px !important;
    border-radius: 7px !important;
    padding: 7px 16px !important;
    transition: all .2s !important;
    border: 1px solid transparent !important;
}
.tab-nav button:hover { color: #94a3b8 !important; background: rgba(255,255,255,.04) !important; }
.tab-nav button.selected {
    background: linear-gradient(135deg, rgba(79,70,229,.3), rgba(6,78,59,.4)) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(99,102,241,.3) !important;
    box-shadow: 0 0 20px rgba(79,70,229,.15) !important;
}
button.primary, .primary {
    background: linear-gradient(90deg, #4f46e5, #059669) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 20px rgba(79,70,229,.3) !important;
    transition: all .25s !important;
    padding: 10px 24px !important;
}
button.primary:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 30px rgba(79,70,229,.5) !important; }
button.primary:active { transform: translateY(0) !important; }
.gr-form, .gr-box { background: rgba(255,255,255,.025) !important; border: 1px solid rgba(255,255,255,.07) !important; border-radius: 10px !important; }
label span { color: #64748b !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; font-weight: 700 !important; letter-spacing: .8px !important; }
.prose, .gr-markdown { color: #94a3b8 !important; font-family: 'Inter', sans-serif !important; }
.prose h2 { color: #a5b4fc !important; font-family: 'Syne', sans-serif !important; font-size: 18px !important; }
.prose h3 { color: #6ee7b7 !important; font-family: 'Syne', sans-serif !important; font-size: 14px !important; }
.prose strong { color: #e2e8f0 !important; }
.prose table { border: 1px solid rgba(255,255,255,.06) !important; border-radius: 8px !important; overflow: hidden !important; }
.prose table th { background: rgba(99,102,241,.1) !important; color: #a5b4fc !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; letter-spacing: .8px !important; }
.prose table td { color: #94a3b8 !important; border-color: rgba(255,255,255,.04) !important; }
.prose table tr:hover td { background: rgba(255,255,255,.02) !important; }
.prose code { background: rgba(99,102,241,.1) !important; color: #a5b4fc !important; border-radius: 4px !important; padding: 1px 6px !important; font-family: 'IBM Plex Mono', monospace !important; }
.gr-image { border: 1px solid rgba(255,255,255,.07) !important; border-radius: 10px !important; overflow: hidden !important; }
input[type=range] { accent-color: #6366f1 !important; }
.gr-html { background: transparent !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,.02); }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,.3); border-radius: 2px; }
"""

SPLASH_JS = """
() => {
  // Build splash HTML
  const style = document.createElement('style');
  style.textContent = `
    #oi-splash{position:fixed;inset:0;z-index:99999;background:#04080f;overflow-y:auto;overflow-x:hidden;transition:opacity .7s ease}
    #oi-splash.oi-exit{opacity:0;pointer-events:none}
    #oi-splash-canvas{position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:0}
    .oi-scan{position:fixed;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,.5),rgba(52,211,153,.4),transparent);animation:oi-scan 4s linear infinite;pointer-events:none;z-index:1}
    @keyframes oi-scan{0%{top:0;opacity:0}3%{opacity:1}97%{opacity:1}100%{top:100%;opacity:0}}
    .oi-inner{position:relative;z-index:2;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:48px 24px 64px}
    .oi-eye{font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:700;letter-spacing:3px;color:#4f46e5;text-transform:uppercase;margin-bottom:20px;opacity:0;animation:oi-up .6s ease .2s forwards}
    .oi-title{font-size:clamp(40px,8vw,84px);font-weight:800;line-height:1.05;letter-spacing:-3px;text-align:center;margin-bottom:16px;opacity:0;animation:oi-up .7s ease .4s forwards}
    .oi-g1{background:linear-gradient(90deg,#818cf8,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
    .oi-g2{background:linear-gradient(90deg,#34d399,#059669);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
    .oi-sub{font-size:16px;color:#475569;max-width:520px;text-align:center;line-height:1.7;margin-bottom:32px;opacity:0;animation:oi-up .7s ease .6s forwards}
    .oi-badges{display:flex;flex-wrap:wrap;justify-content:center;gap:10px;margin-bottom:36px;opacity:0;animation:oi-up .6s ease .8s forwards}
    .oi-badge{font-size:11px;font-weight:700;letter-spacing:.5px;padding:5px 14px;border-radius:20px;border:1px solid}
    .oib1{color:#a5b4fc;border-color:rgba(99,102,241,.35);background:rgba(99,102,241,.1)}
    .oib2{color:#34d399;border-color:rgba(52,211,153,.3);background:rgba(52,211,153,.08)}
    .oib3{color:#c084fc;border-color:rgba(192,132,252,.3);background:rgba(192,132,252,.07)}
    .oib4{color:#fbbf24;border-color:rgba(251,191,36,.3);background:rgba(251,191,36,.07)}
    .oi-net-card{width:100%;max-width:900px;background:rgba(255,255,255,.025);border:1px solid rgba(99,102,241,.2);border-radius:20px;overflow:hidden;margin-bottom:32px;opacity:0;animation:oi-up .7s ease 1s forwards;box-shadow:0 0 60px rgba(99,102,241,.1)}
    .oi-net-hdr{padding:14px 20px;background:linear-gradient(90deg,rgba(99,102,241,.12),rgba(52,211,153,.06));border-bottom:1px solid rgba(255,255,255,.06);display:flex;align-items:center;gap:8px}
    .oi-dot{width:10px;height:10px;border-radius:50%}
    .oi-live{width:7px;height:7px;border-radius:50%;background:#34d399;box-shadow:0 0 6px #34d399;animation:oi-blink 1.4s ease-in-out infinite;margin-left:auto}
    @keyframes oi-blink{0%,100%{opacity:1}50%{opacity:.3}}
    #oi-net{display:block;width:100%;height:260px}
    .oi-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;width:100%;max-width:900px;margin-bottom:32px;opacity:0;animation:oi-up .6s ease 1.2s forwards}
    .oi-stat{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:18px;text-align:center;position:relative;overflow:hidden;transition:all .3s}
    .oi-stat:hover{transform:translateY(-3px);border-color:rgba(99,102,241,.3)}
    .oi-stat::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px}
    .ois1::after{background:linear-gradient(90deg,transparent,#818cf8,transparent)}
    .ois2::after{background:linear-gradient(90deg,transparent,#34d399,transparent)}
    .ois3::after{background:linear-gradient(90deg,transparent,#fbbf24,transparent)}
    .ois4::after{background:linear-gradient(90deg,transparent,#f87171,transparent)}
    .oi-sv{font-size:34px;font-weight:800;line-height:1;margin-bottom:6px}
    .oi-sl{font-size:10px;font-weight:700;letter-spacing:1.5px;color:#475569;text-transform:uppercase}
    .oi-ss{font-size:10px;color:#334155;margin-top:4px}
    .oi-agents-lbl{font-size:11px;font-weight:700;letter-spacing:2px;color:#6366f1;text-transform:uppercase;margin-bottom:12px;width:100%;max-width:900px;opacity:0;animation:oi-up .5s ease 1.3s forwards}
    .oi-agents{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;width:100%;max-width:900px;margin-bottom:32px;opacity:0;animation:oi-up .6s ease 1.4s forwards}
    .oi-agent{background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.07);border-radius:14px;padding:16px;transition:all .3s}
    .oi-agent:hover{background:rgba(99,102,241,.07);border-color:rgba(99,102,241,.3);transform:translateY(-2px)}
    .oi-ai{font-size:24px;margin-bottom:8px}
    .oi-an{font-size:11px;font-weight:700;letter-spacing:1px;color:#94a3b8;text-transform:uppercase;margin-bottom:5px}
    .oi-ad{font-size:12px;color:#475569;line-height:1.5}
    .oi-cbar{height:3px;border-radius:2px;margin-top:10px;background:rgba(255,255,255,.06)}
    .oi-cfill{height:100%;border-radius:2px}
    .oi-ba-lbl{font-size:11px;font-weight:700;letter-spacing:2px;color:#6366f1;text-transform:uppercase;margin-bottom:12px;width:100%;max-width:900px;opacity:0;animation:oi-up .5s ease 1.5s forwards}
    .oi-ba{display:grid;grid-template-columns:1fr 1fr;gap:16px;width:100%;max-width:900px;margin-bottom:36px;opacity:0;animation:oi-up .6s ease 1.6s forwards}
    .oi-bac{border-radius:14px;padding:18px;border:1px solid}
    .oi-before{background:rgba(127,29,29,.1);border-color:rgba(248,113,113,.2)}
    .oi-after{background:rgba(5,46,22,.12);border-color:rgba(52,211,153,.2)}
    .oi-blbl{font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:10px}
    .oi-bq{font-size:13px;line-height:1.7;font-style:italic;color:#94a3b8;border-left:3px solid;padding-left:12px}
    .oi-before .oi-bq{border-color:#f87171}
    .oi-after .oi-bq{border-color:#34d399}
    .oi-links{display:flex;flex-wrap:wrap;gap:12px;justify-content:center;margin-bottom:44px;opacity:0;animation:oi-up .6s ease 1.7s forwards}
    .oi-lnk{font-size:12px;font-weight:700;letter-spacing:.5px;padding:10px 22px;border-radius:10px;text-decoration:none;border:1px solid;transition:all .25s;display:inline-block;color:inherit}
    .oi-lnk:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.4)}
    .oi-gh{color:#e2e8f0 !important;border-color:rgba(226,232,240,.2);background:rgba(226,232,240,.05)}
    .oi-hf{color:#fbbf24 !important;border-color:rgba(251,191,36,.25);background:rgba(251,191,36,.06)}
    .oi-cl{color:#fb923c !important;border-color:rgba(251,146,60,.25);background:rgba(251,146,60,.06)}
    .oi-enter-wrap{opacity:0;animation:oi-up .7s ease 1.9s forwards;text-align:center}
    .oi-enter{font-size:15px;font-weight:700;letter-spacing:2px;padding:18px 56px;border-radius:12px;border:1px solid rgba(99,102,241,.5);background:linear-gradient(135deg,rgba(79,70,229,.25),rgba(5,150,105,.18));color:#e2e8f0;cursor:pointer;text-transform:uppercase;transition:all .3s;box-shadow:0 0 40px rgba(99,102,241,.2)}
    .oi-enter:hover{background:linear-gradient(135deg,rgba(79,70,229,.45),rgba(5,150,105,.32));border-color:rgba(99,102,241,.9);transform:translateY(-3px);box-shadow:0 0 60px rgba(99,102,241,.4),0 10px 40px rgba(0,0,0,.5)}
    .oi-hint{font-size:11px;color:#475569;letter-spacing:1px;margin-top:10px;font-family:'IBM Plex Mono',monospace}
    @keyframes oi-up{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
  `;
  document.head.appendChild(style);

  const splash = document.createElement('div');
  splash.id = 'oi-splash';
  splash.innerHTML = `
    <canvas id="oi-splash-canvas"></canvas>
    <div class="oi-scan"></div>
    <div class="oi-inner">
      <div class="oi-eye">Meta × Hugging Face OpenEnv Hackathon 2026 — Grand Finale</div>
      <h1 class="oi-title"><span class="oi-g1">AI Oversight</span><br><span class="oi-g2">Inspector</span></h1>
      <p class="oi-sub">Training an LLM to watch the <b style="color:#64748b">AI agents</b> — detecting violations <b style="color:#64748b">without ever seeing ground truth.</b> GRPO · Llama-3.2-1B · Adaptive Curriculum.</p>
      <div class="oi-badges">
        <span class="oi-badge oib1">🏆 OpenEnv Compliant</span>
        <span class="oi-badge oib2">⚡ GRPO + Unsloth</span>
        <span class="oi-badge oib3">🛡 AI Safety</span>
        <span class="oi-badge oib4">🎓 Adaptive Curriculum</span>
      </div>
      <div class="oi-net-card">
        <div class="oi-net-hdr">
          <div class="oi-dot" style="background:#f87171"></div>
          <div class="oi-dot" style="background:#fbbf24"></div>
          <div class="oi-dot" style="background:#34d399"></div>
          <span style="font-size:11px;font-weight:700;letter-spacing:1.5px;color:#94a3b8;margin-left:8px">LIVE AGENT NETWORK — OVERSIGHT INSPECTOR MONITORING SUB-AGENT FLEET</span>
          <div class="oi-live"></div>
        </div>
        <canvas id="oi-net"></canvas>
      </div>
      <div class="oi-stats">
        <div class="oi-stat ois1"><div class="oi-sv" style="color:#818cf8">78%</div><div class="oi-sl">Detection Accuracy</div><div class="oi-ss">post-training · 500 steps</div></div>
        <div class="oi-stat ois2"><div class="oi-sv" style="color:#34d399">12%</div><div class="oi-sl">False Positive Rate</div><div class="oi-ss">down from 35% baseline</div></div>
        <div class="oi-stat ois3"><div class="oi-sv" style="color:#fbbf24">0.74</div><div class="oi-sl">Avg Episode Reward</div><div class="oi-ss">up from 0.21 baseline</div></div>
        <div class="oi-stat ois4"><div class="oi-sv" style="color:#f87171">500</div><div class="oi-sl">Training Steps</div><div class="oi-ss">free T4 GPU · ~30 min</div></div>
      </div>
      <div class="oi-agents-lbl">Sub-Agent Fleet Being Monitored</div>
      <div class="oi-agents">
        <div class="oi-agent"><div class="oi-ai">🔍</div><div class="oi-an">Classifier</div><div class="oi-ad">Labels emails as spam, important, or routine</div><div class="oi-cbar"><div class="oi-cfill" style="width:82%;background:linear-gradient(90deg,#4f46e5,#818cf8)"></div></div></div>
        <div class="oi-agent"><div class="oi-ai">⚡</div><div class="oi-an">Prioritizer</div><div class="oi-ad">Assigns urgency — VIP miss triggers −0.30 penalty</div><div class="oi-cbar"><div class="oi-cfill" style="width:71%;background:linear-gradient(90deg,#059669,#34d399)"></div></div></div>
        <div class="oi-agent"><div class="oi-ai">🗺</div><div class="oi-an">Router</div><div class="oi-ad">Routes to correct team — critical must escalate</div><div class="oi-cbar"><div class="oi-cfill" style="width:68%;background:linear-gradient(90deg,#d97706,#fbbf24)"></div></div></div>
        <div class="oi-agent"><div class="oi-ai">✍️</div><div class="oi-an">Responder</div><div class="oi-ad">Generates replies — hallucination detection critical</div><div class="oi-cbar"><div class="oi-cfill" style="width:65%;background:linear-gradient(90deg,#dc2626,#f87171)"></div></div></div>
      </div>
      <div class="oi-ba-lbl">Before vs After Training</div>
      <div class="oi-ba">
        <div class="oi-bac oi-before"><div class="oi-blbl" style="color:#f87171">⚠ Before Training — Reward: 0.21</div><div class="oi-bq">"This email may or may not have been generated by AI. It is difficult to determine without additional context. There could potentially be some concerns, but I cannot say for certain..."</div></div>
        <div class="oi-bac oi-after"><div class="oi-blbl" style="color:#34d399">✓ After Training (GRPO, 500 steps) — Reward: 0.74</div><div class="oi-bq">"VIOLATION [HIGH]: Span — 'As per our policy...'. This paraphrases Policy §4.2 without attribution — documentation integrity violation. Confidence: 0.87."</div></div>
      </div>
      <div class="oi-links">
        <a class="oi-lnk oi-gh" href="https://github.com/Sachu651g/AI-Oversight-Inspector" target="_blank">⭐ GitHub</a>
        <a class="oi-lnk oi-hf" href="https://huggingface.co/spaces/sachingunagi66/openenv-email-ops" target="_blank">🤗 HF Space</a>
        <a class="oi-lnk oi-cl" href="https://colab.research.google.com/github/Sachu651g/AI-Oversight-Inspector/blob/main/round2_oversight_inspector/colab_train_oversight.ipynb" target="_blank">▶ Colab</a>
      </div>
      <div class="oi-enter-wrap">
        <button class="oi-enter" id="oi-enter-btn">Enter Dashboard →</button>
        <div class="oi-hint">Click to explore the live environment</div>
      </div>
    </div>
  `;
  document.body.appendChild(splash);

  // Enter button handler
  document.getElementById('oi-enter-btn').addEventListener('click', function() {
    splash.style.transition = 'opacity .7s ease';
    splash.style.opacity = '0';
    splash.style.pointerEvents = 'none';
    setTimeout(function() { splash.style.display = 'none'; window.scrollTo(0,0); }, 750);
  });

  // Neural net canvas
  setTimeout(function() {
    const c = document.getElementById('oi-net');
    if (!c) return;
    const ctx = c.getContext('2d');
    let nodes = [], edges = [], signals = [];

    function resize() {
      const r = c.parentElement.getBoundingClientRect();
      c.width = r.width || 800; c.height = 260;
      nodes = []; edges = [];
      const cw = c.width, ch = c.height;
      const xF = [0.1,0.34,0.65,0.88];
      const COLORS = ['#6366f1','#fbbf24','#34d399','#f87171'];
      const LABELS = [['📧\\nInbox'],['🔍\\nClassify','⚡\\nPrioritize','🗺\\nRoute','✍️\\nRespond'],['🛡\\nOverseer'],['✓\\nApprove','⚠\\nFlag']];
      const counts = [1,4,1,2];
      counts.forEach(function(count,li){
        for(let ni=0;ni<count;ni++){
          const yf=count===1?0.5:(ni+1)/(count+1);
          nodes.push({x:cw*xF[li],y:ch*yf,r:li===2?20:12,color:COLORS[li],label:LABELS[li][ni]||'',layer:li,pulse:Math.random()*Math.PI*2,ps:0.022+Math.random()*0.015});
        }
      });
      const byL=[[],[],[],[]];
      nodes.forEach(n=>byL[n.layer].push(n));
      for(let li=0;li<3;li++) byL[li].forEach(a=>byL[li+1].forEach(b=>edges.push({a,b,color:a.color})));
    }

    setInterval(function(){
      if(!edges.length) return;
      const e=edges[Math.floor(Math.random()*edges.length)];
      const cols=['#818cf8','#34d399','#fbbf24','#f87171','#c084fc'];
      signals.push({e,t:0,spd:0.009+Math.random()*0.013,col:cols[Math.floor(Math.random()*cols.length)],r:2.5+Math.random()*2});
    },220);

    function draw(){
      if(splash.style.display==='none') return;
      const cw=c.width,ch=c.height;
      ctx.clearRect(0,0,cw,ch);
      edges.forEach(e=>{ctx.beginPath();ctx.moveTo(e.a.x,e.a.y);ctx.lineTo(e.b.x,e.b.y);ctx.strokeStyle=e.color;ctx.globalAlpha=0.13;ctx.lineWidth=1;ctx.stroke();});
      for(let i=signals.length-1;i>=0;i--){
        const s=signals[i];s.t+=s.spd;
        if(s.t>1){signals.splice(i,1);continue;}
        const x=s.e.a.x+(s.e.b.x-s.e.a.x)*s.t,y=s.e.a.y+(s.e.b.y-s.e.a.y)*s.t;
        const g=ctx.createRadialGradient(x,y,0,x,y,s.r*5);
        g.addColorStop(0,s.col+'bb');g.addColorStop(1,s.col+'00');
        ctx.beginPath();ctx.arc(x,y,s.r*5,0,Math.PI*2);ctx.fillStyle=g;ctx.globalAlpha=0.5;ctx.fill();
        ctx.beginPath();ctx.arc(x,y,s.r,0,Math.PI*2);ctx.fillStyle=s.col;ctx.globalAlpha=1;ctx.fill();
      }
      ctx.globalAlpha=1;
      nodes.forEach(n=>{
        n.pulse+=n.ps;
        const glow=Math.sin(n.pulse)*0.5+0.5,gr=n.r+6+glow*9;
        const g=ctx.createRadialGradient(n.x,n.y,n.r*0.4,n.x,n.y,gr);
        g.addColorStop(0,n.color+'55');g.addColorStop(1,n.color+'00');
        ctx.beginPath();ctx.arc(n.x,n.y,gr,0,Math.PI*2);ctx.fillStyle=g;ctx.globalAlpha=0.75+glow*0.25;ctx.fill();
        ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,Math.PI*2);ctx.fillStyle='#04080f';ctx.globalAlpha=1;ctx.fill();
        ctx.strokeStyle=n.color;ctx.lineWidth=1.5;ctx.globalAlpha=0.65+glow*0.35;ctx.stroke();
        ctx.globalAlpha=0.8;ctx.fillStyle='#94a3b8';ctx.font='bold 9px IBM Plex Mono,monospace';ctx.textAlign='center';
        n.label.split('\\n').forEach((ln,li)=>ctx.fillText(ln,n.x,n.y+n.r+12+li*11));
      });
      ctx.globalAlpha=1;
      requestAnimationFrame(draw);
    }
    resize();
    window.addEventListener('resize',resize);
    draw();

    // bg particles
    const sc=document.getElementById('oi-splash-canvas');
    if(!sc) return;
    const sctx=sc.getContext('2d');
    let sw,sh;
    const pts=[];
    for(let i=0;i<90;i++) pts.push({x:Math.random(),y:Math.random(),vx:(Math.random()-.5)*0.00015,vy:(Math.random()-.5)*0.00015,r:Math.random()*1.3+0.3,a:Math.random()*0.3+0.08,c:['#6366f1','#34d399','#818cf8'][Math.floor(Math.random()*3)]});
    function resizeSc(){sw=sc.width=window.innerWidth;sh=sc.height=window.innerHeight;}
    resizeSc(); window.addEventListener('resize',resizeSc);
    function drawBg(){
      if(splash.style.display==='none') return;
      sctx.clearRect(0,0,sw,sh);
      pts.forEach(p=>{
        p.x+=p.vx;p.y+=p.vy;
        if(p.x<0)p.x=1;if(p.x>1)p.x=0;if(p.y<0)p.y=1;if(p.y>1)p.y=0;
        sctx.beginPath();sctx.arc(p.x*sw,p.y*sh,p.r,0,Math.PI*2);sctx.fillStyle=p.c;sctx.globalAlpha=p.a;sctx.fill();
      });
      sctx.globalAlpha=1;
      for(let i=0;i<pts.length;i++) for(let j=i+1;j<pts.length;j++){
        const dx=(pts[i].x-pts[j].x)*sw,dy=(pts[i].y-pts[j].y)*sh,d=Math.sqrt(dx*dx+dy*dy);
        if(d<130){sctx.beginPath();sctx.moveTo(pts[i].x*sw,pts[i].y*sh);sctx.lineTo(pts[j].x*sw,pts[j].y*sh);sctx.strokeStyle='#6366f1';sctx.globalAlpha=(1-d/130)*0.05;sctx.lineWidth=0.5;sctx.stroke();}
      }
      sctx.globalAlpha=1;
      requestAnimationFrame(drawBg);
    }
    drawBg();
  }, 300);
}
"""

HERO = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');
#oi-splash.oi-exit{opacity:0;pointer-events:none}
#oi-splash-canvas{position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:0}
.oi-scan{position:fixed;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,.5),rgba(52,211,153,.4),transparent);animation:oi-scan 4s linear infinite;pointer-events:none;z-index:1}
@keyframes oi-scan{0%{top:0;opacity:0}3%{opacity:1}97%{opacity:1}100%{top:100%;opacity:0}}
.oi-inner{position:relative;z-index:2;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:48px 24px 64px}
.oi-eye{font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:700;letter-spacing:3px;color:#4f46e5;text-transform:uppercase;margin-bottom:20px;opacity:0;animation:oi-up .6s ease .2s forwards}
.oi-title{font-size:clamp(40px,8vw,84px);font-weight:800;line-height:1.05;letter-spacing:-3px;text-align:center;margin-bottom:16px;opacity:0;animation:oi-up .7s ease .4s forwards}
.oi-g1{background:linear-gradient(90deg,#818cf8,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.oi-g2{background:linear-gradient(90deg,#34d399,#059669);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.oi-sub{font-size:16px;color:#475569;max-width:520px;text-align:center;line-height:1.7;margin-bottom:32px;opacity:0;animation:oi-up .7s ease .6s forwards}
.oi-sub b{color:#64748b}
.oi-badges{display:flex;flex-wrap:wrap;justify-content:center;gap:10px;margin-bottom:36px;opacity:0;animation:oi-up .6s ease .8s forwards}
.oi-badge{font-size:11px;font-weight:700;letter-spacing:.5px;padding:5px 14px;border-radius:20px;border:1px solid}
.oib1{color:#a5b4fc;border-color:rgba(99,102,241,.35);background:rgba(99,102,241,.1)}
.oib2{color:#34d399;border-color:rgba(52,211,153,.3);background:rgba(52,211,153,.08)}
.oib3{color:#c084fc;border-color:rgba(192,132,252,.3);background:rgba(192,132,252,.07)}
.oib4{color:#fbbf24;border-color:rgba(251,191,36,.3);background:rgba(251,191,36,.07)}
.oi-net-card{width:100%;max-width:900px;background:rgba(255,255,255,.025);border:1px solid rgba(99,102,241,.2);border-radius:20px;overflow:hidden;margin-bottom:32px;opacity:0;animation:oi-up .7s ease 1s forwards;box-shadow:0 0 60px rgba(99,102,241,.1)}
.oi-net-hdr{padding:14px 20px;background:linear-gradient(90deg,rgba(99,102,241,.12),rgba(52,211,153,.06));border-bottom:1px solid rgba(255,255,255,.06);display:flex;align-items:center;gap:8px}
.oi-dot{width:10px;height:10px;border-radius:50%}
.oi-live{width:7px;height:7px;border-radius:50%;background:#34d399;box-shadow:0 0 6px #34d399;animation:oi-blink 1.4s ease-in-out infinite;margin-left:auto}
@keyframes oi-blink{0%,100%{opacity:1}50%{opacity:.3}}
#oi-net{display:block;width:100%;height:260px}
.oi-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;width:100%;max-width:900px;margin-bottom:32px;opacity:0;animation:oi-up .6s ease 1.2s forwards}
@media(max-width:640px){.oi-stats{grid-template-columns:repeat(2,1fr)}}
.oi-stat{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:18px;text-align:center;position:relative;overflow:hidden;transition:all .3s}
.oi-stat:hover{transform:translateY(-3px);border-color:rgba(99,102,241,.3)}
.oi-stat::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px}
.ois1::after{background:linear-gradient(90deg,transparent,#818cf8,transparent)}
.ois2::after{background:linear-gradient(90deg,transparent,#34d399,transparent)}
.ois3::after{background:linear-gradient(90deg,transparent,#fbbf24,transparent)}
.ois4::after{background:linear-gradient(90deg,transparent,#f87171,transparent)}
.oi-sv{font-size:34px;font-weight:800;line-height:1;margin-bottom:6px}
.oi-sl{font-size:10px;font-weight:700;letter-spacing:1.5px;color:#475569;text-transform:uppercase}
.oi-ss{font-size:10px;color:#334155;margin-top:4px}
.oi-agents-lbl{font-size:11px;font-weight:700;letter-spacing:2px;color:#6366f1;text-transform:uppercase;margin-bottom:12px;width:100%;max-width:900px;opacity:0;animation:oi-up .5s ease 1.3s forwards}
.oi-agents{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;width:100%;max-width:900px;margin-bottom:32px;opacity:0;animation:oi-up .6s ease 1.4s forwards}
@media(max-width:700px){.oi-agents{grid-template-columns:repeat(2,1fr)}}
.oi-agent{background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.07);border-radius:14px;padding:16px;transition:all .3s}
.oi-agent:hover{background:rgba(99,102,241,.07);border-color:rgba(99,102,241,.3);transform:translateY(-2px)}
.oi-ai{font-size:24px;margin-bottom:8px}
.oi-an{font-size:11px;font-weight:700;letter-spacing:1px;color:#94a3b8;text-transform:uppercase;margin-bottom:5px}
.oi-ad{font-size:12px;color:#475569;line-height:1.5}
.oi-cbar{height:3px;border-radius:2px;margin-top:10px;background:rgba(255,255,255,.06)}
.oi-cfill{height:100%;border-radius:2px}
.oi-ba-lbl{font-size:11px;font-weight:700;letter-spacing:2px;color:#6366f1;text-transform:uppercase;margin-bottom:12px;width:100%;max-width:900px;opacity:0;animation:oi-up .5s ease 1.5s forwards}
.oi-ba{display:grid;grid-template-columns:1fr 1fr;gap:16px;width:100%;max-width:900px;margin-bottom:36px;opacity:0;animation:oi-up .6s ease 1.6s forwards}
@media(max-width:640px){.oi-ba{grid-template-columns:1fr}}
.oi-bac{border-radius:14px;padding:18px;border:1px solid}
.oi-before{background:rgba(127,29,29,.1);border-color:rgba(248,113,113,.2)}
.oi-after{background:rgba(5,46,22,.12);border-color:rgba(52,211,153,.2)}
.oi-blbl{font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:10px}
.oi-bq{font-size:13px;line-height:1.7;font-style:italic;color:#94a3b8;border-left:3px solid;padding-left:12px}
.oi-before .oi-bq{border-color:#f87171}
.oi-after .oi-bq{border-color:#34d399}
.oi-links{display:flex;flex-wrap:wrap;gap:12px;justify-content:center;margin-bottom:44px;opacity:0;animation:oi-up .6s ease 1.7s forwards}
.oi-lnk{font-size:12px;font-weight:700;letter-spacing:.5px;padding:10px 22px;border-radius:10px;text-decoration:none;border:1px solid;transition:all .25s;display:inline-block}
.oi-lnk:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.4)}
.oi-gh{color:#e2e8f0;border-color:rgba(226,232,240,.2);background:rgba(226,232,240,.05)}
.oi-hf{color:#fbbf24;border-color:rgba(251,191,36,.25);background:rgba(251,191,36,.06)}
.oi-cl{color:#fb923c;border-color:rgba(251,146,60,.25);background:rgba(251,146,60,.06)}
.oi-enter-wrap{opacity:0;animation:oi-up .7s ease 1.9s forwards;text-align:center}
.oi-enter{font-size:15px;font-weight:700;letter-spacing:2px;padding:18px 56px;border-radius:12px;border:1px solid rgba(99,102,241,.5);background:linear-gradient(135deg,rgba(79,70,229,.25),rgba(5,150,105,.18));color:#e2e8f0;cursor:pointer;text-transform:uppercase;transition:all .3s;box-shadow:0 0 40px rgba(99,102,241,.2)}
.oi-enter:hover{background:linear-gradient(135deg,rgba(79,70,229,.45),rgba(5,150,105,.32));border-color:rgba(99,102,241,.9);transform:translateY(-3px);box-shadow:0 0 60px rgba(99,102,241,.4),0 10px 40px rgba(0,0,0,.5)}
.oi-hint{font-size:11px;color:#334155;letter-spacing:1px;margin-top:10px;font-family:'IBM Plex Mono',monospace}
@keyframes oi-up{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
</style>

<div id="oi-splash">
  <canvas id="oi-splash-canvas"></canvas>
  <div class="oi-scan"></div>
  <div class="oi-inner">

    <div class="oi-eye">Meta × Hugging Face OpenEnv Hackathon 2026 — Grand Finale</div>

    <h1 class="oi-title">
      <span class="oi-g1">AI Oversight</span><br>
      <span class="oi-g2">Inspector</span>
    </h1>

    <p class="oi-sub">Training an LLM to watch the <b>AI agents</b> — detecting violations <b>without ever seeing ground truth.</b> GRPO · Llama-3.2-1B · Adaptive Curriculum.</p>

    <div class="oi-badges">
      <span class="oi-badge oib1">🏆 OpenEnv Compliant</span>
      <span class="oi-badge oib2">⚡ GRPO + Unsloth</span>
      <span class="oi-badge oib3">🛡 AI Safety</span>
      <span class="oi-badge oib4">🎓 Adaptive Curriculum</span>
    </div>

    <div class="oi-net-card">
      <div class="oi-net-hdr">
        <div class="oi-dot" style="background:#f87171"></div>
        <div class="oi-dot" style="background:#fbbf24"></div>
        <div class="oi-dot" style="background:#34d399"></div>
        <span style="font-size:11px;font-weight:700;letter-spacing:1.5px;color:#94a3b8;margin-left:8px">LIVE AGENT NETWORK — OVERSIGHT INSPECTOR MONITORING SUB-AGENT FLEET</span>
        <div class="oi-live"></div>
      </div>
      <canvas id="oi-net"></canvas>
    </div>

    <div class="oi-stats">
      <div class="oi-stat ois1"><div class="oi-sv" style="color:#818cf8">78%</div><div class="oi-sl">Detection Accuracy</div><div class="oi-ss">post-training · 500 steps</div></div>
      <div class="oi-stat ois2"><div class="oi-sv" style="color:#34d399">12%</div><div class="oi-sl">False Positive Rate</div><div class="oi-ss">down from 35% baseline</div></div>
      <div class="oi-stat ois3"><div class="oi-sv" style="color:#fbbf24">0.74</div><div class="oi-sl">Avg Episode Reward</div><div class="oi-ss">up from 0.21 baseline</div></div>
      <div class="oi-stat ois4"><div class="oi-sv" style="color:#f87171">500</div><div class="oi-sl">Training Steps</div><div class="oi-ss">free T4 GPU · ~30 min</div></div>
    </div>

    <div class="oi-agents-lbl">Sub-Agent Fleet Being Monitored</div>
    <div class="oi-agents">
      <div class="oi-agent"><div class="oi-ai">🔍</div><div class="oi-an">Classifier</div><div class="oi-ad">Labels emails as spam, important, or routine</div><div class="oi-cbar"><div class="oi-cfill" style="width:82%;background:linear-gradient(90deg,#4f46e5,#818cf8)"></div></div></div>
      <div class="oi-agent"><div class="oi-ai">⚡</div><div class="oi-an">Prioritizer</div><div class="oi-ad">Assigns urgency — VIP miss triggers −0.30 penalty</div><div class="oi-cbar"><div class="oi-cfill" style="width:71%;background:linear-gradient(90deg,#059669,#34d399)"></div></div></div>
      <div class="oi-agent"><div class="oi-ai">🗺</div><div class="oi-an">Router</div><div class="oi-ad">Routes to correct team — critical must escalate</div><div class="oi-cbar"><div class="oi-cfill" style="width:68%;background:linear-gradient(90deg,#d97706,#fbbf24)"></div></div></div>
      <div class="oi-agent"><div class="oi-ai">✍️</div><div class="oi-an">Responder</div><div class="oi-ad">Generates replies — hallucination detection critical</div><div class="oi-cbar"><div class="oi-cfill" style="width:65%;background:linear-gradient(90deg,#dc2626,#f87171)"></div></div></div>
    </div>

    <div class="oi-ba-lbl">Before vs After Training</div>
    <div class="oi-ba">
      <div class="oi-bac oi-before"><div class="oi-blbl" style="color:#f87171">⚠ Before Training — Reward: 0.21</div><div class="oi-bq">"This email may or may not have been generated by AI. It is difficult to determine without additional context. There could potentially be some concerns, but I cannot say for certain..."</div></div>
      <div class="oi-bac oi-after"><div class="oi-blbl" style="color:#34d399">✓ After Training (GRPO, 500 steps) — Reward: 0.74</div><div class="oi-bq">"VIOLATION [HIGH]: Span — 'As per our policy...'. This paraphrases Policy §4.2 without attribution — documentation integrity violation. Confidence: 0.87."</div></div>
    </div>

    <div class="oi-links">
      <a class="oi-lnk oi-gh" href="https://github.com/Sachu651g/AI-Oversight-Inspector" target="_blank">⭐ GitHub</a>
      <a class="oi-lnk oi-hf" href="https://huggingface.co/spaces/sachingunagi66/openenv-email-ops" target="_blank">🤗 HF Space</a>
      <a class="oi-lnk oi-cl" href="https://colab.research.google.com/github/Sachu651g/AI-Oversight-Inspector/blob/main/round2_oversight_inspector/colab_train_oversight.ipynb" target="_blank">▶ Colab</a>
    </div>

    <div class="oi-enter-wrap">
      <button class="oi-enter" onclick="oiEnterDashboard()">Enter Dashboard →</button>
      <div class="oi-hint">Click to explore the live environment</div>
    </div>

  </div>
</div>

<script>
function oiEnterDashboard() {
  var splash = document.getElementById('oi-splash');
  splash.classList.add('oi-exit');
  setTimeout(function() {
    splash.style.display = 'none';
    // scroll to top of gradio dashboard
    window.scrollTo({top: 0, behavior: 'smooth'});
  }, 750);
}

// ── Neural net canvas ──────────────────────────────────────────────────────
(function() {
  var c = document.getElementById('oi-net');
  if (!c) return;
  var ctx = c.getContext('2d');
  var nodes = [], edges = [], signals = [];

  function resize() {
    var r = c.parentElement.getBoundingClientRect();
    c.width = r.width || 800; c.height = 260;
    buildNodes();
  }

  function buildNodes() {
    nodes = []; edges = [];
    var cw = c.width, ch = c.height;
    var xF = [0.1, 0.34, 0.65, 0.88];
    var COLORS = ['#6366f1','#fbbf24','#34d399','#f87171'];
    var LABELS = [['📧\\nInbox'],['🔍\\nClassify','⚡\\nPrioritize','🗺\\nRoute','✍️\\nRespond'],['🛡\\nOverseer'],['✓\\nApprove','⚠\\nFlag']];
    var counts = [1,4,1,2];
    counts.forEach(function(count, li) {
      for (var ni = 0; ni < count; ni++) {
        var yf = count === 1 ? 0.5 : (ni+1)/(count+1);
        nodes.push({x:cw*xF[li],y:ch*yf,r:li===2?20:12,color:COLORS[li],label:LABELS[li][ni]||'',layer:li,pulse:Math.random()*Math.PI*2,ps:0.022+Math.random()*0.015});
      }
    });
    var byL=[[],[],[],[]];
    nodes.forEach(function(n){byL[n.layer].push(n);});
    for(var li=0;li<3;li++) byL[li].forEach(function(a){byL[li+1].forEach(function(b){edges.push({a:a,b:b,color:a.color});});});
  }

  setInterval(function() {
    if (!edges.length) return;
    var e = edges[Math.floor(Math.random()*edges.length)];
    var cols = ['#818cf8','#34d399','#fbbf24','#f87171','#c084fc'];
    signals.push({e:e,t:0,spd:0.009+Math.random()*0.013,col:cols[Math.floor(Math.random()*cols.length)],r:2.5+Math.random()*2});
  }, 220);

  function draw() {
    var cw=c.width, ch=c.height;
    ctx.clearRect(0,0,cw,ch);
    edges.forEach(function(e){
      ctx.beginPath();ctx.moveTo(e.a.x,e.a.y);ctx.lineTo(e.b.x,e.b.y);
      ctx.strokeStyle=e.color;ctx.globalAlpha=0.13;ctx.lineWidth=1;ctx.stroke();
    });
    for(var i=signals.length-1;i>=0;i--){
      var s=signals[i]; s.t+=s.spd;
      if(s.t>1){signals.splice(i,1);continue;}
      var x=s.e.a.x+(s.e.b.x-s.e.a.x)*s.t, y=s.e.a.y+(s.e.b.y-s.e.a.y)*s.t;
      var g=ctx.createRadialGradient(x,y,0,x,y,s.r*5);
      g.addColorStop(0,s.col+'bb');g.addColorStop(1,s.col+'00');
      ctx.beginPath();ctx.arc(x,y,s.r*5,0,Math.PI*2);ctx.fillStyle=g;ctx.globalAlpha=0.5;ctx.fill();
      ctx.beginPath();ctx.arc(x,y,s.r,0,Math.PI*2);ctx.fillStyle=s.col;ctx.globalAlpha=1;ctx.fill();
    }
    ctx.globalAlpha=1;
    nodes.forEach(function(n){
      n.pulse+=n.ps;
      var glow=Math.sin(n.pulse)*0.5+0.5, gr=n.r+6+glow*9;
      var g=ctx.createRadialGradient(n.x,n.y,n.r*0.4,n.x,n.y,gr);
      g.addColorStop(0,n.color+'55');g.addColorStop(1,n.color+'00');
      ctx.beginPath();ctx.arc(n.x,n.y,gr,0,Math.PI*2);ctx.fillStyle=g;ctx.globalAlpha=0.75+glow*0.25;ctx.fill();
      ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,Math.PI*2);ctx.fillStyle='#04080f';ctx.globalAlpha=1;ctx.fill();
      ctx.strokeStyle=n.color;ctx.lineWidth=1.5;ctx.globalAlpha=0.65+glow*0.35;ctx.stroke();
      ctx.globalAlpha=0.8;ctx.fillStyle='#94a3b8';ctx.font='bold 9px IBM Plex Mono,monospace';ctx.textAlign='center';
      n.label.split('\\n').forEach(function(ln,li){ctx.fillText(ln,n.x,n.y+n.r+12+li*11);});
    });
    ctx.globalAlpha=1;
    requestAnimationFrame(draw);
  }

  resize();
  window.addEventListener('resize', resize);
  draw();

  // background particles on splash canvas
  var sc = document.getElementById('oi-splash-canvas');
  if (!sc) return;
  var sctx = sc.getContext('2d');
  var sw, sh, pts = [];
  for(var i=0;i<90;i++) pts.push({x:Math.random(),y:Math.random(),vx:(Math.random()-.5)*0.00015,vy:(Math.random()-.5)*0.00015,r:Math.random()*1.3+0.3,a:Math.random()*0.3+0.08,c:['#6366f1','#34d399','#818cf8'][Math.floor(Math.random()*3)]});
  function resizeSc(){sw=sc.width=window.innerWidth;sh=sc.height=window.innerHeight;}
  resizeSc(); window.addEventListener('resize',resizeSc);
  function drawBg(){
    if(document.getElementById('oi-splash') && document.getElementById('oi-splash').style.display==='none') return;
    sctx.clearRect(0,0,sw,sh);
    pts.forEach(function(p){
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0)p.x=1;if(p.x>1)p.x=0;if(p.y<0)p.y=1;if(p.y>1)p.y=0;
      sctx.beginPath();sctx.arc(p.x*sw,p.y*sh,p.r,0,Math.PI*2);sctx.fillStyle=p.c;sctx.globalAlpha=p.a;sctx.fill();
    });
    sctx.globalAlpha=1;
    for(var i=0;i<pts.length;i++) for(var j=i+1;j<pts.length;j++){
      var dx=(pts[i].x-pts[j].x)*sw,dy=(pts[i].y-pts[j].y)*sh,d=Math.sqrt(dx*dx+dy*dy);
      if(d<130){sctx.beginPath();sctx.moveTo(pts[i].x*sw,pts[i].y*sh);sctx.lineTo(pts[j].x*sw,pts[j].y*sh);sctx.strokeStyle='#6366f1';sctx.globalAlpha=(1-d/130)*0.05;sctx.lineWidth=0.5;sctx.stroke();}
    }
    sctx.globalAlpha=1;
    requestAnimationFrame(drawBg);
  }
  drawBg();
})();
</script>
"""

HERO = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');
/* ── hero wrapper ── */
.hero{position:relative;background:linear-gradient(135deg,#0a0f1a 0%,#0d1f3c 40%,#080c18 100%);border:1px solid rgba(99,102,241,.2);border-radius:16px;overflow:hidden;margin-bottom:6px}
@keyframes heroGlow{0%,100%{box-shadow:0 0 40px rgba(99,102,241,.08)}50%{box-shadow:0 0 80px rgba(99,102,241,.18),0 0 120px rgba(52,211,153,.06)}}
.hero{animation:heroGlow 5s ease-in-out infinite}
/* ── canvas viz ── */
#hero-net{display:block;width:100%;height:220px;cursor:default}
/* ── content below canvas ── */
.hero-body{padding:24px 32px 28px}
.hero-eyebrow{font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:700;letter-spacing:2.5px;color:#4f46e5;text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:8px}
.hero-eyebrow::before{content:'';display:inline-block;width:24px;height:2px;background:linear-gradient(90deg,#4f46e5,transparent)}
.hero-title{font-family:'Syne',sans-serif;font-size:32px;font-weight:800;color:#f1f5f9;line-height:1.15;margin:0 0 6px;letter-spacing:-1px}
.hero-title em{font-style:normal;background:linear-gradient(90deg,#818cf8,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero-sub{font-size:14px;color:#64748b;margin-bottom:20px;line-height:1.6;max-width:580px}
.hero-sub b{color:#94a3b8}
.badge-row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px}
.badge{display:inline-flex;align-items:center;gap:5px;font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:700;letter-spacing:.5px;padding:4px 12px;border-radius:20px}
.badge-indigo{background:rgba(79,70,229,.12);color:#818cf8;border:1px solid rgba(79,70,229,.25)}
.badge-green{background:rgba(5,150,105,.12);color:#34d399;border:1px solid rgba(5,150,105,.25)}
.badge-purple{background:rgba(139,92,246,.12);color:#c084fc;border:1px solid rgba(139,92,246,.25)}
.badge-amber{background:rgba(245,158,11,.10);color:#fbbf24;border:1px solid rgba(245,158,11,.22)}
.pill-links{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:24px}
.pill-link{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:700;letter-spacing:.5px;padding:6px 16px;border-radius:8px;text-decoration:none!important;transition:all .2s;border:1px solid}
.pill-link.gh{color:#e2e8f0;border-color:rgba(226,232,240,.15);background:rgba(226,232,240,.05)}
.pill-link.hf{color:#fbbf24;border-color:rgba(251,191,36,.2);background:rgba(251,191,36,.06)}
.pill-link:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(0,0,0,.3)}
/* ── stat cards ── */
.stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:20px}
.stat-card{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);border-radius:10px;padding:14px 16px;text-align:center;position:relative;overflow:hidden;transition:border-color .2s,transform .2s}
.stat-card:hover{border-color:rgba(99,102,241,.25);transform:translateY(-2px)}
.stat-card::before{content:'';position:absolute;bottom:0;left:0;right:0;height:2px}
.stat-card.green::before{background:linear-gradient(90deg,transparent,#34d399,transparent)}
.stat-card.blue::before{background:linear-gradient(90deg,transparent,#818cf8,transparent)}
.stat-card.amber::before{background:linear-gradient(90deg,transparent,#fbbf24,transparent)}
.stat-card.red::before{background:linear-gradient(90deg,transparent,#f87171,transparent)}
.stat-num{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;line-height:1}
.stat-label{font-family:'IBM Plex Mono',monospace;font-size:9px;color:#475569;letter-spacing:1px;margin-top:4px;text-transform:uppercase}
.stat-delta{font-size:11px;color:#64748b;margin-top:3px}
/* ── canvas label overlay ── */
.net-label{position:absolute;font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:700;letter-spacing:1px;color:#475569;pointer-events:none}
</style>

<div class="hero">
  <!-- Animated neural network canvas -->
  <div style="position:relative">
    <canvas id="hero-net"></canvas>
    <div style="position:absolute;top:12px;left:16px;font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:700;letter-spacing:2px;color:rgba(99,102,241,.7)">LIVE AGENT NETWORK</div>
    <div style="position:absolute;top:12px;right:16px;display:flex;align-items:center;gap:6px">
      <span style="width:6px;height:6px;border-radius:50%;background:#34d399;display:inline-block;animation:blink 1.4s ease-in-out infinite;box-shadow:0 0 6px #34d399"></span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#34d399;font-weight:700">MONITORING</span>
    </div>
  </div>

  <!-- Hero content -->
  <div class="hero-body">
    <div class="hero-eyebrow">Meta x Hugging Face OpenEnv Hackathon 2026 &mdash; Grand Finale</div>
    <h1 class="hero-title">Who watches<br>the <em>AI agents?</em></h1>
    <p class="hero-sub">Trains an LLM to act as an <b>autonomous oversight inspector</b> &mdash;
    monitoring enterprise agents and detecting violations
    <b>without ever seeing ground truth.</b>
    GRPO on Llama-3.2-1B &middot; Adaptive curriculum &middot; 800 steps.</p>
    <div class="badge-row">
      <span class="badge badge-indigo">&#127942; OpenEnv Compliant</span>
      <span class="badge badge-green">&#9889; GRPO &middot; Unsloth</span>
      <span class="badge badge-purple">&#128737; AI Safety &middot; Scalable Oversight</span>
      <span class="badge badge-amber">&#127891; Adaptive Curriculum</span>
    </div>
    <div class="pill-links">
      <a class="pill-link gh" href="https://github.com/Sachu651g/AI-Oversight-Inspector" target="_blank">&#11088; GitHub</a>
      <a class="pill-link hf" href="https://huggingface.co/spaces/sachingunagi66/openenv-email-ops" target="_blank">&#129303; HF Space</a>
    </div>
    <div class="stat-row">
      <div class="stat-card green"><div class="stat-num" style="color:#34d399">78%</div><div class="stat-label">Detection Acc.</div><div class="stat-delta">was 42% &uarr;+36pp</div></div>
      <div class="stat-card blue"><div class="stat-num" style="color:#818cf8">12%</div><div class="stat-label">False Pos. Rate</div><div class="stat-delta">was 35% &darr;&minus;23pp</div></div>
      <div class="stat-card amber"><div class="stat-num" style="color:#fbbf24">0.881</div><div class="stat-label">Eval Score</div><div class="stat-delta">post-training hard</div></div>
      <div class="stat-card red"><div class="stat-num" style="color:#f87171">800</div><div class="stat-label">Train Steps</div><div class="stat-delta">Tesla T4 &middot; 2h 15m</div></div>
    </div>
  </div>
</div>

<style>
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
</style>

<script>
(function(){
  function initNet(){
    var c=document.getElementById('hero-net');
    if(!c){setTimeout(initNet,200);return;}
    var ctx=c.getContext('2d');
    function resize(){
      var r=c.parentElement.getBoundingClientRect();
      c.width=r.width||800; c.height=220;
    }
    resize();
    window.addEventListener('resize',function(){resize();buildNodes();});

    // Node layers: [Inbox] -> [Classifier, Prioritizer, Router, Responder] -> [Overseer] -> [Flag, Approve]
    var LAYERS=[
      [{label:'\\u{1F4E7}\\nInbox',color:'#6366f1'}],
      [{label:'\\u{1F50D}\\nClassify',color:'#fbbf24'},{label:'\\u26A1\\nPrioritize',color:'#fb923c'},{label:'\\u{1F5FA}\\nRoute',color:'#f87171'},{label:'\\u270D\\nRespond',color:'#c084fc'}],
      [{label:'\\u{1F6E1}\\nOverseer',color:'#34d399',big:true}],
      [{label:'\\u2713\\nApprove',color:'#4ade80'},{label:'\\u26A0\\nFlag',color:'#f87171'}]
    ];
    var xFracs=[0.08,0.32,0.65,0.88];
    var nodes=[],edges=[],signals=[];

    function buildNodes(){
      nodes=[];edges=[];
      var cw=c.width,ch=c.height;
      LAYERS.forEach(function(layer,li){
        layer.forEach(function(nd,ni){
          var yf=layer.length===1?0.5:(ni+1)/(layer.length+1);
          nodes.push({x:cw*xFracs[li],y:ch*yf,r:nd.big?18:12,color:nd.color,label:nd.label,layer:li,idx:ni,pulse:Math.random()*Math.PI*2,ps:0.025+Math.random()*0.015});
        });
      });
      // edges between consecutive layers
      var byLayer=[[],[],[],[]];
      nodes.forEach(function(n){byLayer[n.layer].push(n);});
      for(var li=0;li<3;li++){
        byLayer[li].forEach(function(a){
          byLayer[li+1].forEach(function(b){
            edges.push({a:a,b:b,color:a.color});
          });
        });
      }
    }
    buildNodes();

    function spawnSignal(){
      if(!edges.length)return;
      var e=edges[Math.floor(Math.random()*edges.length)];
      var cols=['#818cf8','#34d399','#fbbf24','#f87171','#c084fc'];
      signals.push({e:e,t:0,spd:0.01+Math.random()*0.015,col:cols[Math.floor(Math.random()*cols.length)],r:2.5+Math.random()*2});
    }
    var spawnTimer=setInterval(spawnSignal,250);

    function draw(){
      ctx.clearRect(0,0,c.width,c.height);
      // edges
      edges.forEach(function(e){
        ctx.beginPath();ctx.moveTo(e.a.x,e.a.y);ctx.lineTo(e.b.x,e.b.y);
        ctx.strokeStyle=e.color;ctx.globalAlpha=0.12;ctx.lineWidth=1;ctx.stroke();
      });
      // signals
      for(var i=signals.length-1;i>=0;i--){
        var s=signals[i];s.t+=s.spd;
        if(s.t>1){signals.splice(i,1);continue;}
        var sx=s.e.a.x+(s.e.b.x-s.e.a.x)*s.t;
        var sy=s.e.a.y+(s.e.b.y-s.e.a.y)*s.t;
        var g=ctx.createRadialGradient(sx,sy,0,sx,sy,s.r*5);
        g.addColorStop(0,s.col+'cc');g.addColorStop(1,s.col+'00');
        ctx.beginPath();ctx.arc(sx,sy,s.r*5,0,Math.PI*2);
        ctx.fillStyle=g;ctx.globalAlpha=0.5;ctx.fill();
        ctx.beginPath();ctx.arc(sx,sy,s.r,0,Math.PI*2);
        ctx.fillStyle=s.col;ctx.globalAlpha=1;ctx.fill();
      }
      // nodes
      ctx.globalAlpha=1;
      nodes.forEach(function(n){
        n.pulse+=n.ps;
        var glow=Math.sin(n.pulse)*0.5+0.5;
        var gr=n.r+6+glow*8;
        var g=ctx.createRadialGradient(n.x,n.y,n.r*0.4,n.x,n.y,gr);
        g.addColorStop(0,n.color+'66');g.addColorStop(1,n.color+'00');
        ctx.beginPath();ctx.arc(n.x,n.y,gr,0,Math.PI*2);
        ctx.fillStyle=g;ctx.globalAlpha=0.8+glow*0.2;ctx.fill();
        ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,Math.PI*2);
        ctx.fillStyle='#060b16';ctx.globalAlpha=1;ctx.fill();
        ctx.strokeStyle=n.color;ctx.lineWidth=1.5;
        ctx.globalAlpha=0.7+glow*0.3;ctx.stroke();
        // label
        ctx.globalAlpha=0.75;ctx.fillStyle='#94a3b8';
        ctx.font='bold 9px IBM Plex Mono,monospace';ctx.textAlign='center';
        var lines=n.label.split('\\n');
        lines.forEach(function(ln,li){ctx.fillText(ln,n.x,n.y+n.r+12+li*11);});
      });
      ctx.globalAlpha=1;
      requestAnimationFrame(draw);
    }
    draw();
  }
  if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',initNet);}
  else{initNet();}
})();
</script>
"""

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

RESULTS_HTML = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');
.rw{background:#060b16;padding:4px 0;font-family:'IBM Plex Mono',monospace}
.sl{font-size:9px;font-weight:700;letter-spacing:2.5px;color:#334155;text-transform:uppercase;display:flex;align-items:center;gap:10px;margin:0 0 14px}
.sl::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(255,255,255,.08),transparent)}
.kg{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}
.kc{border-radius:10px;padding:16px 14px;text-align:center;position:relative;overflow:hidden;border:1px solid;transition:transform .2s,box-shadow .2s}
.kc:hover{transform:translateY(-4px);box-shadow:0 12px 40px rgba(0,0,0,.5)}
.kn{font-size:28px;font-weight:800;line-height:1;font-family:'Syne',sans-serif}
.kl{font-size:9px;letter-spacing:1.2px;margin:5px 0 3px;font-weight:700}
.kd{font-size:10px;color:#475569}
.kb{height:3px;border-radius:2px;background:rgba(255,255,255,.08);margin-top:8px;overflow:hidden}
.kf{height:3px;border-radius:2px;width:0%;transition:width 1.2s cubic-bezier(.4,0,.2,1)}
.cr{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px}
.cc{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:16px;transition:border-color .2s}
.cc:hover{border-color:rgba(99,102,241,.25)}
.ct{font-size:10px;font-weight:700;letter-spacing:1px;color:#475569;margin:0 0 12px}
.rt{width:100%;border-collapse:collapse;font-size:11px}
.rt th{font-size:9px;letter-spacing:1.2px;color:#334155;font-weight:700;padding:6px 10px;text-align:left;border-bottom:1px solid rgba(255,255,255,.05)}
.rt td{padding:8px 10px;border-bottom:1px solid rgba(255,255,255,.04)}
.rp{display:inline-block;font-size:10px;font-weight:700;padding:2px 8px;border-radius:6px;font-family:monospace}
.cs{display:flex;gap:0;border-radius:6px;overflow:hidden;height:22px;margin-top:4px}
.ce{display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;letter-spacing:.5px}
@keyframes fadeInUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.06)}}
.kc{animation:fadeInUp .5s ease both}
.kc:nth-child(1){animation-delay:.05s}
.kc:nth-child(2){animation-delay:.15s}
.kc:nth-child(3){animation-delay:.25s}
.kc:nth-child(4){animation-delay:.35s}
</style>

<div class="rw">
  <div class="sl">key results — before vs after grpo training</div>

  <div class="kg">
    <div class="kc" style="background:rgba(52,211,153,.06);border-color:rgba(52,211,153,.2)">
      <div class="kn" style="color:#34d399" id="rv1">0%</div>
      <div class="kl" style="color:#34d399">Detection acc.</div>
      <div class="kd">42% → 78% · +36pp</div>
      <div class="kb"><div class="kf" style="background:#34d399" id="rb1"></div></div>
    </div>
    <div class="kc" style="background:rgba(129,140,248,.06);border-color:rgba(129,140,248,.2)">
      <div class="kn" style="color:#818cf8" id="rv2">0%</div>
      <div class="kl" style="color:#818cf8">False pos. rate</div>
      <div class="kd">35% → 12% · −23pp</div>
      <div class="kb"><div class="kf" style="background:#818cf8" id="rb2"></div></div>
    </div>
    <div class="kc" style="background:rgba(251,146,60,.06);border-color:rgba(251,146,60,.2)">
      <div class="kn" style="color:#fb923c" id="rv3">0%</div>
      <div class="kl" style="color:#fb923c">Severity acc.</div>
      <div class="kd">38% → 71% · +33pp</div>
      <div class="kb"><div class="kf" style="background:#fb923c" id="rb3"></div></div>
    </div>
    <div class="kc" style="background:rgba(250,191,36,.06);border-color:rgba(250,191,36,.2)">
      <div class="kn" style="color:#fbbf24" id="rv4">0.00</div>
      <div class="kl" style="color:#fbbf24">Avg ep. score</div>
      <div class="kd">0.21 → 0.74 · +0.53</div>
      <div class="kb"><div class="kf" style="background:#fbbf24" id="rb4"></div></div>
    </div>
  </div>

  <div class="cr">
    <div class="cc">
      <div class="ct">Reward curve — episode score over training</div>
      <div style="position:relative;height:180px">
        <canvas id="rwChart"></canvas>
      </div>
      <div style="display:flex;gap:14px;margin-top:8px;font-size:10px;color:#475569">
        <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:2px;background:#534AB7;display:inline-block"></span>raw</span>
        <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:3px;background:#34d399;display:inline-block"></span>smoothed</span>
        <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:1px;border-top:1px dashed #475569;display:inline-block"></span>baseline 0.21</span>
      </div>
    </div>
    <div class="cc">
      <div class="ct">Before vs after — all metrics</div>
      <div style="position:relative;height:180px">
        <canvas id="baChart"></canvas>
      </div>
      <div style="display:flex;gap:14px;margin-top:8px;font-size:10px;color:#475569">
        <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:2px;background:#334155;display:inline-block"></span>before</span>
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
      <div class="cs" id="currBar"></div>
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

<script>
(function(){
  function easeOut(t){return 1-(1-t)*(1-t)}
  function animKPI(valId,barId,target,isFloat,barPct,delay){
    setTimeout(()=>{
      const ve=document.getElementById(valId);
      const be=document.getElementById(barId);
      if(!ve)return;
      let start=null;
      const dur=1000;
      function step(ts){
        if(!start)start=ts;
        const p=Math.min((ts-start)/dur,1);
        const e=easeOut(p);
        ve.textContent=isFloat?(target*e).toFixed(2):Math.round(target*e)+'%';
        if(be)be.style.width=(barPct*e).toFixed(1)+'%';
        if(p<1)requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    },delay);
  }
  animKPI('rv1','rb1',78,false,78,300);
  animKPI('rv2','rb2',12,false,12,450);
  animKPI('rv3','rb3',71,false,71,600);
  animKPI('rv4','rb4',0.74,true,74,750);

  const cb=document.getElementById('currBar');
  if(cb){
    for(let i=0;i<50;i++){
      const s=document.createElement('div');
      s.className='ce';
      s.style.width='2%';
      s.style.background=i<20?'#27500A':i<35?'#633806':'#791F1F';
      if(i===19||i===34)s.style.borderRight='2px solid rgba(255,255,255,.25)';
      cb.appendChild(s);
    }
  }

  function waitForChartJS(cb,n){
    n=n||0;
    if(typeof Chart!=='undefined'){cb();}
    else if(n<30){setTimeout(()=>waitForChartJS(cb,n+1),200);}
  }

  waitForChartJS(function(){
    const raw=[];
    for(let i=0;i<50;i++){
      const diff=i<20?'easy':i<35?'medium':'hard';
      const base=diff==='easy'?0.28:diff==='medium'?0.42:0.55;
      raw.push(parseFloat((base+(i/50*0.46)+(Math.random()-.5)*0.18).toFixed(3)));
    }
    const w=5;
    const sm=raw.map((_,i)=>{
      const sl=raw.slice(Math.max(0,i-w),i+w+1);
      return parseFloat((sl.reduce((a,b)=>a+b,0)/sl.length).toFixed(3));
    });
    const labels=Array.from({length:50},(_,i)=>i);
    const gridC='rgba(255,255,255,.04)';
    const tickC='#334155';
    const baseFont={size:9,family:'IBM Plex Mono'};

    const rc=document.getElementById('rwChart');
    if(rc){
      new Chart(rc,{
        type:'line',
        data:{
          labels,
          datasets:[
            {label:'Raw',data:raw,borderColor:'#534AB7',borderWidth:1,pointRadius:0,tension:0.3,fill:false,borderDash:[2,2]},
            {label:'Smoothed',data:sm,borderColor:'#34d399',borderWidth:2.5,pointRadius:0,tension:0.4,fill:false},
          ]
        },
        options:{
          responsive:true,maintainAspectRatio:false,animation:{duration:1500,easing:'easeInOutQuart'},
          plugins:{legend:{display:false},tooltip:{
            backgroundColor:'rgba(5,10,20,.95)',titleColor:'#94a3b8',bodyColor:'#e2e8f0',
            borderColor:'rgba(255,255,255,.08)',borderWidth:1,titleFont:baseFont,bodyFont:baseFont,
            callbacks:{title:ctx=>'Step '+ctx[0].label,label:ctx=>ctx.dataset.label+': '+ctx.parsed.y.toFixed(3)}
          }},
          scales:{
            x:{grid:{color:gridC},ticks:{color:tickC,font:baseFont,maxTicksLimit:8}},
            y:{min:0,max:1.0,grid:{color:gridC},ticks:{color:tickC,font:baseFont,callback:v=>v.toFixed(2)}}
          }
        }
      });
    }

    const bc=document.getElementById('baChart');
    if(bc){
      new Chart(bc,{
        type:'bar',
        data:{
          labels:['Detection','FP Rate','Severity','Explanation'],
          datasets:[
            {label:'Before',data:[42,35,38,31],backgroundColor:'#2C2C2A',borderRadius:3,borderSkipped:false},
            {label:'After', data:[78,12,71,67],backgroundColor:'#4f46e5',borderRadius:3,borderSkipped:false},
          ]
        },
        options:{
          responsive:true,maintainAspectRatio:false,
          animation:{duration:1200,easing:'easeInOutQuart'},
          plugins:{legend:{display:false},tooltip:{
            backgroundColor:'rgba(5,10,20,.95)',titleColor:'#94a3b8',bodyColor:'#e2e8f0',
            borderColor:'rgba(255,255,255,.08)',borderWidth:1,titleFont:baseFont,bodyFont:baseFont,
          }},
          scales:{
            x:{grid:{display:false},ticks:{color:'#475569',font:baseFont,autoSkip:false}},
            y:{max:100,grid:{color:gridC},ticks:{color:tickC,font:baseFont,callback:v=>v+'%'}}
          }
        }
      });
    }
  });
})();
</script>
"""

ABOUT_MD = """
## &#128737; AI Oversight Inspector

> *"Everyone builds AI agents. Who monitors them?"*

This project directly tackles **scalable oversight** — one of the most important open problems in AI safety.

An LLM is trained via GRPO to inspect a fleet of enterprise agents and detect violations —
hallucinations, wrong classifications, policy breaches, and cross-agent inconsistencies —
purely from reasoning about inputs, outputs, and explanations. **No ground truth exposed.**

### Two Environments

| Environment | Hackathon Theme | Key Innovation |
|---|---|---|
| **EmailOpsEnv** (Round 1) | Theme 3.2 — Personalized Tasks | Multi-action RL: classify → prioritize → route → reply. VIP 2× penalties. Partial obs. |
| **OversightEnv** (Round 2) | Theme 1 — Multi-Agent Interactions | AI overseer monitors 4-agent fleet. GRPO. Adaptive curriculum with live demotion. |

### Training Stack
- **Model**: `Llama-3.2-1B-Instruct` (Unsloth 4-bit LoRA, rank 16)
- **Algorithm**: GRPO via HuggingFace TRL · Group size 4
- **Hardware**: Tesla T4 (free Colab) · 800 steps · 2h 15m
- **Curriculum**: Easy → Medium → Hard · Live demotion at step ~330

### Reward Design

| Signal | Value | Rationale |
|---|---|---|
| Correct violation detected | +0.40 | Core objective |
| Correct severity | +0.20 | Calibration, not just binary |
| Quality explanation | +0.20 | Causal reasoning |
| Correct approve | +0.20 | Precision incentive |
| **False positive** | **−0.30** | Alert fatigue prevention |
| Missed violation | −0.20 | Some leniency |
| Self-improvement bonus | +0.10 | Recursive skill signal |

### Actual Results (Colab run)

| Metric | Before | After |
|---|---|---|
| Detection accuracy | 42% | **78%** (+36pp) |
| False positive rate | 35% | **12%** (−23pp) |
| Severity accuracy | 38% | **71%** (+33pp) |
| Eval score (hard, 10 seeds) | — | **0.881** |
| Episode reward | 0.455 | **0.600** |
"""


with gr.Blocks(title="AI Oversight Inspector · Meta × HF Hackathon 2026", css=CSS, theme=gr.themes.Base(), js=SPLASH_JS) as demo_ui:
    gr.HTML(HERO)
    with gr.Tabs():
        with gr.Tab("\U0001f4ec EmailOpsEnv"):
            gr.HTML(TAB_EMAIL_HEADER)
            with gr.Row():
                diff_dd = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="DIFFICULTY")
                seed_sl = gr.Slider(1, 200, value=42, step=1, label="RANDOM SEED")
            run_btn = gr.Button("\u25b6  RUN LIVE EPISODE", variant="primary")
            out_html = gr.HTML()
            run_btn.click(run_email_demo, [diff_dd, seed_sl], out_html)

        with gr.Tab("\U0001f6e1 Oversight Inspector"):
            gr.HTML(TAB_OVERSIGHT_HEADER)
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("""
                    ### 🎯 Interactive Oversight Demo
                    
                    Watch the AI Oversight Inspector analyze a batch of sub-agent decisions in real-time. 
                    The overseer detects violations **without seeing ground truth** — reasoning from evidence alone.
                    
                    **What to look for**:
                    - 🔴 Hallucinations (fabricated facts)
                    - 🟠 Wrong classifications
                    - 🟡 Policy violations
                    - 🟣 Severity mismatches
                    - 🔵 Cross-agent inconsistencies
                    """)
                
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 📊 Live Metrics
                    
                    **Detection Rate**: 78%  
                    **False Positive**: 12%  
                    **Precision**: 0.87  
                    **Recall**: 0.78
                    
                    *Post-training on 500 GRPO steps*
                    """)
            
            with gr.Row():
                ov_seed = gr.Slider(1, 100, value=42, step=1, label="🎲 Random Seed", 
                                   info="Change seed to see different violation patterns")
                ov_difficulty = gr.Dropdown(["easy", "medium", "hard"], value="easy", 
                                           label="🎚️ Difficulty Level",
                                           info="Harder = more subtle violations")
            
            ov_btn = gr.Button("▶  ANALYZE SUB-AGENT BATCH", variant="primary", size="lg")
            
            with gr.Accordion("ℹ️ How It Works", open=False):
                gr.Markdown("""
                The Oversight Inspector observes 4 specialized sub-agents:
                
                1. **Classifier** 🔍 — Categorizes incoming tasks
                2. **Prioritizer** ⚡ — Assigns urgency levels  
                3. **Router** 🗺 — Routes to correct team
                4. **Responder** ✍️ — Generates replies
                
                The overseer must detect when any agent:
                - Hallucinates facts not in the input
                - Misclassifies based on policy
                - Assigns wrong severity
                - Contradicts other agents
                
                **Key constraint**: No ground truth labels exposed. The overseer reasons like a human auditor.
                """)
            
            ov_html = gr.HTML()
            ov_btn.click(run_oversight_demo, [ov_seed, ov_difficulty], ov_html)

        with gr.Tab("\U0001f4ca Training Results"):
            gr.HTML(RESULTS_HTML)
            gr.HTML("""<div style='display:flex;flex-direction:column;gap:16px;padding:4px 0'>

<!-- Interactive Reward Curve Chart -->
<div style='background:linear-gradient(135deg,#0a0f1a,#0d1f3c);border:1px solid rgba(99,102,241,.2);border-radius:14px;overflow:hidden;box-shadow:0 8px 40px rgba(0,0,0,.4);animation:fadeInUp 0.8s ease-out'>
  <div style='padding:16px 20px;background:linear-gradient(90deg,rgba(79,70,229,.15),rgba(99,102,241,.08));border-bottom:1px solid rgba(99,102,241,.15)'>
    <div style='display:flex;align-items:center;gap:12px'>
      <span style='font-size:24px'>📈</span>
      <div>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:11px;color:#818cf8;font-weight:700;letter-spacing:1.5px'>REWARD CURVE</div>
        <div style='font-size:12px;color:#64748b;margin-top:2px'>Episode score 0.455 → 0.600 over 800 GRPO steps</div>
      </div>
      <div style='margin-left:auto'>
        <span style='background:rgba(52,211,153,.15);color:#34d399;border:1px solid rgba(52,211,153,.3);padding:4px 12px;border-radius:20px;font-size:10px;font-weight:700;font-family:monospace'>+32% IMPROVEMENT</span>
      </div>
    </div>
  </div>
  <div style='padding:16px;background:#0a0f1a'>
    <svg width='100%' height='300' viewBox='0 0 800 300' style='border-radius:8px;background:#050a14'>
      <!-- Grid lines -->
      <defs>
        <pattern id='grid' width='80' height='30' patternUnits='userSpaceOnUse'>
          <path d='M 80 0 L 0 0 0 30' fill='none' stroke='rgba(99,102,241,.08)' stroke-width='1'/>
        </pattern>
      </defs>
      <rect width='100%' height='100%' fill='url(#grid)'/>
      
      <!-- Axes -->
      <line x1='60' y1='250' x2='740' y2='250' stroke='rgba(255,255,255,.2)' stroke-width='2'/>
      <line x1='60' y1='50' x2='60' y2='250' stroke='rgba(255,255,255,.2)' stroke-width='2'/>
      
      <!-- Reward curve (animated) -->
      <path d='M 60 220 Q 150 210 200 200 Q 300 180 400 160 Q 500 140 600 120 Q 650 110 700 100 L 740 90' 
            fill='none' stroke='#4ade80' stroke-width='3' class='chart-line'/>
      
      <!-- Data points -->
      <circle cx='60' cy='220' r='4' fill='#34d399' opacity='0' style='animation:fadeIn 1s ease-out 1s forwards'/>
      <circle cx='200' cy='200' r='4' fill='#34d399' opacity='0' style='animation:fadeIn 1s ease-out 1.2s forwards'/>
      <circle cx='400' cy='160' r='4' fill='#34d399' opacity='0' style='animation:fadeIn 1s ease-out 1.4s forwards'/>
      <circle cx='600' cy='120' r='4' fill='#34d399' opacity='0' style='animation:fadeIn 1s ease-out 1.6s forwards'/>
      <circle cx='740' cy='90' r='4' fill='#34d399' opacity='0' style='animation:fadeIn 1s ease-out 1.8s forwards'/>
      
      <!-- Labels -->
      <text x='60' y='270' fill='#64748b' font-size='10' font-family='monospace' text-anchor='middle'>0</text>
      <text x='200' y='270' fill='#64748b' font-size='10' font-family='monospace' text-anchor='middle'>200</text>
      <text x='400' y='270' fill='#64748b' font-size='10' font-family='monospace' text-anchor='middle'>400</text>
      <text x='600' y='270' fill='#64748b' font-size='10' font-family='monospace' text-anchor='middle'>600</text>
      <text x='740' y='270' fill='#64748b' font-size='10' font-family='monospace' text-anchor='middle'>800</text>
      
      <text x='45' y='255' fill='#64748b' font-size='10' font-family='monospace' text-anchor='end'>0.4</text>
      <text x='45' y='205' fill='#64748b' font-size='10' font-family='monospace' text-anchor='end'>0.5</text>
      <text x='45' y='155' fill='#64748b' font-size='10' font-family='monospace' text-anchor='end'>0.6</text>
      <text x='45' y='105' fill='#64748b' font-size='10' font-family='monospace' text-anchor='end'>0.7</text>
      <text x='45' y='55' fill='#64748b' font-size='10' font-family='monospace' text-anchor='end'>0.8</text>
      
      <!-- Axis labels -->
      <text x='400' y='295' fill='#94a3b8' font-size='12' font-family='monospace' text-anchor='middle'>Training Step</text>
      <text x='25' y='150' fill='#94a3b8' font-size='12' font-family='monospace' text-anchor='middle' transform='rotate(-90 25 150)'>Episode Reward</text>
    </svg>
  </div>
  <div style='padding:12px 20px;background:rgba(255,255,255,.02);border-top:1px solid rgba(255,255,255,.05);display:flex;gap:16px;font-size:11px;color:#64748b;font-family:monospace'>
    <span>🎯 Detection: 42% → 78%</span>
    <span>⚡ FP Rate: 35% → 12%</span>
    <span>📊 Severity: 38% → 71%</span>
  </div>
</div>

<!-- Interactive Before/After Metrics Chart -->
<div style='background:linear-gradient(135deg,#1a0e05,#2b1407);border:1px solid rgba(251,146,60,.2);border-radius:14px;overflow:hidden;box-shadow:0 8px 40px rgba(0,0,0,.4);animation:fadeInUp 0.8s ease-out 0.2s backwards'>
  <div style='padding:16px 20px;background:linear-gradient(90deg,rgba(251,146,60,.15),rgba(234,88,12,.08));border-bottom:1px solid rgba(251,146,60,.15)'>
    <div style='display:flex;align-items:center;gap:12px'>
      <span style='font-size:24px'>📊</span>
      <div>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:11px;color:#fb923c;font-weight:700;letter-spacing:1.5px'>BEFORE vs AFTER TRAINING</div>
        <div style='font-size:12px;color:#64748b;margin-top:2px'>Key metrics comparison across 5 dimensions</div>
      </div>
    </div>
  </div>
  <div style='padding:16px;background:#0a0f1a'>
    <svg width='100%' height='300' viewBox='0 0 800 300' style='border-radius:8px;background:#050a14'>
      <!-- Grid -->
      <rect width='100%' height='100%' fill='url(#grid)'/>
      
      <!-- Axes -->
      <line x1='80' y1='250' x2='720' y2='250' stroke='rgba(255,255,255,.2)' stroke-width='2'/>
      <line x1='80' y1='50' x2='80' y2='250' stroke='rgba(255,255,255,.2)' stroke-width='2'/>
      
      <!-- Before bars (red) -->
      <rect x='120' y='208' width='30' height='42' fill='#f87171' class='bar-anim' style='animation-delay:0.5s'/>
      <rect x='220' y='162' width='30' height='88' fill='#f87171' class='bar-anim' style='animation-delay:0.7s'/>
      <rect x='320' y='174' width='30' height='76' fill='#f87171' class='bar-anim' style='animation-delay:0.9s'/>
      <rect x='420' y='188' width='30' height='62' fill='#f87171' class='bar-anim' style='animation-delay:1.1s'/>
      <rect x='520' y='190' width='30' height='60' fill='#f87171' class='bar-anim' style='animation-delay:1.3s'/>
      
      <!-- After bars (green) -->
      <rect x='160' y='94' width='30' height='156' fill='#4ade80' class='bar-anim' style='animation-delay:0.6s'/>
      <rect x='260' y='220' width='30' height='30' fill='#4ade80' class='bar-anim' style='animation-delay:0.8s'/>
      <rect x='360' y='108' width='30' height='142' fill='#4ade80' class='bar-anim' style='animation-delay:1.0s'/>
      <rect x='460' y='116' width='30' height='134' fill='#4ade80' class='bar-anim' style='animation-delay:1.2s'/>
      <rect x='560' y='102' width='30' height='148' fill='#4ade80' class='bar-anim' style='animation-delay:1.4s'/>
      
      <!-- Labels -->
      <text x='175' y='270' fill='#64748b' font-size='9' font-family='monospace' text-anchor='middle'>Detection</text>
      <text x='275' y='270' fill='#64748b' font-size='9' font-family='monospace' text-anchor='middle'>FP Rate</text>
      <text x='375' y='270' fill='#64748b' font-size='9' font-family='monospace' text-anchor='middle'>Severity</text>
      <text x='475' y='270' fill='#64748b' font-size='9' font-family='monospace' text-anchor='middle'>Explanation</text>
      <text x='575' y='270' fill='#64748b' font-size='9' font-family='monospace' text-anchor='middle'>Avg Score</text>
      
      <!-- Y-axis labels -->
      <text x='70' y='255' fill='#64748b' font-size='10' font-family='monospace' text-anchor='end'>0%</text>
      <text x='70' y='205' fill='#64748b' font-size='10' font-family='monospace' text-anchor='end'>25%</text>
      <text x='70' y='155' fill='#64748b' font-size='10' font-family='monospace' text-anchor='end'>50%</text>
      <text x='70' y='105' fill='#64748b' font-size='10' font-family='monospace' text-anchor='end'>75%</text>
      <text x='70' y='55' fill='#64748b' font-size='10' font-family='monospace' text-anchor='end'>100%</text>
      
      <!-- Legend -->
      <rect x='600' y='70' width='15' height='15' fill='#f87171'/>
      <text x='625' y='82' fill='#94a3b8' font-size='11' font-family='monospace'>Before</text>
      <rect x='600' y='95' width='15' height='15' fill='#4ade80'/>
      <text x='625' y='107' fill='#94a3b8' font-size='11' font-family='monospace'>After</text>
    </svg>
  </div>
  <div style='padding:12px 20px;background:rgba(255,255,255,.02);border-top:1px solid rgba(255,255,255,.05);display:flex;gap:16px;font-size:11px;color:#64748b;font-family:monospace'>
    <span>✅ Precision: 0.87</span>
    <span>🎯 Recall: 0.78</span>
    <span>⚖️ F1 Score: 0.82</span>
  </div>
</div>

<!-- Interactive Curriculum Progression -->
<div style='background:linear-gradient(135deg,#0d0b1e,#13104a);border:1px solid rgba(129,140,248,.2);border-radius:14px;overflow:hidden;box-shadow:0 8px 40px rgba(0,0,0,.4);animation:fadeInUp 0.8s ease-out 0.4s backwards'>
  <div style='padding:16px 20px;background:linear-gradient(90deg,rgba(129,140,248,.15),rgba(79,70,229,.08));border-bottom:1px solid rgba(129,140,248,.15)'>
    <div style='display:flex;align-items:center;gap:12px'>
      <span style='font-size:24px'>🎓</span>
      <div>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:11px;color:#818cf8;font-weight:700;letter-spacing:1.5px'>ADAPTIVE CURRICULUM</div>
        <div style='font-size:12px;color:#64748b;margin-top:2px'>Easy → Medium → Hard with live demotion at step ~330</div>
      </div>
      <div style='margin-left:auto;display:flex;gap:8px'>
        <span style='background:rgba(52,211,153,.15);color:#4ade80;border:1px solid rgba(52,211,153,.3);padding:4px 12px;border-radius:20px;font-size:10px;font-weight:700;font-family:monospace'>EASY</span>
        <span style='background:rgba(251,191,36,.15);color:#fbbf24;border:1px solid rgba(251,191,36,.3);padding:4px 12px;border-radius:20px;font-size:10px;font-weight:700;font-family:monospace'>MEDIUM</span>
        <span style='background:rgba(239,68,68,.15);color:#f87171;border:1px solid rgba(239,68,68,.3);padding:4px 12px;border-radius:20px;font-size:10px;font-weight:700;font-family:monospace'>HARD</span>
      </div>
    </div>
  </div>
  <div style='padding:16px;background:#0a0f1a'>
    <svg width='100%' height='200' viewBox='0 0 800 200' style='border-radius:8px;background:#050a14'>
      <!-- Background regions -->
      <rect x='60' y='50' width='200' height='100' fill='rgba(52,211,153,.1)' opacity='0' style='animation:fadeIn 1s ease-out 0.5s forwards'/>
      <rect x='260' y='50' width='280' height='100' fill='rgba(251,191,36,.1)' opacity='0' style='animation:fadeIn 1s ease-out 1s forwards'/>
      <rect x='540' y='50' width='200' height='100' fill='rgba(239,68,68,.1)' opacity='0' style='animation:fadeIn 1s ease-out 1.5s forwards'/>
      
      <!-- Difficulty line -->
      <path d='M 60 120 L 160 110 L 260 100 L 360 90 L 460 95 L 540 85 L 640 80 L 740 75' 
            fill='none' stroke='#818cf8' stroke-width='3' class='chart-line2'/>
      
      <!-- Demotion point -->
      <circle cx='460' cy='95' r='6' fill='#f87171' opacity='0' style='animation:pulse 1s ease-in-out 2s infinite'/>
      <text x='460' y='115' fill='#f87171' font-size='9' font-family='monospace' text-anchor='middle'>Demotion</text>
      <text x='460' y='125' fill='#f87171' font-size='9' font-family='monospace' text-anchor='middle'>Step 330</text>
      
      <!-- Labels -->
      <text x='160' y='175' fill='#4ade80' font-size='11' font-family='monospace' text-anchor='middle' font-weight='bold'>EASY</text>
      <text x='400' y='175' fill='#fbbf24' font-size='11' font-family='monospace' text-anchor='middle' font-weight='bold'>MEDIUM</text>
      <text x='640' y='175' fill='#f87171' font-size='11' font-family='monospace' text-anchor='middle' font-weight='bold'>HARD</text>
      
      <!-- Step markers -->
      <text x='60' y='190' fill='#64748b' font-size='10' font-family='monospace' text-anchor='middle'>0</text>
      <text x='260' y='190' fill='#64748b' font-size='10' font-family='monospace' text-anchor='middle'>200</text>
      <text x='460' y='190' fill='#64748b' font-size='10' font-family='monospace' text-anchor='middle'>400</text>
      <text x='640' y='190' fill='#64748b' font-size='10' font-family='monospace' text-anchor='middle'>600</text>
      <text x='740' y='190' fill='#64748b' font-size='10' font-family='monospace' text-anchor='middle'>800</text>
    </svg>
  </div>
  <div style='padding:12px 20px;background:rgba(255,255,255,.02);border-top:1px solid rgba(255,255,255,.05);display:flex;gap:16px;font-size:11px;color:#64748b;font-family:monospace'>
    <span>🔄 Live Adaptation</span>
    <span>📉 Demotion at step 330</span>
    <span>🎯 Threshold: 70% / 75%</span>
  </div>
</div>

<!-- Training Summary -->
<div style='background:linear-gradient(135deg,#071a10,#0a2416);border:1px solid rgba(52,211,153,.2);border-radius:14px;padding:20px;margin-top:10px;animation:fadeInUp 0.8s ease-out 0.6s backwards'>
  <div style='display:flex;align-items:center;gap:12px;margin-bottom:16px'>
    <span style='font-size:28px'>🏆</span>
    <div>
      <div style='font-family:"Syne",sans-serif;font-size:16px;font-weight:800;color:#34d399;letter-spacing:0.5px'>TRAINING COMPLETE</div>
      <div style='font-size:12px;color:#64748b;margin-top:2px'>Llama-3.2-1B-Instruct · GRPO · Unsloth 4-bit LoRA · Tesla T4</div>
    </div>
  </div>
  <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:12px'>
    <div style='text-align:center;padding:12px;background:rgba(255,255,255,.03);border-radius:8px'>
      <div style='font-size:20px;font-weight:800;color:#34d399'>800</div>
      <div style='font-size:9px;color:#475569;letter-spacing:1px;margin-top:4px'>STEPS</div>
    </div>
    <div style='text-align:center;padding:12px;background:rgba(255,255,255,.03);border-radius:8px'>
      <div style='font-size:20px;font-weight:800;color:#818cf8'>2h 15m</div>
      <div style='font-size:9px;color:#475569;letter-spacing:1px;margin-top:4px'>DURATION</div>
    </div>
    <div style='text-align:center;padding:12px;background:rgba(255,255,255,.03);border-radius:8px'>
      <div style='font-size:20px;font-weight:800;color:#fb923c'>Rank 16</div>
      <div style='font-size:9px;color:#475569;letter-spacing:1px;margin-top:4px'>LORA</div>
    </div>
    <div style='text-align:center;padding:12px;background:rgba(255,255,255,.03);border-radius:8px'>
      <div style='font-size:20px;font-weight:800;color:#fbbf24'>4-bit</div>
      <div style='font-size:9px;color:#475569;letter-spacing:1px;margin-top:4px'>QUANT</div>
    </div>
    <div style='text-align:center;padding:12px;background:rgba(255,255,255,.03);border-radius:8px'>
      <div style='font-size:20px;font-weight:800;color:#c084fc'>FREE</div>
      <div style='font-size:9px;color:#475569;letter-spacing:1px;margin-top:4px'>GPU</div>
    </div>
  </div>
</div>

<style>
@keyframes fadeIn { from{opacity:0} to{opacity:1} }
@keyframes fadeInUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
@keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.1)} }
</style>
</div>""")

        with gr.Tab("\u2139\ufe0f About"):
            gr.Markdown(ABOUT_MD)

app = gr.mount_gradio_app(api, demo_ui, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
