"""Builds each player's state + typed questions, calls Jev, returns a decision dict."""
import os, random, threading
from sim import L, W, GOAL_W, BAR, MATCH, dist, attack_x, seg_dist

BRIEF = ("You are a football {role}. Play your position: defenders protect their goal and pass safely; "
         "midfielders link defence and attack; forwards create chances and shoot. Pass to an open teammate "
         "who is closer to the opponent goal, shoot when within ~15 m of goal with a clear lane, otherwise dribble.")
BRIEF_GK = ("You are the GOALKEEPER. Stay close to your goal line and track the ball's y position. Come out only "
            "when the ball is loose and close to your goal. When you hold the ball, clear it upfield with a "
            "powerful kick or pass safely to a defender. Never run far from your goal.")


BACKEND = os.getenv("BACKEND", "mock").lower()     # jev | laya | mock
_laya_lock = threading.Lock()                       # torch forwards must not run concurrently


def make_client():
    """Returns (backend_name, client) or None for mock mode."""
    if os.getenv("MOCK") == "1" or BACKEND == "mock":
        print("MOCK mode: random decisions, no model calls"); return None
    if BACKEND == "jev":
        from typesafe_sdk import TypeSafeClient
        print("Backend: Jev (TypeSafe API)"); return ("jev", TypeSafeClient())     # reads TYPESAFE_API_KEY
    if BACKEND == "laya":
        from laya import Router
        print("Backend: Laya (local). Loading model, first run downloads it...")
        r = Router(); r.preload([os.getenv("LAYA_MODEL") or "english"]); print("Laya ready"); return ("laya", r)
    raise SystemExit(f"Unknown BACKEND={BACKEND!r}, use jev, laya or mock")


def to_text(st):
    """Compact plain-text version of the state (Laya has a small context window)."""
    y, b, p = st["you"], st["ball"], st["pitch"]
    f = lambda m: f"{m['id']} {m['role']} ({m['pos'][0]},{m['pos'][1]})"
    return (f"{st['role_brief']}\nSituation: {' '.join(st['situation'])}\n"
            f"Pitch x 0-{p['x_range'][1]:g}, y 0-{p['y_range'][1]:g}. You attack the goal at x={p['opponent_goal_x']:g}; "
            f"your own goal is at x={p['own_goal_x']:g}. Goal spans y {p['goal_y_range'][0]}-{p['goal_y_range'][1]}. 'top' = small y, 'bottom' = large y.\n"
            f"You: {y['id']} {y['role']} at ({y['pos'][0]},{y['pos'][1]}), home ({y['home_pos'][0]},{y['home_pos'][1]}), has ball: {y['has_ball']}. "
            f"Distance to ball {y['dist_to_ball']} m, to opponent goal {y['dist_to_opponent_goal']} m, nearest opponent {y['nearest_opponent_dist']} m. Closest to ball in your team: {y['closest_to_ball_in_team']}.\n"
            f"Ball at ({b['pos'][0]},{b['pos'][1]}), height {b['height']}, speed {b['speed']}, holder {b['holder']}.\n"
            f"Teammates: " + "; ".join(f(m) + f" {m['dist']} m away, {'open' if m['open'] else 'marked'}" for m in st["teammates"]) + ".\n"
            f"Opponents: " + "; ".join(f(o) for o in st["opponents"]) + ".\n"
            f"Score: you {st['score']['you']}, them {st['score']['them']}. Time left {st['time_left_s']} s.\n"
            f"Recent events: " + (" | ".join(st["recent_events"]) or "none yet") + ".")


def _r(v): return round(v, 1)


def prepare(w, p):
    """Returns (state, questions) for one player. Runs on the game loop thread."""
    b, ox, has = w.ball, attack_x(p.team), w.owner is p
    mates = [q for q in w.players if q.team == p.team and q is not p]
    opps = [q for q in w.players if q.team != p.team]
    h, dball, dgoal = w.owner, dist(p, b), ((p.x - ox) ** 2 + (p.y - W / 2) ** 2) ** .5
    sit = [("You have the ball." if has else f"Your teammate {h.id} has the ball." if h and h.team == p.team
            else f"The opponent {h.id} has the ball, {_r(dball)} m from you." if h else f"The ball is loose, {_r(dball)} m from you.")]
    if has:
        lane = all(seg_dist(o.x, o.y, p.x, p.y, ox, W / 2) > 2 for o in opps)
        sit.append(f"You are {_r(dgoal)} m from the goal; the shooting lane is {'clear' if lane else 'blocked'}.")
        sit.append("An opponent is close to you." if min(dist(p, o) for o in opps) < 4 else "No opponent is close to you.")
        sit.append("A teammate is open and closer to the goal than you." if any(min(dist(m, o) for o in opps) > 4 and abs(m.x - ox) < abs(p.x - ox) for m in mates) else "No teammate is open ahead of you.")
    state = {
        "situation": sit,
        "role_brief": BRIEF_GK if p.role == "GK" else BRIEF.format(role=p.role),
        "pitch": {"x_range": [0, L], "y_range": [0, W], "opponent_goal_x": ox, "own_goal_x": L - ox,
                  "goal_y_range": [_r(W / 2 - GOAL_W / 2), _r(W / 2 + GOAL_W / 2)], "crossbar_height": BAR,
                  "note": "y grows downward on screen; 'top' = small y, 'bottom' = large y"},
        "you": {"id": p.id, "role": p.role, "pos": [_r(p.x), _r(p.y)], "home_pos": [_r(p.home[0]), _r(p.home[1])],
                "has_ball": has, "dist_to_ball": _r(dist(p, b)),
                "dist_to_opponent_goal": _r(((p.x - ox) ** 2 + (p.y - W / 2) ** 2) ** .5),
                "nearest_opponent_dist": _r(min(dist(p, o) for o in opps)),
                "closest_to_ball_in_team": p is min(mates + [p], key=lambda q: dist(q, b))},
        "ball": {"pos": [_r(b.x), _r(b.y)], "height": _r(b.z), "speed": _r((b.vx ** 2 + b.vy ** 2) ** .5),
                 "holder": w.owner.id if w.owner else None},
        "teammates": [{"id": q.id, "role": q.role, "pos": [_r(q.x), _r(q.y)], "dist": _r(dist(p, q)),
                       "open": min(dist(q, o) for o in opps) > 4} for q in mates],
        "opponents": [{"id": o.id, "role": o.role, "pos": [_r(o.x), _r(o.y)]} for o in opps],
        "score": {"you": w.score[p.team], "them": w.score["B" if p.team == "A" else "A"]},
        "time_left_s": _r(MATCH - w.t),
        "recent_events": w.recent(5),                        # short-term memory of the match
    }
    gk, q, p.menu = p.role == "GK", {}, {}
    if has:
        acts = [("Pass to a teammate", "pass", "pass to a teammate, best when an opponent is close or a teammate is open ahead"),
                ("Clear the ball" if gk else "Take a shot", "shoot", "clear the ball upfield, best when no teammate is open" if gk else "shoot at the goal, best when within 15 m with a clear lane"),
                ("Dribble forward", "dribble", "keep running with the ball, best when no opponent is close")]
        q["action"] = {"type": "choice", "instructions": "You have the ball. What do you do?", "criteria": {l: d for l, _, d in acts}}
        q["pass_target"] = {"type": "choice", "instructions": "Which teammate should receive the pass? Prefer open teammates closer to the opponent goal.",
                            "criteria": {f"Pass to {m.id}": f"teammate {m.id} ({m.role})" for m in mates}}
        p.menu = {"action": {l: k for l, k, _ in acts}, "pass_target": {f"Pass to {m.id}": m.id for m in mates}}
    else:
        mn = build_menu(w, p)
        q["intent"] = {"type": "choice", "instructions": "What should this player do next, given the situation?", "criteria": {l: d for l, d, _, _ in mn}}
        p.menu = {"intent": {l: (k, ar) for l, _, k, ar in mn}}
    return state, q


CONT = " · continue"


def build_menu(w, p):
    """Situational, named options (like 'Track B2'); the previous choice comes back as '... · continue'."""
    h = w.owner
    if p.role == "GK":
        m = [("Hold the goal line", "stay on the goal line tracking the ball", "hold_line", None),
             ("Come out to meet the ball", "advance to narrow the angle or claim a loose ball near your goal", "come_out", None)]
    else:
        m = []
        if h is None: m.append(("Go for the loose ball", "run to the loose ball, best when it is near you", "chase", None))
        if h and h.team != p.team:
            m += [("Press the ball carrier", "close down the opponent with the ball, best when he is near you", "press", None),
                  ("Block the route to goal", "get between the ball and your goal, best when the opponent attacks", "block", None)]
        if h and h.team == p.team:
            m += [("Support from behind", "stay behind the ball to offer a safe pass", "behind", None),
                  ("Come short", "move close to the ball carrier to receive a short pass", "short", None),
                  ("Make a forward run", "run ahead of the ball toward the opponent goal", "run", None)]
        m += [("Move forward", "advance up the pitch", "forward", None), ("Rejoin the team shape", "return to your usual zone", "shape", None),
              ("Hold position", "stay where you are", "hold", None), ("Drop back", "retreat toward your own goal", "drop", None)]
        for o in sorted((o for o in w.players if o.team != p.team and o.role != "GK"), key=lambda o: dist(p, o))[:2]:
            m.append((f"Track {o.id}", f"stay goal-side of {o.id}, best when {o.id} is close to your goal", "track", o.id))
    return [(l + CONT, "keep doing the same as before: " + d, k, ar) if l == p.label else (l, d, k, ar) for l, d, k, ar in m]


def _g(a, k): return a.get(k) if isinstance(a, dict) else getattr(a, k, None)


def _ans(resp, k):
    for attr in ("answers", "choices", "scores"):
        d = resp.get(attr) if isinstance(resp, dict) else getattr(resp, attr, None)
        if d and k in d: return d[k]
    raise KeyError(k)


def _top3(ans, labels):
    """Top 3 options as [label, percent], the chosen one first (needs per-option probabilities from the model)."""
    ch = _g(ans, "choice"); pr = next((_g(ans, k) for k in ("probabilities", "probs", "distribution") if _g(ans, k) is not None), None)
    if isinstance(pr, dict): it = list(pr.items())
    elif isinstance(pr, (list, tuple)) and len(pr) == len(labels): it = list(zip(labels, pr))
    else: it = [(ch, _g(ans, "confidence") or 1.0)]
    tot = sum(float(v) for _, v in it) or 1
    it = sorted(((str(k), float(v) / tot) for k, v in it), key=lambda t: -t[1])
    top = [t for t in it if t[0] == ch] + [t for t in it if t[0] != ch]
    return [[k, round(v * 100)] for k, v in top[:3]]


def _decide(resp, questions):
    out, info = {}, {}
    for k, q in questions.items():
        ans = _ans(resp, k); out[k] = _g(ans, "choice"); info[k] = _top3(ans, list(q["criteria"]))
    out["_info"] = info
    return out


def _mock(questions):
    ans = {}
    for k, q in questions.items():
        ls = list(q["criteria"]); wt = [random.random() ** 3 for _ in ls]; t = sum(wt)
        ans[k] = {"choice": random.choices(ls, wt)[0], "probabilities": {l: x / t for l, x in zip(ls, wt)}}
    return {"answers": ans}


def call(client, state, questions):
    """Blocking call, run in a worker thread. Returns chosen labels plus '_info' (top-3 probabilities)."""
    if client is None: resp = _mock(questions)
    elif client[0] == "jev": resp = client[1].system_one(state=state, questions=questions)
    else:
        kw = {"model": os.getenv("LAYA_MODEL")} if os.getenv("LAYA_MODEL") else {}
        with _laya_lock: resp = client[1].predict(to_text(state), questions, **kw)
        if os.getenv("DEBUG") == "2": print(resp)
    out = _decide(resp, questions)
    if os.getenv("DEBUG") == "1": print(state["you"]["id"], {k: v for k, v in out.items() if k != "_info"}, out["_info"])
    return out


def call_batch(client, items):
    """Laya only: answer many players in ONE GPU pass. items = [(state, questions), ...]."""
    model = os.getenv("LAYA_MODEL")
    reqs = [{"state": to_text(s), "questions": q, **({"model": model} if model else {})} for s, q in items]
    with _laya_lock: res = client[1].predict_batch(reqs)
    return [_decide(r, q) for r, (_, q) in zip(res, items)]