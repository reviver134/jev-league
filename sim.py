import math, random

L, W = 50.0, 30.0                        # pitch, meters (x: 0..L, y: 0..W)
GOAL_W, BAR, HEAD, G, DT = 5.0, 2.44, 2.0, 9.8, 1 / 30
MATCH = 180.0                           # match length, seconds
SPEEDS = {"jog": 5.0, "sprint": 8.0}
INTENTS = {"go_to_ball", "hold_position", "support_attack", "press_carrier",
           "mark_opponent", "drop_back", "hold_line", "come_out"}
FORM = [("GK", .04, .5), ("DEF", .30, .35), ("FWD", .60, .65)]   # 3v3, index 0 = goalkeeper, index 2 = kicks off


def dist(a, b): return math.hypot(a.x - b.x, a.y - b.y)
def clamp(v, lo, hi): return max(lo, min(hi, v))
def attack_x(team): return L if team == "A" else 0.0     # A attacks right, B attacks left
def sign(team): return 1 if team == "A" else -1

def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = clamp(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy or 1e-9), 0, 1)
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


class P:
    def __init__(s, id, team, role, home):
        s.id, s.team, s.role, s.home = id, team, role, home
        s.x, s.y = home; s.fx, s.fy = sign(team), 0.0
        s.intent, s.pace, s.cd, s.tk = "shape", "jog", 0.0, 0.0
        s.arg, s.menu, s.info, s.label, s.auto = None, {}, {}, "", False
        s.st = dict.fromkeys(("goals", "assists", "passes", "passes_ok", "progressive", "interceptions", "tackles_won", "saves"), 0)


class Ball:
    def __init__(s): s.x, s.y, s.z, s.vx, s.vy, s.vz = L / 2, W / 2, 0.0, 0.0, 0.0, 0.0


class World:
    def __init__(s):
        s.players = []
        for team in "AB":
            for i, (role, fx, fy) in enumerate(FORM):
                hx, hy = (fx * L, fy * W) if team == "A" else ((1 - fx) * L, (1 - fy) * W)
                s.players.append(P(f"{team}{i}", team, role, (hx, hy)))
        s.by_id = {p.id: p for p in s.players}
        s.score, s.t, s.events, s.log, s.last_kicker = {"A": 0, "B": 0}, 0.0, [], [], None
        s.pending, s.assist, s.last_shot = None, None, None
        s.kickoff("A")

    def note(s, text):                                        # event memory: shown on screen and sent to the AI
        s.events.append(text); s.log.append((s.t, text))

    def recent(s, n=5):
        return [f"{max(0, int(s.t - t))}s ago: {x}" for t, x in s.log[-n:]]

    def kickoff(s, team):
        for p in s.players: p.x, p.y = p.home; p.cd = 0.0; p.intent = "shape"
        s.pending = s.assist = None
        s.ball = Ball(); k = s.by_id[team + "2"]
        k.x, k.y = L / 2 - 0.5 * sign(team), W / 2; s.owner = k

    # ---------- decisions from the AI are applied here ----------
    def apply(s, p, a):
        p.info, m = a.get("_info", {}), p.menu
        lab = a.get("intent")
        if lab in m.get("intent", {}): p.intent, p.arg = m["intent"][lab]; p.label = lab.replace(" · continue", "")
        act, tgt = m.get("action", {}).get(a.get("action")), m.get("pass_target", {}).get(a.get("pass_target"))
        if s.owner is p:
            if act == "pass": s.kick_pass(p, tgt)
            elif act == "shoot": s.kick_shot(p, "center", "medium")

    def gain(s, p):                                           # a player takes the ball: settle pass / interception stats
        pp, s.pending = s.pending, None
        if pp and p.id != pp["from"]:
            if p.team == pp["team"]:
                fr = s.by_id[pp["from"]]; fr.st["passes_ok"] += 1; s.assist = (fr.id, p.id, s.t)
                if abs(attack_x(p.team) - pp["x0"]) - abs(attack_x(p.team) - p.x) >= 5: fr.st["progressive"] += 1
            else: p.st["interceptions"] += 1
        s.owner = p

    def launch(s, p, tx, ty, speed, deg):
        s.last_kicker = p.id; s.pending = None
        b = s.ball; ang = math.atan2(ty - b.y, tx - b.x) + random.gauss(0, 0.015 + 0.0015 * speed)
        th = math.radians(deg); s.owner = None; p.cd = 0.6
        b.vx, b.vy, b.vz = speed * math.cos(th) * math.cos(ang), speed * math.cos(th) * math.sin(ang), speed * math.sin(th)

    def kick_pass(s, p, mid):
        m = s.by_id.get(mid)
        if not m or m.team != p.team or m is p: return
        d = dist(p, m)
        blocked = any(o.team != p.team and seg_dist(o.x, o.y, p.x, p.y, m.x, m.y) < 2.5 for o in s.players)
        if blocked and d > 10: deg, sp = 35, math.sqrt(d * G / math.sin(math.radians(70)))   # lob over heads
        else: deg, sp = 0, clamp(0.6 * d + 3, 6, 30)                                          # ground pass
        s.note(f"{p.id} {'lobbed' if deg else 'passed'} to {m.id}")
        s.launch(p, m.x, m.y, sp, deg)
        p.st["passes"] += 1; s.pending = {"from": p.id, "team": p.team, "x0": p.x}

    def kick_shot(s, p, aim, power):
        if p.role == "GK":                                   # keeper "shoot" = clearance to midfield, never a shot on goal
            tx, ty = L / 2, W / 2 + {"top": -8, "center": 0, "bottom": 8}.get(aim, 0)
            d = math.hypot(tx - p.x, ty - p.y) * 0.75        # ball also bounces and rolls on
            s.note(f"{p.id} (goalkeeper) cleared the ball upfield")
            return s.launch(p, tx, ty, math.sqrt(d * G / math.sin(math.radians(60))), 30)
        d = abs(attack_x(p.team) - p.x)
        if d > 28: return                                    # too far to shoot: keep dribbling
        gk = s.by_id[("B" if p.team == "A" else "A") + "0"]
        gy = W / 2 + (1.8 if gk.y < W / 2 else -1.8)         # aim at the side away from the keeper
        power = "medium" if d <= 18 else "high"
        sp, deg = {"medium": (23, 3), "high": (30, 10)}[power]
        s.last_shot = (p.team, s.t)
        s.note(f"{p.id} shot at goal ({power} power)")
        s.launch(p, attack_x(p.team), gy, sp, deg)

    # ---------- movement targets ----------
    def target(s, p):
        b, sg, ox = s.ball, sign(p.team), attack_x(p.team); own = L - ox; k, h = p.intent, s.owner; p.auto = False
        if h is p: return ox, W / 2
        if p.role != "GK" and h is None:                       # safety rule: closest outfielder chases a loose ball (shown in the panel)
            if p is min((q for q in s.players if q.team == p.team and q.role != "GK"), key=lambda q: dist(q, b)): p.auto = True; return b.x, b.y
        if p.role == "GK":
            if k == "come_out": return own + sg * min(8, abs(b.x - own)), clamp(b.y, W / 2 - 8, W / 2 + 8)
            return own + sg * 1.5, clamp(b.y, W / 2 - GOAL_W / 2, W / 2 + GOAL_W / 2)
        if k in ("chase", "press"): return b.x, b.y
        if k == "track" and p.arg in s.by_id: o = s.by_id[p.arg]; return o.x - sg * 2.0, o.y
        if k == "block":
            dx, dy = own - b.x, W / 2 - b.y; d = math.hypot(dx, dy) or 1; t = min(6, d) / d; return b.x + dx * t, b.y + dy * t
        if k == "behind": return b.x - sg * 6, (p.home[1] + b.y) / 2
        if k == "short" and h and h.team == p.team: return h.x - sg * 3, h.y + (5 if h.y < W / 2 else -5)
        if k == "run": return b.x + sg * 10, (p.home[1] + b.y) / 2
        if k == "forward": return clamp(p.x + sg * 8, 0, L), p.y
        if k == "drop": return p.home[0] - sg * 12, p.home[1]
        if k == "hold": return p.x, p.y
        return p.home[0] + (b.x - L / 2) * 0.3, p.home[1]       # "shape": usual zone, drifting with the ball

    # ---------- physics step ----------
    def step(s, dt=DT):
        s.t += dt; b = s.ball
        for p in s.players:
            p.cd = max(0.0, p.cd - dt); p.tk = max(0.0, p.tk - dt)
            tx, ty = s.target(p); dx, dy = tx - p.x, ty - p.y; d = math.hypot(dx, dy)
            if d > 0.3:
                m = min(SPEEDS["sprint" if d > 6 else p.pace] * (0.9 if s.owner is p else 1) * dt, d)   # sprint when far from target
                p.x += dx / d * m; p.y += dy / d * m; p.fx, p.fy = dx / d, dy / d
            p.x, p.y = clamp(p.x, 0, L), clamp(p.y, 0, W)
        for i, a in enumerate(s.players):                      # separation: no overlapping players
            for c in s.players[i + 1:]:
                d = math.hypot(a.x - c.x, a.y - c.y)
                if 0 < d < 1.0:
                    k = (1.0 - d) / 2 / d; dx, dy = (a.x - c.x) * k, (a.y - c.y) * k
                    a.x += dx; a.y += dy; c.x -= dx; c.y -= dy
        if s.owner:                                            # engine rule: close defenders try to tackle (random order = fair to both teams)
            tk = [q for q in s.players if q.team != s.owner.team and q.tk == 0 and dist(q, s.owner) < 2.0]
            random.shuffle(tk)
            for q in tk:
                q.tk = 0.8; victim = s.owner.id
                if random.random() < 0.35: s.owner = q; s.pending = None; q.st["tackles_won"] += 1; s.note(f"{q.id} won the ball from {victim}"); break
                s.note(f"{q.id} failed a tackle on {victim}")
        if s.owner:
            o = s.owner; b.x, b.y, b.z = o.x + o.fx * .6, o.y + o.fy * .6, 0.0; b.vx = b.vy = b.vz = 0.0
        else:
            b.vz -= G * dt; b.z += b.vz * dt
            if b.z <= 0: b.z = 0.0; b.vz = -b.vz * .5 if b.vz < -1 else 0.0
            if b.z == 0: f = math.exp(-0.6 * dt); b.vx *= f; b.vy *= f
            b.x += b.vx * dt; b.y += b.vy * dt
            sp = math.hypot(b.vx, b.vy)
            for p in random.sample(s.players, len(s.players)):     # random order so neither team has priority
                                                                   # interception: only below head height
                gk = p.role == "GK"
                if p.cd > 0 or b.z > (2.6 if gk else HEAD) or math.hypot(p.x - b.x, p.y - b.y) > (1.8 if gk else 1.0): continue
                if gk:                                               # keeper: catch chance drops with shot speed, else parry
                    if s.last_shot and s.last_shot[0] != p.team and s.t - s.last_shot[1] < 4: p.st["saves"] += 1
                    if sp <= 10 or random.random() < max(0.35, 0.95 - 0.03 * (sp - 10)): s.gain(p); s.note(f"{p.id} (goalkeeper) saved and caught the ball")
                    else: b.vx *= -.4; b.vy *= -.4; p.cd = .5; s.note(f"{p.id} (goalkeeper) parried the ball")
                    break
                if sp > 25: continue
                s.gain(p); s.note(f"{p.id} got the ball"); break
            if b.x <= 0 or b.x >= L:
                if abs(b.y - W / 2) <= GOAL_W / 2 and b.z < BAR:  # under the crossbar = goal
                    t = "A" if b.x >= L else "B"; s.score[t] += 1; s.note(f"GOAL for team {t}! scored by {s.last_kicker}")
                    sc = s.by_id.get(s.last_kicker)
                    if sc and sc.team == t:
                        sc.st["goals"] += 1
                        if s.assist and s.assist[1] == sc.id and s.t - s.assist[2] < 6: s.by_id[s.assist[0]].st["assists"] += 1
                    s.kickoff("B" if t == "A" else "A"); return
                s.pending = None; s.note("the ball went out / over the bar at the goal line"); b.x = clamp(b.x, .5, L - .5); b.vx = b.vy = 0.0   # over the bar / wide: stops at line
            if b.y < 0 or b.y > W: b.y = clamp(b.y, 0, W); b.vx = b.vy = 0.0

    def snapshot(s):
        b = s.ball
        return {"left": round(max(0, MATCH - s.t), 1), "score": s.score, "ball": [round(b.x, 2), round(b.y, 2), round(b.z, 2)],
                "owner": s.owner.id if s.owner else None, "events": s.events[-6:],
                "players": [[p.id, round(p.x, 2), round(p.y, 2)] for p in s.players],
                "detail": {p.id: {"role": p.role, "label": p.label, "auto": p.auto, "info": p.info, "st": p.st} for p in s.players}}