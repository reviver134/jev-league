import time, torch
from laya import Router
import sim, agent

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
r = Router(); r.preload(["typed-decisions"])
w = sim.World()
st, q = agent.prepare(w, w.by_id["A2"])        # A2 starts with the ball
text = agent.to_text(st)
r.predict(text, q, model="typed-decisions")    # warm-up, not timed
t = time.time()
for _ in range(20): r.predict(text, q, model="typed-decisions")
ms = (time.time() - t) / 20 * 1000
print(f"{ms:.0f} ms per call, so about {ms*6:.0f} ms for all 6 players")