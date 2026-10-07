import asyncio, json, os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from sim import World, DT, MATCH
import agent

DECISION_EVERY = float(os.getenv("DECISION_EVERY", "0.5"))   # game-seconds between decisions per player
w, client = World(), agent.make_client()
clients, inflight = set(), set()
paused = False


async def think(p):
    try:
        state, q = agent.prepare(w, p)
        a = await asyncio.wait_for(asyncio.to_thread(agent.call, client, state, q), timeout=3)
        w.apply(p, a)
    except Exception as e:      # on any failure the player simply keeps its last intent
        print(p.id, "decision failed:", repr(e))
    finally:
        inflight.discard(p.id)


batching = False


async def think_all():
    """One batched Laya call for all players (single GPU pass)."""
    global batching
    try:
        ps = list(w.players)
        items = [agent.prepare(w, p) for p in ps]
        outs = await asyncio.wait_for(asyncio.to_thread(agent.call_batch, client, items), timeout=5)
        for p, a in zip(ps, outs): w.apply(p, a)
    except Exception as e:
        print("batch decision failed:", repr(e)[:200])
    finally:
        batching = False


async def game_loop():
    global batching
    nxt = 0.0
    while True:
        if w.t < MATCH and not paused:
            w.step()
            if w.t >= nxt:
                nxt = w.t + DECISION_EVERY
                if client and client[0] == "laya":          # Laya: all players in one batched call
                    if not batching: batching = True; asyncio.create_task(think_all())
                else:
                    for p in w.players:
                        if p.id not in inflight:        # skip players whose last call is still running
                            inflight.add(p.id); asyncio.create_task(think(p))
        msg = json.dumps({**w.snapshot(), "paused": paused})
        for c in list(clients):
            try: await c.send_text(msg)
            except Exception: clients.discard(c)
        await asyncio.sleep(DT)


@asynccontextmanager
async def lifespan(app):
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(48))
    task = asyncio.create_task(game_loop()); yield; task.cancel()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def index(): return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

@app.websocket("/ws")
async def ws(sock: WebSocket):
    global paused
    await sock.accept(); clients.add(sock)
    try:
        while True:
            if await sock.receive_text() == "toggle": paused = not paused
    except WebSocketDisconnect: clients.discard(sock)