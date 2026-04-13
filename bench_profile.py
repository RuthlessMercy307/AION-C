"""Quick profiling of 3.6B vs 1.46B on H200."""
import sys, gc, time, torch, torch.nn.functional as F
sys.path.insert(0,".")
torch.cuda.empty_cache()

SEQ = 512
BAR = "=" * 60

def timed(fn, label, n=3):
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter()-t0)*1000)
    avg = sum(times)/len(times)
    print(f"  {label:<30s}: {avg:>8.1f} ms")
    return r, avg

# ================================================================
print(BAR)
print("TEST 1: Component profiling - 3.6B (hd=1536, dl=28)")
print(BAR)

from router.pipeline import MoSEPipeline, MoSEConfig

cfg36 = MoSEConfig(
    hidden_dim=1536, vocab_size=32_000,
    enc_n_layers=14, dec_n_layers=28,
    enc_state_dim=16, dec_state_dim=16,
    dec_n_heads=16, motor_n_heads=8, unif_n_heads=8,
    dec_max_seq_len=2048, orch_mlp_hidden=2048, motor_max_nodes=32,
)
p = MoSEPipeline(cfg36).cuda()
p.train()
ids = torch.randint(1, 32000, (1, SEQ), device="cuda")

# Warmup
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = p(ids); out.logits.sum().backward(); p.zero_grad()
torch.cuda.synchronize()

print(f"\nForward components (batch=1, seq={SEQ}):")
_, t_enc = timed(lambda: p.encoder(ids), "encoder (14 layers)")

concepts = p.encoder(ids)
_, t_orch = timed(lambda: p.orchestrator(concepts), "orchestrator")

motor = p.motors["cora"]
_, t_cryst = timed(lambda: motor.build_graph(concepts), "crystallizer (cora)")

cryst = motor.build_graph(concepts)
n = cryst.node_counts[0]
nf = cryst.node_vectors[0, :n]
g = cryst.graphs[0]
_, t_cre = timed(lambda: motor.reason(g, nf, n_iterations=10), "CRE (cora, 10 iters)")

graph_repr = torch.zeros(1, 32, 1536, device="cuda")
def dec_fwd():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        return p.decoder(ids, graph_repr, concepts)
_, t_dec = timed(dec_fwd, "decoder (28 layers)")

_, t_fwd = timed(lambda: p(ids), "FULL forward")

def do_bwd():
    p.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        o = p(ids)
        l = F.cross_entropy(o.logits[0, :-1], ids[0, 1:])
    l.backward()
_, t_bwd = timed(do_bwd, "FULL fwd+backward")

print(f"\n  Summary 3.6B:")
print(f"    encoder:  {t_enc:>6.0f} ms")
print(f"    orch:     {t_orch:>6.0f} ms")
print(f"    cryst:    {t_cryst:>6.0f} ms")
print(f"    CRE:      {t_cre:>6.0f} ms")
print(f"    decoder:  {t_dec:>6.0f} ms")
print(f"    fwd total:{t_fwd:>6.0f} ms")
print(f"    fwd+bwd:  {t_bwd:>6.0f} ms")

del p, concepts, cryst, nf, g, graph_repr, out
gc.collect(); torch.cuda.empty_cache()

# ================================================================
print(f"\n{BAR}")
print("TEST 2: Component profiling - 1.46B (hd=1024, dl=24)")
print(BAR)

cfg14 = MoSEConfig(
    hidden_dim=1024, vocab_size=32_000,
    enc_n_layers=14, dec_n_layers=24,
    enc_state_dim=16, dec_state_dim=16,
    dec_max_seq_len=2048, orch_mlp_hidden=512, motor_max_nodes=32,
)
p2 = MoSEPipeline(cfg14).cuda()
p2.train()
ids = torch.randint(1, 32000, (1, SEQ), device="cuda")
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    o = p2(ids); o.logits.sum().backward(); p2.zero_grad()
torch.cuda.synchronize()

print(f"\nForward components (batch=1, seq={SEQ}):")
_, t2_enc = timed(lambda: p2.encoder(ids), "encoder (14 layers)")
c2 = p2.encoder(ids)
_, t2_orch = timed(lambda: p2.orchestrator(c2), "orchestrator")
m2 = p2.motors["cora"]
_, t2_cryst = timed(lambda: m2.build_graph(c2), "crystallizer (cora)")
cr2 = m2.build_graph(c2)
nn2 = cr2.node_counts[0]; nf2 = cr2.node_vectors[0, :nn2]; g2 = cr2.graphs[0]
_, t2_cre = timed(lambda: m2.reason(g2, nf2, n_iterations=10), "CRE (cora, 10 iters)")
gr2 = torch.zeros(1, 32, 1024, device="cuda")
def dec_fwd2():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        return p2.decoder(ids, gr2, c2)
_, t2_dec = timed(dec_fwd2, "decoder (24 layers)")

def do_bwd2():
    p2.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        o = p2(ids); l = F.cross_entropy(o.logits[0, :-1], ids[0, 1:]); l.backward()
_, t2_bwd = timed(do_bwd2, "FULL fwd+backward")

print(f"\n  Summary 1.46B:")
print(f"    encoder:  {t2_enc:>6.0f} ms")
print(f"    orch:     {t2_orch:>6.0f} ms")
print(f"    cryst:    {t2_cryst:>6.0f} ms")
print(f"    CRE:      {t2_cre:>6.0f} ms")
print(f"    decoder:  {t2_dec:>6.0f} ms")
print(f"    fwd+bwd:  {t2_bwd:>6.0f} ms")

print(f"\n  Slowdown 3.6B vs 1.46B:")
print(f"    encoder:  {t_enc/t2_enc:.1f}x")
print(f"    CRE:      {t_cre/t2_cre:.1f}x")
print(f"    decoder:  {t_dec/t2_dec:.1f}x")
print(f"    fwd+bwd:  {t_bwd/t2_bwd:.1f}x")

del p2, c2, cr2, nf2, g2, gr2
gc.collect(); torch.cuda.empty_cache()

# ================================================================
print(f"\n{BAR}")
print("TEST 3: Vocab size impact (decoder only, hd=1024, 8 layers)")
print(BAR)

from decoder.model import StreamDecoder
from decoder.config import StreamDecoderConfig

for vs in [512, 8000, 32000]:
    dc = StreamDecoderConfig(vocab_size=vs, hidden_dim=1024, n_layers=8, n_heads=8,
                             node_dim=1024, max_graph_nodes=32, max_seq_len=512,
                             state_dim=16, expand=2, d_conv=4, ffn_mult=4)
    d = StreamDecoder(dc).cuda(); d.train()
    ti = torch.randint(1, vs, (1, SEQ), device="cuda")
    gr = torch.zeros(1, 32, 1024, device="cuda")
    ec = torch.randn(1, SEQ, 1024, device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): d(ti, gr, ec)
    torch.cuda.synchronize()
    def dec_v(dd=d, tt=ti, gg=gr, ee=ec):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return dd(tt, gg, ee)
    timed(dec_v, f"decoder vocab={vs}")
    del d, ti, gr, ec; gc.collect(); torch.cuda.empty_cache()

# ================================================================
print(f"\n{BAR}")
print("TEST 4: Seq length impact (3.6B full pipeline)")
print(BAR)

p3 = MoSEPipeline(cfg36).cuda(); p3.train()
for sl in [64, 128, 256, 512]:
    ids_sl = torch.randint(1, 32000, (1, sl), device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        o = p3(ids_sl); o.logits.sum().backward(); p3.zero_grad()
    torch.cuda.synchronize()
    def do_sl(s=sl):
        p3.zero_grad()
        i = torch.randint(1, 32000, (1, s), device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = p3(i); l = F.cross_entropy(o.logits[0, :-1], i[0, 1:]); l.backward()
    timed(do_sl, f"seq={sl} fwd+bwd", n=2)

print(f"\n{BAR}")
print("PROFILING DONE")
print(BAR)
