"""Analyze token lengths of Opus + IT datasets."""
import sys, json
sys.path.insert(0, ".")
from tokenizer import AIONTokenizer
from pathlib import Path

tok = AIONTokenizer(Path("tokenizer/aion_32k.model"))

def stats(lengths, name):
    if not lengths:
        print(f"  {name}: NO DATA")
        return
    lengths.sort()
    n = len(lengths)
    mean = sum(lengths) / n
    med = lengths[n // 2]
    p75 = lengths[int(n * 0.75)]
    p90 = lengths[int(n * 0.90)]
    p95 = lengths[int(n * 0.95)]
    p99 = lengths[int(n * 0.99)]
    mx = lengths[-1]
    pad = " " * 14
    print(f"  {name:<14s} n={n:>6,}  mean={mean:>5.0f}  med={med:>4}  "
          f"p75={p75:>4}  p90={p90:>4}  p95={p95:>5}  p99={p99:>5}  max={mx:>5}")
    trunc_parts = []
    for sl in [64, 128, 256, 512]:
        cut = sum(1 for l in lengths if l > sl)
        trunc_parts.append(f"{sl}={cut/n*100:>5.1f}%")
    print(f"  {pad} truncated at: {'  '.join(trunc_parts)}")

print("=" * 85)
print("OPUS DATASETS (BPE 32K tokenizer)")
print("=" * 85)

opus_dir = Path("datasets/opus")
all_lens = []
for fname in sorted(opus_dir.glob("mose_*.jsonl")):
    if "combined" in fname.name:
        continue
    motor = fname.stem.replace("mose_", "")
    lens = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            text = ex.get("input", "") + " " + ex.get("expected_output", "")
            toks = tok.encode(text)
            lens.append(len(toks))
    stats(lens, motor)
    all_lens.extend(lens)
    print()

stats(all_lens, "ALL OPUS")

print()
print("=" * 85)
print("INSTRUCTION TUNING DATASET (BPE 32K tokenizer)")
print("=" * 85)

it_path = Path("datasets/instruction_tuning.jsonl")
it_lens = []
with open(it_path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        ex = json.loads(line)
        text = ex.get("instruction", "") + " " + ex.get("response", "")
        if ex.get("system_prompt"):
            text = ex["system_prompt"] + " " + text
        toks = tok.encode(text)
        it_lens.append(len(toks))

stats(it_lens, "INSTR TUNING")

print()
print("=" * 85)
print("RECOMMENDATION: Truncation impact per seq_len")
print("=" * 85)
for sl in [64, 128, 192, 256, 384, 512, 768, 1024]:
    opus_trunc = sum(1 for l in all_lens if l > sl) / len(all_lens) * 100
    it_trunc = sum(1 for l in it_lens if l > sl) / len(it_lens) * 100
    opus_kept = 100 - opus_trunc
    it_kept = 100 - it_trunc
    print(f"  seq={sl:>4}:  Opus {opus_kept:>5.1f}% kept ({opus_trunc:>5.1f}% truncated)  |  "
          f"IT {it_kept:>5.1f}% kept ({it_trunc:>5.1f}% truncated)")
