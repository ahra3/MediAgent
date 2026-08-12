"""Quick script to compute aggregate evaluation statistics."""
import json

with open("eval/results/evaluation_results.json", "r") as f:
    data = json.load(f)

ok = [r for r in data if "error" not in r]
n = len(ok)
print(f"Total cases: {n}")

for m in ["med_recall", "med_precision", "cond_recall", "ddi_recall",
          "guideline_topic_recall", "citation_coverage"]:
    values = [r.get(m, 0) for r in ok]
    avg = sum(values) / len(values)
    print(f"  {m}: {avg:.3f} ({avg:.1%})")

# DDI-specific
ddi_c = [r for r in ok if r["ddi_expected"] > 0]
print(f"\nDDI cases with expected interactions: {len(ddi_c)}")
if ddi_c:
    dr = sum(r["ddi_recall"] for r in ddi_c) / len(ddi_c)
    print(f"  DDI recall (only those): {dr:.3f} ({dr:.1%})")
    matched = sum(1 for r in ddi_c if r["ddi_recall"] > 0)
    print(f"  Cases where at least 1 expected DDI found: {matched}/{len(ddi_c)}")

# Citation-specific
cite_c = [r for r in ok if r.get("total_recommendations", 0) > 0]
print(f"\nCases with recommendations: {len(cite_c)}")
if cite_c:
    cc = sum(r["citation_coverage"] for r in cite_c) / len(cite_c)
    print(f"  Citation coverage (only those): {cc:.3f} ({cc:.1%})")

# Latency
times = [r["timings"]["total"] for r in ok]
ts = sorted(times)
print(f"\nLatency:")
print(f"  Mean:   {sum(times)/n:.1f}s")
print(f"  Median: {ts[n//2]:.1f}s")
print(f"  Min:    {min(times):.1f}s")
print(f"  Max:    {max(times):.1f}s")
print(f"  P95:    {ts[int(n*0.95)]:.1f}s")

# Risk level
risk_match = sum(1 for r in ok if r.get("risk_level_match"))
print(f"\nRisk level match: {risk_match}/{n} ({risk_match/n:.1%})")

# Edge cases
zero_recs = sum(1 for r in ok if r.get("total_recommendations", 0) == 0)
zero_med = sum(1 for r in ok if r["med_recall"] == 0)
full_med = sum(1 for r in ok if r["med_recall"] == 1.0)
print(f"Cases with 0 recommendations: {zero_recs}/{n}")
print(f"Med recall=0: {zero_med}/{n}")
print(f"Med recall=1: {full_med}/{n}")
