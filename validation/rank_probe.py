"""Show what the five selected residues look like under a substantive ordering.

The shipped hit list assigns every selected residue the same score, because the
subset objective gives each member of the optimal subset the same marginal
utility. The order in the file is therefore matrix-index order. This probe
recomputes the propagation score the pipeline itself ranks cavities with, and
prints the ordering it would induce, without writing anything.
"""

from validation.context import build, selected_five

for target in ("kras", "bcr", "myosin", "cmyc"):
    ctx = build(target)
    keys, hop = ctx["keys"], ctx["hop"]
    idx = {k: i for i, k in enumerate(keys)}
    shipped = selected_five(target)
    scored = [(k, float(hop[idx[k]])) for k in shipped]
    order = sorted(scored, key=lambda x: -x[1])
    print("%-7s" % target)
    print("   shipped  : " + "  ".join(k for k, _ in scored))
    print("   by hop   : " + "  ".join("%s(%.4g)" % (k, v) for k, v in order))
    print("   순서 동일 %-5s | 전파점수 전부 동일 %s"
          % ([k for k, _ in scored] == [k for k, _ in order],
             len({round(v, 12) for _, v in scored}) == 1))
