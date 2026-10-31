"""Results table and plots from everything under results/runs/.

    python -m scripts.report                      # results/results.md + results/plots/*.png
    python -m scripts.report --readme             # also paste the table into README.md
    python -m scripts.report --results results/smoke

Accuracy is the mean of the last five evaluations of each seed (non-IID
curves jump around too much for a single last value to mean anything), then
mean ± std over seeds. "Rounds to target" is the first round at which a
seed's global model reaches the target, averaged over the seeds that got there.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from fl.config import fmt  # noqa: E402
from fl.data import CLASSES  # noqa: E402
from fl.metrics import read_jsonl  # noqa: E402

# ---------------------------------------------------------------- palette (validated, light surface)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]       # categorical, fixed order
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]                 # ordinal (light -> dark)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK2,
})


# ---------------------------------------------------------------- loading

def load_runs(results: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for summary_path in sorted(results.glob("runs/*/seed*/summary.json")):
        folder = summary_path.parent
        summary = json.loads(summary_path.read_text())
        if not summary.get("ok"):
            continue
        spec = json.loads((folder / "config.json").read_text())
        history = [h for h in read_jsonl(folder / "metrics.jsonl") if h.get("test_acc") is not None]
        run = runs.setdefault(summary["run_name"], {"name": summary["run_name"], "spec": spec, "seeds": []})
        run["seeds"].append({"seed": spec["seed"], "summary": summary, "history": history})
    return runs


def final_accs(run) -> np.ndarray:
    return np.array([s["summary"]["final_acc_last5"] for s in run["seeds"]])


def rounds_to(run, target: float) -> list[int | None]:
    return [next((h["round"] for h in s["history"] if h["test_acc"] >= target), None) for s in run["seeds"]]


def curve(run) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and std of test accuracy per evaluated round, over the seeds."""
    by_round = defaultdict(list)
    for s in run["seeds"]:
        for h in s["history"]:
            by_round[h["round"]].append(h["test_acc"])
    rounds = np.array(sorted(r for r, v in by_round.items() if len(v) == len(run["seeds"])))
    vals = [by_round[r] for r in rounds]
    return rounds, np.array([np.mean(v) for v in vals]), np.array([np.std(v) for v in vals])


# ---------------------------------------------------------------- labels & ordering

def label(spec) -> str:
    a = spec["algorithm"]
    name = {"centralized": "Centralized", "local": "Local only", "fedavg": "FedAvg"}.get(a)
    if a == "fedprox":
        name = f"FedProx (μ = {fmt(spec['mu'])})"
    if spec.get("dropout", 0) > 0:
        name += f" + {spec['dropout']:.0%} dropout"
    return name


def split_label(spec) -> str:
    if spec["algorithm"] == "centralized":
        return "pooled (all data)"
    a = spec["alpha"]
    return f"IID (α = {fmt(a)})" if a >= 100 else f"Dirichlet α = {fmt(a)}"


def clients_label(spec) -> str:
    if spec["algorithm"] == "centralized":
        return "1"
    n, p = spec["num_clients"], spec["participation"]
    return f"{n}" if p >= 1 else f"{n} ({p:.0%} per round)"


def sort_key(run, n_main: int):
    s = run["spec"]
    a = s["algorithm"]
    main = s.get("num_clients") == n_main and s.get("participation", 1) == 1
    if a == "centralized":
        section = 0
    elif a == "local":
        section = 1
    elif s.get("dropout", 0) > 0:
        section = 5
    elif not main:
        section = 4
    else:
        section = 2 if a == "fedavg" else 3
    return (section, -s.get("alpha", 0), s.get("num_clients", 0), s.get("participation", 1), s.get("mu", 0))


def fmt_acc(x: np.ndarray) -> str:
    return f"{100 * x.mean():.1f} ± {100 * x.std():.1f}" if len(x) > 1 else f"{100 * x.mean():.1f}"


def fmt_rounds(rs: list[int | None]) -> str:
    hit = [r for r in rs if r is not None]
    if not hit:
        return "not reached"
    txt = f"{np.mean(hit):.0f}"
    return txt if len(hit) == len(rs) else f"{txt} ({len(hit)}/{len(rs)} seeds)"


# ---------------------------------------------------------------- table & headline numbers

def table(runs, target, n_main) -> str:
    rows = sorted(runs.values(), key=lambda r: sort_key(r, n_main))
    lines = ["| Run | Data split | Clients | Final test accuracy (%) | Rounds to "
             f"{target:.0%} | Seeds |", "|---|---|---|---|---|---|"]
    for r in rows:
        s = r["spec"]
        rt = fmt_rounds(rounds_to(r, target))
        unit = (" epoch" if rt == "1" else " epochs") if s["algorithm"] in ("centralized", "local") else ""
        lines.append(f"| {label(s)} | {split_label(s)} | {clients_label(s)} | {fmt_acc(final_accs(r))} | "
                     f"{rt}{unit if rt != 'not reached' else ''} | {len(r['seeds'])} |")
    return "\n".join(lines)


def find(runs, **want):
    out = []
    for r in runs.values():
        s = r["spec"]
        if all(abs(s.get(k, 0) - v) < 1e-9 if isinstance(v, float) else s.get(k) == v for k, v in want.items()):
            out.append(r)
    return out


def headline(runs, n_main) -> str:
    """The numbers the CV bullets ask for, computed rather than copied by hand."""
    out = []
    base = dict(num_clients=n_main, participation=1.0, dropout=0.0)
    fedavg = {r["spec"]["alpha"]: r for r in find(runs, algorithm="fedavg", **base)}
    cent = find(runs, algorithm="centralized")
    if cent:
        out.append(f"- Centralized upper bound: {100 * final_accs(cent[0]).mean():.1f}%")
    if fedavg:
        a_iid, a_skew = max(fedavg), min(fedavg)
        iid, skew = final_accs(fedavg[a_iid]).mean(), final_accs(fedavg[a_skew]).mean()
        if a_iid != a_skew:
            out.append(f"- FedAvg, {n_main} clients: {100 * iid:.1f}% at α = {fmt(a_iid)} → "
                       f"{100 * skew:.1f}% at α = {fmt(a_skew)} ({100 * (skew - iid):+.1f} points)")
        else:
            out.append(f"- FedAvg, {n_main} clients, α = {fmt(a_skew)}: {100 * skew:.1f}%")
        prox = find(runs, algorithm="fedprox", alpha=a_skew, **base)
        if prox:
            best = max(prox, key=lambda r: final_accs(r).mean())
            gain = final_accs(best).mean() - skew
            line = (f"- Best FedProx at α = {fmt(a_skew)}: μ = {fmt(best['spec']['mu'])}, "
                    f"{100 * final_accs(best).mean():.1f}% ({100 * gain:+.1f} points vs FedAvg")
            if iid > skew:
                line += f"; recovers {gain / (iid - skew):.0%} of the gap to α = {fmt(a_iid)}"
            out.append(line + ")")
        local = find(runs, algorithm="local", alpha=a_skew, num_clients=n_main)
        if local:
            out.append(f"- Local-only lower bound at α = {fmt(a_skew)}: {100 * final_accs(local[0]).mean():.1f}%")
    for r in [r for r in runs.values() if r["spec"].get("dropout", 0) > 0]:
        s = r["spec"]
        twin = find(runs, algorithm=s["algorithm"], alpha=s["alpha"], mu=s["mu"], num_clients=s["num_clients"],
                    participation=s["participation"], dropout=0.0)
        if twin:
            d, t = final_accs(r).mean(), final_accs(twin[0]).mean()
            out.append(f"- {label(s)} at α = {fmt(s['alpha'])}: {100 * d:.1f}% vs {100 * t:.1f}% without dropout "
                       f"(keeps {d / t:.0%} of its accuracy)")
    return "\n".join(o for o in out if o)


# ---------------------------------------------------------------- plots

SMOOTH = 5   # rounds in the centred moving average applied to long curves (display only)


def smooth(x: np.ndarray, w: int = SMOOTH) -> np.ndarray:
    """Centred moving average with shorter windows at the edges."""
    if len(x) <= 30:
        return x
    h = w // 2
    return np.array([x[max(0, i - h):i + h + 1].mean() for i in range(len(x))])


def plot_curves(ax, entries, end_labels=True):
    """entries: [(run, legend label, color, linestyle)]. Smoothed mean line + ±1 std wash."""
    ends = []
    for run, lab, color, ls in entries:
        rounds, mean, std = curve(run)
        if len(rounds) == 0:
            continue
        mean, std = smooth(mean), smooth(std)
        ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.10, linewidth=0)
        ax.plot(rounds, mean, color=color, linestyle=ls, label=lab)
        ax.plot(rounds[-1], mean[-1], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
        ends.append((mean[-1], f"{100 * mean[-1]:.1f}%", rounds[-1]))
    # Direct end labels only when they cannot collide (<= 4 lines, ends well apart).
    ys = sorted(e[0] for e in ends)
    if end_labels and len(ends) <= 4 and all(b - a > 0.035 for a, b in zip(ys, ys[1:])):
        for y, txt, x in ends:
            ax.annotate(txt, (x, y), xytext=(8, 0), textcoords="offset points", va="center",
                        color=INK, fontsize=9, fontweight="bold")
    ax.set_xlabel("round")
    ax.set_ylabel("test accuracy")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.margins(x=0.02)


def ref_line(ax, value, text, side="left"):
    """Dashed horizontal reference (e.g. the centralized upper bound), labelled in place."""
    ax.axhline(value, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
    x, ha = (0.01, "left") if side == "left" else (0.99, "right")
    ax.annotate(f"{text} {100 * value:.1f}%", (x, value), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", color=INK2, fontsize=8.5, ha=ha)


def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def plot_noniid(runs, n_main, out):
    fed = sorted(find(runs, algorithm="fedavg", num_clients=n_main, participation=1.0, dropout=0.0),
                 key=lambda r: -r["spec"]["alpha"])
    if not fed:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    plot_curves(ax, [(r, split_label(r["spec"]), SERIES[i], "-") for i, r in enumerate(fed[:5])])
    cent = find(runs, algorithm="centralized")
    if cent:
        ref_line(ax, final_accs(cent[0]).mean(), "centralized")
    local = find(runs, algorithm="local", alpha=min(r["spec"]["alpha"] for r in fed), num_clients=n_main)
    if local:
        ref_line(ax, final_accs(local[0]).mean(), f"local only, α = {fmt(local[0]['spec']['alpha'])}")
    ax.set_title(f"FedAvg with {n_main} clients: accuracy falls as client data gets more skewed")
    ax.legend(loc="lower right")
    save(fig, out / "noniid_fedavg.png")


def plot_fedprox(runs, n_main, out):
    prox = find(runs, algorithm="fedprox", num_clients=n_main, participation=1.0, dropout=0.0)
    if not prox:
        return
    alpha = min(r["spec"]["alpha"] for r in prox)
    prox = sorted([r for r in prox if r["spec"]["alpha"] == alpha], key=lambda r: r["spec"]["mu"])
    fedavg = find(runs, algorithm="fedavg", alpha=alpha, num_clients=n_main, participation=1.0, dropout=0.0)
    # mu is ordered, so its lines share one hue from light (small mu) to dark (large mu).
    ramp = BLUE_RAMP[-len(prox):] if len(prox) <= len(BLUE_RAMP) else plt.cm.Blues(np.linspace(0.35, 1, len(prox)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    entries = [(fedavg[0], "FedAvg (μ = 0)", SERIES[1], "-")] if fedavg else []
    entries += [(r, f"FedProx μ = {fmt(r['spec']['mu'])}", ramp[i], "-") for i, r in enumerate(prox)]
    plot_curves(a1, entries)
    a1.set_title(f"FedProx vs FedAvg at α = {fmt(alpha)}")
    a1.legend(loc="lower right")

    mus = [r["spec"]["mu"] for r in prox]
    means = [final_accs(r).mean() for r in prox]
    stds = [final_accs(r).std() for r in prox]
    a2.errorbar(mus, means, yerr=stds, color=SERIES[0], marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, capsize=3, elinewidth=1)
    if fedavg:
        ref_line(a2, final_accs(fedavg[0]).mean(), "FedAvg (μ = 0)", side="right")
    a2.set_xscale("log")
    a2.set_xlabel("μ (proximal coefficient)")
    a2.set_ylabel("accuracy, mean of last 5 rounds")
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a2.set_title("Final accuracy by μ (± std over seeds)")
    save(fig, out / "fedprox_mu.png")


def plot_scaling(runs, n_main, out):
    fed = [r for r in find(runs, algorithm="fedavg", dropout=0.0)]
    by_alpha = defaultdict(set)
    for r in fed:
        by_alpha[r["spec"]["alpha"]].add((r["spec"]["num_clients"], r["spec"]["participation"]))
    if not by_alpha:
        return
    alpha = max(by_alpha, key=lambda a: len(by_alpha[a]))
    if len(by_alpha[alpha]) < 3:
        return
    parts = sorted({p for _, p in by_alpha[alpha]})
    ns = sorted({n for n, _ in by_alpha[alpha]})
    fig, axes = plt.subplots(1, len(parts), figsize=(5.2 * len(parts), 4), sharey=True, squeeze=False)
    for ax, p in zip(axes[0], parts):
        entries = []
        for i, n in enumerate(ns):
            r = find(runs, algorithm="fedavg", dropout=0.0, alpha=alpha, num_clients=n, participation=p)
            if r:
                entries.append((r[0], f"{n} clients", SERIES[i % len(SERIES)], "-"))
        plot_curves(ax, entries)
        ax.set_title(f"{p:.0%} of clients per round")
        ax.legend(loc="lower right")
    fig.suptitle(f"FedAvg at α = {fmt(alpha)}: number of clients and participation", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    save(fig, out / "scaling.png")


def plot_dropout(runs, out):
    dropped = [r for r in runs.values() if r["spec"].get("dropout", 0) > 0]
    if not dropped:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    entries = []
    for i, r in enumerate(sorted(dropped, key=lambda r: (r["spec"]["algorithm"], r["spec"]["mu"]))):
        s = r["spec"]
        color = SERIES[i % len(SERIES)]
        twin = find(runs, algorithm=s["algorithm"], alpha=s["alpha"], mu=s["mu"], num_clients=s["num_clients"],
                    participation=s["participation"], dropout=0.0)
        if twin:
            entries.append((twin[0], label(twin[0]["spec"]), color, "-"))
        entries.append((r, label(s), color, (0, (5, 3))))
    plot_curves(ax, entries)
    ax.set_title(f"Client dropout at α = {fmt(dropped[0]['spec']['alpha'])} (dashed = with dropout)")
    ax.legend(loc="lower right")
    save(fig, out / "dropout.png")


def plot_partitions(runs, data_dir: Path, n_main, out):
    """Heatmap of samples per (client, class) for each alpha, seed 0."""
    dataset = next(iter(runs.values()))["spec"]["dataset"] if runs else "cifar10"
    files = []
    for f in sorted(data_dir.glob(f"shards/{dataset}_n{n_main}_a*_s0/partition.json")):
        info = json.loads(f.read_text())
        files.append((info["alpha"], np.array(info["counts"])))
    if not files:
        return
    files.sort(key=lambda t: -t[0])
    cmap = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, axes = plt.subplots(1, len(files), figsize=(4.2 * len(files), 4.2), squeeze=False)
    vmax = max(c.max() for _, c in files)
    for ax, (alpha, counts) in zip(axes[0], files):
        im = ax.imshow(counts, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(split_label({"algorithm": "x", "alpha": alpha}))
        ax.set_xticks(range(counts.shape[1]))
        names = CLASSES if counts.shape[1] == len(CLASSES) and dataset == "cifar10" else range(counts.shape[1])
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(counts.shape[0]))
        ax.set_ylabel("client")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="training images")
    fig.suptitle("What the Dirichlet partition gives each client", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.savefig(out / "partitions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / 'partitions.png'}")


# ---------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description="Build the results table and plots")
    p.add_argument("--results", default="results")
    p.add_argument("--data", default="data")
    p.add_argument("--target", type=float, help="target accuracy (default: target_acc from the configs)")
    p.add_argument("--readme", action="store_true", help="paste the table into README.md between the markers")
    args = p.parse_args()
    results = Path(args.results)
    runs = load_runs(results)
    if not runs:
        raise SystemExit(f"no finished runs under {results}/runs")
    target = args.target or Counter(r["spec"]["target_acc"] for r in runs.values()).most_common(1)[0][0]
    fed_ns = [r["spec"]["num_clients"] for r in runs.values() if r["spec"]["algorithm"] in ("fedavg", "fedprox")]
    n_main = Counter(fed_ns).most_common(1)[0][0] if fed_ns else 10

    plots = results / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_partitions(runs, Path(args.data), n_main, plots)
    plot_noniid(runs, n_main, plots)
    plot_fedprox(runs, n_main, plots)
    plot_scaling(runs, n_main, plots)
    plot_dropout(runs, plots)

    md = (f"## Results\n\nMean ± std over seeds of the test accuracy averaged over each run's last five "
          f"evaluations. Centralized and local-only rows count epochs instead of rounds.\n\n"
          f"{table(runs, target, n_main)}\n\n### Headline numbers\n\n{headline(runs, n_main)}\n")
    (results / "results.md").write_text(md, encoding="utf-8")
    print(f"wrote {results / 'results.md'}")
    try:
        print(md)
    except UnicodeEncodeError:          # e.g. a Windows console without UTF-8
        print(md.encode("ascii", "replace").decode())

    if args.readme:
        readme = Path("README.md")
        text = readme.read_text(encoding="utf-8")
        start, end = "<!-- results:start -->", "<!-- results:end -->"
        if start not in text:
            raise SystemExit("README.md has no results markers")
        body = f"{start}\n{table(runs, target, n_main)}\n\n{headline(runs, n_main)}\n{end}"
        readme.write_text(re.sub(f"{start}.*?{end}", lambda _: body, text, flags=re.S), encoding="utf-8")
        print("updated README.md")


if __name__ == "__main__":
    main()
