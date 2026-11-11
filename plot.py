"""Render figures from saved numbers only; no model loading or training."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_metrics(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


def save(fig, folder, name, extension):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.{extension}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_run(run, extension, epoch, prediction_path=None):
    metrics = read_metrics(run / "metrics.csv")
    out = run / "figures"
    for prefix, title, name in [("loss_", "Training losses (unweighted components)", "loss_history"),
                                 ("weight_", "Adaptive condition weights", "adaptive_weights")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        if prefix == "loss_":
            ax.plot(metrics["epoch"], metrics["total_loss"], label="weighted total", linewidth=2)
        for field in metrics.dtype.names:
            if field.startswith(prefix):
                ax.plot(metrics["epoch"], metrics[field], label=field[len(prefix):])
        ax.set(xlabel="Epoch", ylabel="Value", title=title)
        ax.set_yscale("log")
        ax.legend()
        save(fig, out, name, extension)
    predictions = sorted((run / "predictions").glob("epoch_*.npz"))
    if epoch is not None:
        predictions = [run / "predictions" / f"epoch_{epoch:07d}.npz"]
    if prediction_path is not None:
        predictions = [prediction_path]
    if not predictions:
        raise FileNotFoundError(f"No numeric predictions in {run / 'predictions'}")
    path = predictions[-1]
    with np.load(path) as data:
        lam = data["lambdas"]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(lam, data["pde_mse"], label="PDE")
        axes[0].plot(lam, data["bc_mse"].sum(axis=1), label="sum BC")
        axes[0].plot(lam, data["nontriviality_loss"], label="slice nontriviality")
        axes[0].plot(lam, data["normalization_loss"], label="slice normalization")
        diagnostic_sum = (data["pde_mse"] + data["bc_mse"].sum(axis=1)
                          + data["nontriviality_loss"] + data["normalization_loss"])
        axes[0].plot(lam, diagnostic_sum, label="slice sum (unweighted)", linestyle="--")
        axes[1].plot(lam, data["relative_pde_mse"], label="PDE / energy")
        axes[1].plot(lam, data["relative_bc_mse"].sum(axis=1), label="sum BC / energy")
        axes[2].plot(lam, data["energy"], label="energy")
        axes[2].plot(lam, data["rms"], label="RMS")
        axes[2].plot(lam, data["peak"], label="peak")
        for ax in axes:
            ax.set_xlabel("lambda")
            ax.set_yscale("log")
            ax.legend()
            for ev in np.unique(data["reference_eigenvalues"]):
                ax.axvline(ev, color="grey", alpha=.3)
        fig.suptitle(f"{path.stem}: fixed-grid diagnostics; slice losses differ from training loss")
        save(fig, out, path.stem + "_spectral_diagnostics", extension)
        spatial = data["spatial"]
        if "reference_field" in data.files:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, key, title in zip(axes, ["raw", "unit_peak", "reference_field"],
                                      ["Raw prediction", "Unit peak (valid slices)", "Reference"]):
                im = ax.pcolormesh(spatial[:, 0], lam, data[key], shading="auto", cmap="RdBu_r")
                ax.set(xlabel="x", ylabel="lambda", title=title)
                ax.set_xlim(spatial[:, 0].min(), spatial[:, 0].max())
                ax.set_ylim(lam.min(), lam.max())
                fig.colorbar(im, ax=ax)
            save(fig, out, path.stem + "_continuous_map", extension)
            # Show a few spectral slices, including both signs of lambda.
            for i in np.linspace(0, len(lam) - 1, 5, dtype=int):
                fig, ax = plt.subplots()
                for key, label in [("raw", "raw"), ("unit_peak", "unit peak"),
                                   ("reference_field", "reference")]:
                    ax.plot(spatial[:, 0], data[key][i], label=label)
                ax.set(xlabel="x", ylabel="u", title=f"lambda={lam[i]:.6g}; valid={data['valid'][i]}")
                ax.legend()
                save(fig, out, path.stem + f"_slice_{i:04d}", extension)
        for ev in np.unique(data["reference_eigenvalues"]):
            i = int(np.argmin(np.abs(lam - ev)))
            selected = np.flatnonzero(np.isclose(data["reference_eigenvalues"], ev))
            title = f"lambda={ev:.6g}; raw peak={data['peak'][i]:.3g}; valid={data['valid'][i]}"
            if spatial.shape[1] == 1:
                fig, ax = plt.subplots()
                ax.plot(spatial[:, 0], data["raw"][i], label="raw prediction")
                ax.plot(spatial[:, 0], data["unit_peak"][i], label="unit peak prediction")
                for j in selected:
                    ax.plot(spatial[:, 0], data["reference_modes"][j], "--", label=f"reference {data['reference_mode_indices'][j]}")
                ax.set(xlabel="x", ylabel="u", title=title + "; sign is arbitrary")
                ax.legend()
            else:
                count = int(round(np.sqrt(len(spatial))))
                fields = [data["raw"][i], data["unit_peak"][i]]
                labels = ["raw prediction", "unit peak prediction"]
                for j in selected:
                    fields.append(data["reference_modes"][j])
                    labels.append(f"reference basis {data['reference_mode_indices'][j]}")
                fig, axes = plt.subplots(1, len(fields), figsize=(4 * len(fields), 4), squeeze=False)
                for ax, field, label in zip(axes[0], fields, labels):
                    im = ax.pcolormesh(spatial[:, 0].reshape(count, count),
                                       spatial[:, 1].reshape(count, count),
                                       field.reshape(count, count), shading="auto", cmap="RdBu_r")
                    ax.set(xlabel="x", ylabel="y", title=label)
                    ax.set_xlim(spatial[:, 0].min(), spatial[:, 0].max())
                    ax.set_ylim(spatial[:, 1].min(), spatial[:, 1].max())
                    fig.colorbar(im, ax=ax)
                fig.suptitle(title + "; degenerate modes may combine basis functions")
            save(fig, out, path.stem + f"_mode_{ev:.6g}", extension)
    print(out)


def aggregate(runs, output, extension):
    """Mean/std on common epochs; save numeric summary alongside its figure."""
    records = [read_metrics(run / "metrics.csv") for run in runs]
    epochs = records[0]["epoch"]
    for record in records[1:]:
        epochs = np.intersect1d(epochs, record["epoch"])
    if len(epochs) == 0:
        raise ValueError("Runs have no common epochs.")
    values = np.stack([record["total_loss"][np.searchsorted(record["epoch"], epochs)] for record in records])
    mean, std = values.mean(axis=0), values.std(axis=0)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "seed_summary.npz", epochs=epochs, total_loss=values,
                        mean=mean, std=std, runs=np.asarray([str(r) for r in runs]))
    fig, ax = plt.subplots()
    for run, row in zip(runs, values):
        ax.plot(epochs, row, alpha=.4, label=str(run.parent.name))
    ax.plot(epochs, mean, color="black", label="mean")
    ax.fill_between(epochs, np.maximum(mean - std, np.finfo(float).tiny), mean + std, color="grey", alpha=.2)
    ax.set(xlabel="Epoch", ylabel="Weighted total loss", yscale="log")
    ax.legend()
    save(fig, output, "seed_summary", extension)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories for the same experiment")
    parser.add_argument("--format", choices=["pdf", "svg", "png"], default="pdf")
    parser.add_argument("--epoch", type=int, help="Default: last saved predictions")
    parser.add_argument("--predictions", type=Path, help="Explicit NPZ, for a single run")
    parser.add_argument("--summary-dir", type=Path, default=Path("figures/seed_summary"))
    args = parser.parse_args()
    if args.predictions and (len(args.runs) != 1 or args.epoch is not None):
        parser.error("--predictions requires one run and cannot be combined with --epoch")
    for run in args.runs:
        plot_run(run, args.format, args.epoch, args.predictions)
    if len(args.runs) > 1:
        aggregate(args.runs, args.summary_dir, args.format)


if __name__ == "__main__":
    main()
