"""Measure Allan deviation of a tone received by PlutoSDR.

Method (frequency-discriminator approach; avoids phase unwrapping):
- Tune Pluto near an external synthesizer tone (EraSynth Micro).
- Collect IQ samples in blocks of duration tau0 = N / fs.
- For each block, estimate average instantaneous frequency using a 1-lag phase
  discriminator:
      f_hat = angle(sum(x[n]*conj(x[n-1]))) * fs / (2*pi)
- Subtract nominal digital IF (f_if) to get frequency error f_err.
- Convert to fractional frequency y[n] = f_err / f0.
- Compute overlapping Allan deviation using allantools.oadev on y[n].

Notes:
- Ensure the received tone is actually located near +f_if in the sampled spectrum
  (by offsetting LO or the source frequency).
- For best results: strong tone (high SNR), avoid clipping, stable gain, long capture.
"""

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import allantools

try:
    import adi  # pyadi-iio
except ImportError as e:
    raise SystemExit("Missing pyadi-iio. Install with: pip install pyadi-iio") from e


@dataclass
class Config:
    uri: str
    f0_hz: float
    fs_hz: float
    rf_bw_hz: float
    rx_gain_db: float
    block_len: int
    duration_s: float
    f_if_hz: float
    skip_blocks: int
    taus: str
    out_prefix: str


def estimate_block_freq_hz(x, fs_hz, lag=10):
    x = x.astype(np.complex128, copy=False)
    s = np.sum(x[lag:] * np.conj(x[:-lag]))
    return float(np.angle(s) * fs_hz / (2 * np.pi * lag))


def parse_taus(taus_str: str, tau0: float, max_duration: float) -> np.ndarray:
    """
    taus_str:
      - "auto" -> log-spaced taus between tau0 and ~duration/5
      - "1,2,5,10" -> multiples of tau0 (integers)
      - "0.1,0.2,0.5" -> seconds (floats)
    """
    if taus_str.strip().lower() == "auto":
        tmin = tau0
        tmax = max(tau0, max_duration / 5.0)
        if tmax <= tmin:
            return np.array([tmin])
        return np.unique(np.logspace(np.log10(tmin), np.log10(tmax), 30))

    parts = [p.strip() for p in taus_str.split(",") if p.strip()]
    vals = np.array([float(p) for p in parts], dtype=float)

    # If all are integers, interpret as multiples of tau0
    if np.all(np.abs(vals - np.round(vals)) < 1e-12):
        return np.unique(vals * tau0)

    # Otherwise interpret as seconds directly
    return np.unique(vals)


def setup_pluto(cfg: Config):
    sdr = adi.Pluto(uri=cfg.uri)
    # Explicitly set single channel for Pluto
    try:
        sdr.rx_enabled_channels = [0]
    except Exception:
        pass

    sdr.sample_rate = int(cfg.fs_hz)
    sdr.rx_rf_bandwidth = int(cfg.rf_bw_hz)
    sdr.rx_lo = int(cfg.f0_hz)

    # Gain control: manual for repeatability
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(cfg.rx_gain_db)

    # Buffer size = block length (one Allan sample per rx() call)
    sdr.rx_buffer_size = int(cfg.block_len)
    return sdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--uri", default="ip:pluto.local", help="IIO URI (e.g., ip:pluto.local or usb:1.2.5)"
    )
    ap.add_argument(
        "--f0", type=float, required=True, help="Tone RF frequency in Hz (e.g., 1.57542e9)"
    )
    ap.add_argument("--fs", type=float, default=2e6, help="Sample rate in Hz")
    ap.add_argument("--rf-bw", type=float, default=1.5e6, help="RF analog bandwidth in Hz")
    ap.add_argument("--rx-gain", type=float, default=30.0, help="RX gain in dB (manual)")
    ap.add_argument(
        "--block-len", type=int, default=200000, help="Samples per block (tau0 = block_len/fs)"
    )
    ap.add_argument("--duration", type=float, default=120.0, help="Capture duration in seconds")
    ap.add_argument(
        "--f-if", type=float, default=100e3, help="Nominal digital IF in Hz (used for f_err)"
    )
    ap.add_argument("--skip-blocks", type=int, default=5, help="Discard initial blocks (settling)")
    ap.add_argument(
        "--taus", default="auto", help='Allan taus: "auto" or comma list (multiples or seconds)'
    )
    ap.add_argument("--out-prefix", default="pluto_adev", help="Output file prefix")
    ap.add_argument(
        "--demean",
        action="store_true",
        help="Subtract mean(y) before Allan deviation (optional; can help if IF offset isn't exact).",
    )
    args = ap.parse_args()

    cfg = Config(
        uri=args.uri,
        f0_hz=args.f0,
        fs_hz=args.fs,
        rf_bw_hz=args.rf_bw,
        rx_gain_db=args.rx_gain,
        block_len=args.block_len,
        duration_s=args.duration,
        f_if_hz=args.f_if,
        skip_blocks=args.skip_blocks,
        taus=args.taus,
        out_prefix=args.out_prefix,
    )

    tau0 = cfg.block_len / cfg.fs_hz
    est_rate = 1.0 / tau0  # y samples per second
    n_blocks = int(np.floor(cfg.duration_s / tau0))
    if n_blocks < (cfg.skip_blocks + 3):
        raise SystemExit(f"Duration too short. Need at least ~{(cfg.skip_blocks + 3) * tau0:.2f}s")

    sdr = setup_pluto(cfg)
    print(f"tau0 = {tau0:.6f} s, blocks = {n_blocks}, est_rate = {est_rate:.3f} Hz")
    print(
        f"Tuning rx_lo={cfg.f0_hz:.3f} Hz, fs={cfg.fs_hz:.3f} Hz, "
        f"rf_bw={cfg.rf_bw_hz:.3f} Hz, f_if={cfg.f_if_hz:.3f} Hz"
    )

    y_list: list[float] = []
    f_err_list: list[float] = []
    t_stamps: list[float] = []

    # Warm up / flush
    for _ in range(cfg.skip_blocks):
        _ = sdr.rx()

    t0 = time.time()
    for _ in range(n_blocks - cfg.skip_blocks):
        x = sdr.rx()
        if x is None or len(x) != cfg.block_len:
            raise RuntimeError("RX returned unexpected block size; check connection/buffer_size.")

        # Frequency estimate relative to DC
        f_hat = estimate_block_freq_hz(x, cfg.fs_hz)

        # Error relative to expected IF
        f_err = f_hat - cfg.f_if_hz
        y = f_err / cfg.f0_hz

        f_err_list.append(f_err)
        y_list.append(y)
        t_stamps.append(time.time() - t0)

    y_arr = np.asarray(y_list, dtype=np.float64)
    f_err_arr = np.asarray(f_err_list, dtype=np.float64)
    t_arr = np.asarray(t_stamps, dtype=np.float64)

    if args.demean:
        y_arr = y_arr - np.mean(y_arr)

    # Allan deviation
    taus = parse_taus(cfg.taus, tau0, cfg.duration_s)

    # Keep only feasible taus: >= tau0 and <= about half the record length
    max_tau = (len(y_arr) / est_rate) / 2.0
    taus = taus[(taus >= tau0) & (taus <= max_tau)]
    if taus.size == 0:
        taus = np.array([tau0])

    tau_out, adev, adev_err, ns = allantools.oadev(
        y_arr,
        rate=est_rate,
        data_type="freq",  # y_arr is fractional frequency
        taus=taus,
    )

    # Save results
    out = {
        "config": cfg.__dict__,
        "tau0_s": tau0,
        "est_rate_hz": est_rate,
        "tau_s": tau_out.tolist(),
        "adev": adev.tolist(),
        "adev_err": adev_err.tolist(),
        "n": ns.tolist(),
        # Helpful diagnostics
        "f_err_mean_hz": float(np.mean(f_err_arr)),
        "f_err_std_hz": float(np.std(f_err_arr)),
        "y_mean": float(np.mean(y_arr)),
        "y_std": float(np.std(y_arr)),
    }
    json_path = f"{cfg.out_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {json_path}")

    # Plot ADEV
    plt.figure()
    plt.loglog(tau_out, adev, marker="o")
    plt.xlabel("Averaging time τ (s)")
    plt.ylabel("Allan deviation σ_y(τ)")
    plt.title("Allan Deviation from Pluto RX (fractional frequency)")
    plt.grid(True, which="both")
    plt.savefig(f"{cfg.out_prefix}.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {cfg.out_prefix}.png")

    # Plot per-block frequency error
    plt.figure()
    plt.plot(t_arr, f_err_arr)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency error (Hz)")
    plt.title("Per-block frequency error estimate (1-lag discriminator)")
    plt.grid(True)
    ferr_path = f"{cfg.out_prefix}_ferr.png"
    plt.savefig(ferr_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {ferr_path}")


if __name__ == "__main__":
    main()
