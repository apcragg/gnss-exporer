"""Capture samples to a file."""

import datetime
import pathlib

import adi
import click
import numpy as np

F_DEFAULT_CENTER = 1575.42e6
F_DEFAULT_SAMPLE = 4e6

N_CAPTURE_BUFFER = int(1e6)
T_DEFAULT_CAPTURE = 1.0


@click.command()
@click.argument("addr")
@click.argument("file_path")
@click.option("--fs", type=float, default=F_DEFAULT_SAMPLE)
@click.option("--fc", type=float, default=F_DEFAULT_CENTER)
@click.option("--t-capture", type=float, default=T_DEFAULT_CAPTURE)
@click.option("--timestamp", is_flag=True)
def main(
    addr: str, file_path: pathlib.Path, fs: int, fc: float, t_capture: int, timestamp: bool
) -> None:
    """Capture samples from a PlutoSDR."""
    rx = adi.Pluto(f"ip:{addr}")

    t_start = datetime.datetime.now(tz=datetime.UTC)

    rx.sample_rate = int(fs)
    rx.rx_lo = int(fc)
    rx.rx_buffer_size = N_CAPTURE_BUFFER

    utc_datetime = datetime.datetime.time(t_start)
    utc_str = utc_datetime.isoformat() + "Z"

    file_path = pathlib.Path(file_path)

    if timestamp:
        timestamped_name = file_path.stem + f"_{utc_str}" + file_path.suffix
        file_path = file_path.with_name(timestamped_name)

    n_capture = int(fs * t_capture)
    x_capture = np.zeros(n_capture, dtype=np.complex64)

    n_buf = 0
    while n_buf < n_capture:
        n_to_capture = min(rx.rx_buffer_size, n_capture - n_buf)

        x = np.array(rx.rx())
        x_capture[n_buf : n_buf + n_to_capture] = x[:n_to_capture]
        n_buf += n_to_capture

    # scale by ADC bits
    x /= (2**15) - 1

    with file_path.open("wb") as f:
        np.save(f, x_capture)


if __name__ == "__main__":
    main()
