"""Command line: ``mumdia-viewer <run-or-experiment-dir> [--compare <dir>] [--port N]``.

``--fasta PATH`` gives the protein sequences for the coverage views. Without it the
viewer uses the FASTA that the run recorded (a run searched from a FASTA), when that
file is still where it was.

The server binds 127.0.0.1 by default and opens a browser. On a remote server, start it
with ``--no-browser`` and forward the port through SSH:
``ssh -L <port>:127.0.0.1:<port> <user>@<server>``, then open the printed address. The
address carries a random token, so other users of the server cannot reach the viewer
without it.
"""

from __future__ import annotations

import argparse
import secrets
import socket
import sys
import threading
import webbrowser
from pathlib import Path


def _free_port(start: int) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit(f"no free port in {start}-{start + 99}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mumdia-viewer", description=__doc__.splitlines()[0])
    p.add_argument("directory", help="a MuMDIA run directory or experiment directory")
    p.add_argument("--compare", help="a second result set to compare with")
    p.add_argument("--port", type=int, help="port (default: the first free port from 8050)")
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to bind (default 127.0.0.1; other values expose the viewer)",
    )
    p.add_argument("--no-browser", action="store_true", help="do not open a browser")
    p.add_argument(
        "--no-token",
        action="store_true",
        help="serve at / instead of a random path (only on a single-user machine)",
    )
    p.add_argument(
        "--remap",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="where files recorded under OLD are now (inputs outside the run)",
    )
    p.add_argument(
        "--fasta",
        action="append",
        default=[],
        metavar="PATH",
        help="FASTA file(s) with the protein sequences, for sequence coverage "
        "(default: the FASTA the run recorded, if it is found)",
    )
    p.add_argument(
        "--allow-unreleased",
        action="store_true",
        help="accept the ion-mobility schema versions of the unreleased branch",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    from mumdia_viewer.runtime import configure_environment

    configure_environment()  # before numpy and pyarrow are imported
    args = _parser().parse_args(argv)

    from mumdia_viewer.runtime import configure_arrow

    configure_arrow()
    from mumdia_viewer.data import ViewerError, open_results
    from mumdia_viewer.data.fasta import Fasta, FastaSource, find_recorded_fasta
    from mumdia_viewer.ui.app import create_app, create_compare_apps

    remaps = {}
    for item in args.remap:
        if "=" not in item:
            print(f"--remap expects OLD=NEW, got {item!r}", file=sys.stderr)
            return 2
        old, new = item.split("=", 1)
        remaps[old] = new
    try:
        rs = open_results(args.directory, remaps=remaps, allow_unreleased=args.allow_unreleased)
        compare = (
            open_results(args.compare, remaps=remaps, allow_unreleased=args.allow_unreleased)
            if args.compare
            else None
        )
    except ViewerError as exc:
        print(f"mumdia-viewer: {exc}", file=sys.stderr)
        return 1
    sources = [FastaSource(Path(f), "given with --fasta") for f in args.fasta]
    if not sources:
        sources = find_recorded_fasta(rs)
    fasta = None
    if sources:
        try:
            fasta = Fasta.read(sources)
        except ViewerError as exc:
            print(f"mumdia-viewer: {exc}", file=sys.stderr)
            return 1
        print(f"mumdia-viewer: protein sequences from {fasta.label}")
    else:
        print("mumdia-viewer: no FASTA; give one with --fasta to show sequence coverage")
    base = "/" if args.no_token else f"/{secrets.token_urlsafe(9)}/"
    port = args.port or _free_port(8050)
    if compare is not None:
        app, _ = create_compare_apps(rs, compare, url_base=base, fasta=fasta)
    else:
        app = create_app(rs, url_base=base, fasta=fasta)
    url = f"http://127.0.0.1:{port}{base}"
    print(f"mumdia-viewer: serving {rs.root} ({rs.kind}) at {url}")
    if compare is not None:
        print(f"mumdia-viewer: comparing with {compare.root} ({compare.kind}), served at {url}b/")
    if args.host != "127.0.0.1":
        print(
            f"mumdia-viewer: bound to {args.host}; anyone who can reach it and knows the "
            "address can read these results"
        )
    print(f"remote server: ssh -L {port}:127.0.0.1:{port} <user>@<server>, then open {url}")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    app.run(host=args.host, port=port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
