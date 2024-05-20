import argparse
import os
import subprocess
import sys
from multiprocessing import Pool

def build(library, args):
    print("Building %s..." % library)
    if args is not None and len(args) != 0:
        q = " ".join(["--build-arg " + x.replace(" ", "\\ ") for x in args])
    else:
        q = ""

    try:
        subprocess.check_call(
            "docker build %s --rm -t dbscan-benchmarks-%s -f" " benchmark/algorithms/%s/Dockerfile  ." % (q, library.lower(), library),
            shell=True,
        )
        return {library: "success"}
    except subprocess.CalledProcessError:
        return {library: "fail"}


def build_multiprocess(args):
    return build(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--proc", default=1, type=int, help="the number of process to build docker images")
    parser.add_argument("--algorithm", metavar="NAME", help="build only the named algorithm image", default=None)
    parser.add_argument("--build-arg", help="pass given args to all docker builds", nargs="+")
    args = parser.parse_args()

    print("Building base image...")
    subprocess.check_call(
         "docker build \
         --rm -t dbscan-benchmark -f benchmark/algorithms/base/Dockerfile .",
         shell=True,
     )

    if args.algorithm:
        tags = [args.algorithm]
    elif os.getenv("LIBRARY"):
        tags = [os.getenv("LIBRARY")]
    else:
        tags = [fn.split(".")[-1] for fn in os.listdir("benchmark/algorithms")]

    print("Building algorithm images... with (%d) processes" % args.proc)

    if args.proc == 1:
        install_status = [build(tag, args.build_arg) for tag in tags]
    else:
        pool = Pool(processes=args.proc)
        install_status = pool.map(build_multiprocess, [(tag, args.build_arg) for tag in tags])
        pool.close()
        pool.join()

    print("\n\nInstall Status:\n" + "\n".join(str(algo) for algo in install_status))

    # Exit 1 if any of the installations fail.
    for x in install_status:
        for (k, v) in x.items():
            if v == "fail":
                sys.exit(1)
