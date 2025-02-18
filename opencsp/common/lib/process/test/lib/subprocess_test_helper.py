import sys
import time

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog=__file__.rstrip(".py"), description="Tests the subprocess tools.")
    parser.add_argument("--simple_stdout", action="store_true", help="Outputs 'Hello\nworld!' on stdout.")
    parser.add_argument("--simple_stderr", action="store_true", help="Outputs 'Goodbye\nworld!' on stderr.")
    parser.add_argument(
        "--mixed_stdout_stderr",
        action="store_true",
        help="Outputs 'foo', 'bar', and 'baz' on stdout, stderr, and stdout, respectively.",
    )
    parser.add_argument("--retcode", type=int, help="Causes this program to exit with the given retcode.")
    parser.add_argument("--delay_before", type=float, help="Sleeps for N seconds before outputing any values.")
    parser.add_argument("--delay_after", type=float, help="Sleeps for N seconds after outputing any values.")
    args = parser.parse_args()

    if args.delay_before != None:
        time.sleep(args.delay_before)
    if args.simple_stdout:
        print("Hello\nworld!", flush=True)
        time.sleep(0.1)
    if args.simple_stderr:
        print("Goodbye\nworld!", file=sys.stderr, flush=True)
        time.sleep(0.1)
    if args.mixed_stdout_stderr:
        print("foo", flush=True)
        time.sleep(0.1)
        print("bar", file=sys.stderr, flush=True)
        time.sleep(0.1)
        print("baz", flush=True)
        time.sleep(0.1)
    if args.delay_after != None:
        time.sleep(args.delay_after)
    if args.retcode != None:
        exit(args.retcode)
