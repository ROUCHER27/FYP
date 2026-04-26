#!/usr/bin/env python3
"""
Phase 2 Experiment: IMADL + M2 Linear Combination (alpha=0.4)
"""
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check


def main():
    parser = build_arg_parser("Phase 2: IMADL + M2 Linear Combination (alpha=0.4)")
    args = parser.parse_args()
    run_sanity_check("imadl_m2_alpha04", args)


if __name__ == "__main__":
    main()
