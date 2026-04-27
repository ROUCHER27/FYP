#!/usr/bin/env python3
"""
Phase 2 Experiment: IMADL + GMADL Weighted Combination (beta=0.5)
"""
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check


def main():
    parser = build_arg_parser("Phase 2: IMADL + GMADL Weighted Combination (beta=0.5)")
    args = parser.parse_args()
    run_sanity_check("imadl_gmadl_beta05", args)


if __name__ == "__main__":
    main()
