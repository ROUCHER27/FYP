#!/usr/bin/env python3
"""
Phase 2 Experiment: M2 Robustness Enhanced (gamma=0.1)
"""
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check


def main():
    parser = build_arg_parser("Phase 2: M2 Robustness Enhanced (gamma=0.1)")
    args = parser.parse_args()
    run_sanity_check("m2_robust_gamma01", args)


if __name__ == "__main__":
    main()
