#!/usr/bin/env python3
"""
Phase 2 Experiment: M2 Robustness Enhanced (gamma=1.0)
"""
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check


def main():
    parser = build_arg_parser("Phase 2: M2 Robustness Enhanced (gamma=1.0)")
    args = parser.parse_args()
    run_sanity_check("m2_robust_gamma10", args)


if __name__ == "__main__":
    main()
