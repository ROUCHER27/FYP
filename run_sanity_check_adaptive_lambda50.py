#!/usr/bin/env python3
"""
Phase 2 Experiment: Adaptive Hybrid (lambda=5.0)
"""
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check


def main():
    parser = build_arg_parser("Phase 2: Adaptive Hybrid (lambda=5.0)")
    args = parser.parse_args()
    run_sanity_check("adaptive_lambda50", args)


if __name__ == "__main__":
    main()
