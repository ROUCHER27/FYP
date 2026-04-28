from sanity_check_signal_tilted import build_arg_parser, run_sanity_check


def main() -> None:
    parser = build_arg_parser("Phase 2 sanity check (m2_robust_gamma10)")
    run_sanity_check("m2_robust_gamma10", parser.parse_args())


if __name__ == "__main__":
    main()
