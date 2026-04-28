from sanity_check_signal_tilted import build_arg_parser, run_sanity_check


def main() -> None:
    parser = build_arg_parser("Phase 2 sanity check (adaptive_lambda100)")
    run_sanity_check("adaptive_lambda100", parser.parse_args())


if __name__ == "__main__":
    main()
