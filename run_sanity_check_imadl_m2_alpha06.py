from sanity_check_signal_tilted import build_arg_parser, run_sanity_check


def main() -> None:
    parser = build_arg_parser("Phase 2 sanity check (imadl_m2_alpha06)")
    run_sanity_check("imadl_m2_alpha06", parser.parse_args())


if __name__ == "__main__":
    main()
