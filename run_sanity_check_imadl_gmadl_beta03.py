from sanity_check_signal_tilted import build_arg_parser, run_sanity_check


def main() -> None:
    parser = build_arg_parser("Phase 2 sanity check (imadl_gmadl_beta03)")
    run_sanity_check("imadl_gmadl_beta03", parser.parse_args())


if __name__ == "__main__":
    main()
