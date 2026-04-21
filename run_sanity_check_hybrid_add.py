from sanity_check_signal_tilted import build_arg_parser, run_sanity_check


def main() -> None:
    parser = build_arg_parser("Sanity check (Hybrid Add)")
    args = parser.parse_args()
    run_sanity_check("hybrid_add", args)


if __name__ == "__main__":
    main()
