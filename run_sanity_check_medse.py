from sanity_check_core import build_arg_parser, run_sanity_check


def main() -> None:
    parser = build_arg_parser("Sanity check (MedSE)")
    args = parser.parse_args()
    run_sanity_check("medse", args)


if __name__ == "__main__":
    main()
