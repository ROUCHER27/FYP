#!/usr/bin/env python3
"""
Generate 13 individual runner scripts for Phase 2 loss functions.
"""
from pathlib import Path

# Phase 2 loss functions with descriptions
PHASE2_LOSSES = [
    # Variant 1: IMADL + M2 Linear (7)
    ("imadl_m2_alpha02", "IMADL + M2 Linear Combination (alpha=0.2)"),
    ("imadl_m2_alpha03", "IMADL + M2 Linear Combination (alpha=0.3)"),
    ("imadl_m2_alpha04", "IMADL + M2 Linear Combination (alpha=0.4)"),
    ("imadl_m2_alpha05", "IMADL + M2 Linear Combination (alpha=0.5)"),
    ("imadl_m2_alpha06", "IMADL + M2 Linear Combination (alpha=0.6)"),
    ("imadl_m2_alpha07", "IMADL + M2 Linear Combination (alpha=0.7)"),
    ("imadl_m2_alpha08", "IMADL + M2 Linear Combination (alpha=0.8)"),
    # Variant 2: IMADL + GMADL Weighted (3)
    ("imadl_gmadl_beta03", "IMADL + GMADL Weighted Combination (beta=0.3)"),
    ("imadl_gmadl_beta05", "IMADL + GMADL Weighted Combination (beta=0.5)"),
    ("imadl_gmadl_beta07", "IMADL + GMADL Weighted Combination (beta=0.7)"),
    # Variant 3: M2 Robustness Enhanced (3)
    ("m2_robust_gamma001", "M2 Robustness Enhanced (gamma=0.01)"),
    ("m2_robust_gamma01", "M2 Robustness Enhanced (gamma=0.1)"),
    ("m2_robust_gamma10", "M2 Robustness Enhanced (gamma=1.0)"),
    # Variant 4: Adaptive Hybrid (3)
    ("adaptive_lambda10", "Adaptive Hybrid (lambda=1.0)"),
    ("adaptive_lambda50", "Adaptive Hybrid (lambda=5.0)"),
    ("adaptive_lambda100", "Adaptive Hybrid (lambda=10.0)"),
]

RUNNER_TEMPLATE = '''#!/usr/bin/env python3
"""
Phase 2 Experiment: {description}
"""
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check


def main():
    parser = build_arg_parser("Phase 2: {description}")
    args = parser.parse_args()
    run_sanity_check("{loss_name}", args)


if __name__ == "__main__":
    main()
'''


def generate_runner_scripts():
    """Generate all 13 runner scripts."""
    created = []

    for loss_name, description in PHASE2_LOSSES:
        filename = f"run_sanity_check_{loss_name}.py"
        filepath = Path(filename)

        # Generate content
        content = RUNNER_TEMPLATE.format(
            loss_name=loss_name,
            description=description
        )

        # Write file
        filepath.write_text(content)
        filepath.chmod(0o755)  # Make executable

        created.append(filename)
        print(f"✓ Created: {filename}")

    print(f"\n✓ Successfully created {len(created)} runner scripts")
    return created


if __name__ == "__main__":
    generate_runner_scripts()
