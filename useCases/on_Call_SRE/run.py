"""
On-call SRE incident triage — SquadAI evaluation scenario.

Run from project root:
    python useCases/on_Call_SRE/run.py
    python useCases/on_Call_SRE/run.py --variant qa
    python useCases/on_Call_SRE/run.py --variant baseline --usage
    python useCases/on_Call_SRE/run.py --variant qa --incident-id checkout-latency-db-pool -v --output-dir useCases/on_Call_SRE/output/checkout-latency-db-pool
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python useCases/on_Call_SRE/run.py` from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from useCases.on_Call_SRE.fixtures import (  # noqa: E402
    DEFAULT_INCIDENT_ID,
    list_incident_ids,
)
from useCases.on_Call_SRE.output_writer import DEFAULT_OUTPUT_DIR, write_triage_output  # noqa: E402
from useCases.on_Call_SRE.workflow import run_incident_triage  # noqa: E402


def _print_header(variant: str, incident: dict, incident_id: str) -> None:
    print("=" * 70)
    print("On-Call SRE: Incident Triage Pipeline")
    print("=" * 70)
    print(f"Variant:      {variant}")
    print(f"Incident ID:  {incident_id}")
    print(f"Service:  {incident['service']}")
    print(f"Region:   {incident['region']}")
    print(f"Alert:    {incident['alert_time']} — {incident['symptom']}")
    print("-" * 70)


def _print_pipeline(result, *, verbose: bool = False) -> None:
    print("Pipeline steps:")
    for step in result.task_results:
        label = step.description[:48] + ("..." if len(step.description) > 48 else "")
        print(f"  [{label}]")
        if verbose:
            print(step.output)
        else:
            preview = step.output.replace("\n", " ")[:100]
            print(f"    -> {preview}{'...' if len(step.output) > 100 else ''}")


def _print_score(score) -> None:
    print("-" * 70)
    print("Evaluation score (0=fail, 1=partial, 2=pass per dimension):")
    for key, value in score.as_dict().items():
        print(f"  {key}: {value}")


def _print_usage(result) -> None:
    if result.usage_display:
        print("-" * 70)
        print(result.usage_display)
    else:
        print("-" * 70)
        print(
            f"Tokens — input: {result.usage.input_tokens}, "
            f"output: {result.usage.output_tokens}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the on-call SRE incident triage SquadAI scenario.",
    )
    parser.add_argument(
        "--variant",
        choices=["baseline", "qa"],
        default="baseline",
        help="baseline = commander + runbook; qa = adds validation gate on commander",
    )
    parser.add_argument(
        "--usage",
        action="store_true",
        help="Include token usage summary in output",
    )
    parser.add_argument(
        "--incident-id",
        default=DEFAULT_INCIDENT_ID,
        choices=list_incident_ids(),
        help="Scenario profile to run (selects fixture data and scoring ground truth)",
    )
    parser.add_argument("--service", default=None, help="Override service from incident profile")
    parser.add_argument("--region", default=None, help="Override region from incident profile")
    parser.add_argument("--alert-time", default=None, help="Override alert time from profile")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print full pipeline step outputs (default: truncated previews)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write triage output files to on_Call_SRE/output/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for incident output files (default: "
            "useCases/on_Call_SRE/output/<incident-id>)"
        ),
    )
    args = parser.parse_args()

    overrides = {}
    if args.service:
        overrides["service"] = args.service
    if args.region:
        overrides["region"] = args.region
    if args.alert_time:
        overrides["alert_time"] = args.alert_time

    from useCases.on_Call_SRE.fixtures import get_incident_profile

    profile = get_incident_profile(args.incident_id)
    incident = {
        "service": profile["service"],
        "region": profile["region"],
        "alert_time": profile["alert_time"],
        "symptom": profile["symptom"],
        **overrides,
    }

    _print_header(args.variant, incident, args.incident_id)

    result, score, bundle = run_incident_triage(
        variant=args.variant,
        include_usage_in_result=args.usage,
        incident_id=args.incident_id,
        **overrides,
    )

    _print_pipeline(result, verbose=args.verbose)

    print("-" * 70)
    print("Commander assessment:")
    print(result.get(bundle.task_commander))

    print("-" * 70)
    print("Final (runbook):")
    print(result.final)

    _print_score(score)

    if args.usage:
        _print_usage(result)

    if not args.no_save:
        output_dir = args.output_dir or (DEFAULT_OUTPUT_DIR / args.incident_id)
        run_dir = write_triage_output(
            result=result,
            bundle=bundle,
            incident=incident,
            output_dir=output_dir,
        )
        print("-" * 70)
        print(f"Incident output saved to: {run_dir}")
        print(f"  - {run_dir / 'incident_report.md'}")
        print(f"  - {run_dir / 'incident.json'}")
        print(f"  - {run_dir / 'execution_trace.json'}")
        print(f"  - {run_dir / 'assessment.txt'}")
        print(f"  - {run_dir / 'runbook.txt'}")
        print(f"  - {run_dir / 'investigations'}/")

    print()


if __name__ == "__main__":
    main()
