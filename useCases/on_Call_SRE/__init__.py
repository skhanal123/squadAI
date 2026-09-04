"""On-call SRE incident triage workflow for SquadAI evaluation."""

__all__ = ["build_incident_squad", "run_incident_triage"]


def __getattr__(name: str):
    if name == "build_incident_squad":
        from useCases.on_Call_SRE.workflow import build_incident_squad

        return build_incident_squad
    if name == "run_incident_triage":
        from useCases.on_Call_SRE.workflow import run_incident_triage

        return run_incident_triage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
