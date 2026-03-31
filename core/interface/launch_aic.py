"""CLI launcher for the AIC consulting system."""
from __future__ import annotations

from pathlib import Path


def main() -> None:
    print("AIC — AI Consulting System")
    print("------------------------------------------")

    base = Path(__file__).parent.parent.parent
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "memory").mkdir(exist_ok=True)

    from ..consulting_service import ConsultingService

    svc = ConsultingService(
        memory_dir=str(base / "memory"),
        db_path=str(base / "data" / "aic.db"),
    )

    print("\nDescribe your business problem (or 'quit' to exit).")

    while True:
        try:
            problem = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not problem or problem.lower() in ("quit", "exit", "q"):
            break

        print("\nOptional context (Enter to skip):")
        industry = input("  Industry [general]: ").strip() or "general"
        company_size = input("  Company size [sme]: ").strip() or "sme"
        market_position = input("  Market position [challenger]: ").strip() or "challenger"

        print("\nRunning analysis...")
        try:
            report = svc.analyze(problem, {
                "industry": industry,
                "company_size": company_size,
                "market_position": market_position,
            })
        except Exception as exc:
            print(f"\nError: {exc}")
            continue

        print("\n" + "=" * 60)
        print(f"Session   : {report.session_id}")
        print(f"Confidence: {report.confidence:.0%}  |  Validation: {report.validation_status}")
        if report.past_sessions_used:
            print(f"(informed by {report.past_sessions_used} similar past session(s))")
        print("\n--- Reasoning ---")
        print(report.reasoning_summary)
        print("\n--- Critique ---")
        print(report.critique_summary)
        if report.recommended_frameworks:
            print("\n--- Recommended Frameworks ---")
            for fw in report.recommended_frameworks:
                print(f"  * {fw}")
        print("=" * 60)

        again = input("\nRun another analysis? [y/N]: ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()
