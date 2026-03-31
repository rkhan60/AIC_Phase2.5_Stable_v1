"""CLI interface for the AIC consulting system."""

from pathlib import Path
import sys


def main():
    print("AIC — AI Consulting System")
    print("------------------------------------------")

    base_dir = Path(__file__).parent.parent.parent
    db_path = str(base_dir / "data" / "aic.db")
    (base_dir / "data").mkdir(parents=True, exist_ok=True)
    (base_dir / "memory").mkdir(exist_ok=True)

    from ..consulting_service import ConsultingService
    svc = ConsultingService(memory_dir=str(base_dir / "memory"), db_path=db_path)

    print("\nDescribe your business problem (or type 'quit' to exit).")

    while True:
        try:
            problem = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not problem or problem.lower() in ("quit", "exit", "q"):
            break

        print("\nOptional context (press Enter to skip):")
        industry = input("  Industry [general]: ").strip() or "general"
        company_size = input("  Company size (micro/sme/mid-market/enterprise) [sme]: ").strip() or "sme"
        market_position = input("  Market position (leader/challenger/follower/niche) [challenger]: ").strip() or "challenger"

        context = {
            "industry": industry,
            "company_size": company_size,
            "market_position": market_position,
        }

        print("\nRunning analysis…")
        try:
            report = svc.analyze(problem, context)
        except Exception as exc:
            print(f"\nError: {exc}")
            continue

        print("\n" + "=" * 60)
        print(f"Session : {report.session_id}")
        print(f"Confidence: {report.confidence:.0%}  |  Validation: {report.validation_status}")
        if report.past_sessions_used:
            print(f"(informed by {report.past_sessions_used} similar past session(s))")

        print("\n--- Reasoning ---")
        print(report.reasoning_summary)

        print("\n--- Self-Critique ---")
        print(report.critique_summary)

        if report.recommended_frameworks:
            print("\n--- Recommended Frameworks ---")
            for fw in report.recommended_frameworks:
                print(f"  • {fw}")

        print("\n--- Framework Analyses ---")
        for fw_name, analysis in report.framework_analyses.items():
            if fw_name in ("problem", "context_summary"):
                continue
            print(f"\n  [{fw_name.upper().replace('_', ' ')}]")
            if isinstance(analysis, dict):
                for k, v in analysis.items():
                    if k not in ("framework", "problem"):
                        print(f"    {k}: {v}")
            else:
                print(f"    {analysis}")

        print("\n" + "=" * 60)
        print("Analysis complete.")

        again = input("\nRun another analysis? [y/N]: ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()
