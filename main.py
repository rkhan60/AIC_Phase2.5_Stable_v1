"""AIC — AI Consulting System entry point."""
from __future__ import annotations

import sys
from pathlib import Path


def setup_environment() -> Path:
    base = Path(__file__).parent
    for d in ["data/raw", "data/processed", "output/reports", "output/visualizations", "memory"]:
        (base / d).mkdir(parents=True, exist_ok=True)
    return base


def run_consulting_analysis(base: Path) -> None:
    from core.consulting_service import ConsultingService

    svc = ConsultingService(
        memory_dir=str(base / "memory"),
        db_path=str(base / "data" / "aic.db"),
    )

    print("\nDescribe your business problem (or 'back' to return):")
    try:
        problem = input("> ").strip()
    except (KeyboardInterrupt, EOFError):
        return
    if not problem or problem.lower() in ("back", "exit"):
        return

    print("\nOptional context (press Enter to skip each):")
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
        return

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


def run_diagnostics() -> None:
    print("\nRunning system diagnostics...")
    base = Path(__file__).parent
    components = [
        ("Logic Engine", "core/engine/logic_engine.py"),
        ("AIC System", "core/aic_system.py"),
        ("Consulting Service", "core/consulting_service.py"),
        ("Data Processor", "core/data/data_processor.py"),
        ("Storage", "core/storage/__init__.py"),
        ("REST API", "api/main.py"),
        ("Streamlit App", "app/streamlit_app.py"),
    ]
    all_ok = True
    for name, path in components:
        ok = (base / path).exists()
        print(f"{'OK' if ok else 'MISSING':8s}  {name}")
        all_ok = all_ok and ok
    print("\nAll components present." if all_ok else "\nSome components missing.")


def run_web_automation() -> None:
    try:
        from core.web_automation.web_controller import WebController
        wc = WebController()
        query = input("Search query: ").strip()
        if not query:
            return
        results = wc.smart_search(query, 3)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r.get('title', 'N/A')}")
            for pt in r.get("analysis", {}).get("key_points", [])[:2]:
                print(f"   - {pt}")
    except Exception as exc:
        print(f"Web automation unavailable: {exc}")


def main() -> int:
    base = setup_environment()
    print("\n=== AIC — AI Consulting System ===")
    print("1. Launch Streamlit Interface")
    print("2. Run Consulting Analysis (CLI)")
    print("3. Run Diagnostics")
    print("4. Web Automation")
    print("5. Exit")

    try:
        choice = input("\nSelect an option (1-5): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return 0

    if choice == "1":
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])
    elif choice == "2":
        run_consulting_analysis(base)
    elif choice == "3":
        run_diagnostics()
    elif choice == "4":
        run_web_automation()
    elif choice == "5":
        print("Goodbye.")
    else:
        print("Invalid option.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
