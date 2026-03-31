"""AIC — main entry point."""

from pathlib import Path
import sys
import os

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def setup_environment() -> Path:
    """Create required directories."""
    dirs = [
        'data/raw', 'data/processed', 'output/reports',
        'output/visualizations', 'memory',
    ]
    base_dir = Path(__file__).parent
    for d in dirs:
        (base_dir / d).mkdir(parents=True, exist_ok=True)
    return base_dir


# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------

def run_consulting_analysis():
    """Interactively run a consulting analysis."""
    print("\n=== Consulting Analysis ===")
    problem = input("Describe your business problem:\n> ").strip()
    if not problem:
        print("❌ No problem entered.")
        return

    print("\nOptional context (press Enter to skip each):")
    industry = input("  Industry (e.g. retail, SaaS): ").strip() or "general"
    company_size = input("  Company size (micro/sme/mid-market/enterprise): ").strip() or "sme"
    market_position = input("  Market position (leader/challenger/follower/niche): ").strip() or "challenger"

    context = {
        "industry": industry,
        "company_size": company_size,
        "market_position": market_position,
    }

    print("\nInitialising consulting service (this may take a moment on first run)…")
    from core.consulting_service import ConsultingService
    db_path = str(Path(__file__).parent / "data" / "aic.db")
    svc = ConsultingService(memory_dir="memory", db_path=db_path)

    print("Running analysis…")
    try:
        report = svc.analyze(problem, context)
    except Exception as exc:
        print(f"\n❌ Analysis failed: {exc}")
        return

    # --- Display results ---
    print("\n" + "=" * 60)
    print(f"Session: {report.session_id}")
    print(f"Confidence: {report.confidence:.0%}")
    if report.past_sessions_used:
        print(f"(informed by {report.past_sessions_used} similar past session(s))")

    print("\n--- Reasoning ---")
    print(report.reasoning_summary)
    print(f"Validation: {report.validation_status}")

    print("\n--- Self-Critique ---")
    print(report.critique_summary)

    if report.recommended_frameworks:
        print("\n--- Recommended Frameworks ---")
        for fw in report.recommended_frameworks:
            print(f"  • {fw}")

    print("\n--- Framework Analyses ---")
    for fw_name, analysis in report.framework_analyses.items():
        print(f"\n  [{fw_name.upper()}]")
        if isinstance(analysis, dict):
            for k, v in analysis.items():
                if k not in ("framework", "problem"):
                    print(f"    {k}: {v}")
        else:
            print(f"    {analysis}")

    print("\n" + "=" * 60)
    print("✓ Analysis complete.")


def run_diagnostics():
    """Check that all core components are present."""
    print("\nRunning system diagnostics…")
    base_dir = Path(__file__).parent
    components = [
        ('Consulting Service', 'core/consulting_service.py'),
        ('Pipeline Autonomy', 'core/pipeline_autonomy.py'),
        ('AIC System', 'core/aic_system.py'),
        ('Config', 'core/config.py'),
        ('Data Processor', 'core/data/data_processor.py'),
        ('Interface', 'core/interface/launch_aic.py'),
        ('Web Controller', 'core/web_automation/web_controller.py'),
    ]
    all_ok = True
    for name, path in components:
        exists = (base_dir / path).exists()
        print(f"{'✓' if exists else '❌'} {name}: {'OK' if exists else f'Missing ({path})'}")
        all_ok = all_ok and exists

    # Import check
    try:
        from core.engine import create_aic_system, ConsultingFrameworkEngine
        print("✓ Engine imports: OK")
    except Exception as exc:
        print(f"❌ Engine imports failed: {exc}")
        all_ok = False

    print("\n✓ All components present." if all_ok else "\n❌ Some components are missing.")


def run_web_automation():
    """Run web automation tasks."""
    print("\n=== Web Automation Menu ===")
    print("1. Smart Web Search")
    print("2. Add Allowed Domain")
    print("3. View Allowed Domains")
    print("4. Back to Main Menu")

    try:
        from core.web_automation.web_controller import WebController
        web_controller = WebController()
    except Exception as exc:
        print(f"❌ Web controller unavailable: {exc}")
        return

    choice = input("\nSelect an option (1-4): ").strip()

    if choice == '1':
        query = input("Enter search query: ").strip()
        num_results = int(input("Number of results (default 5): ").strip() or "5")
        results = web_controller.smart_search(query, num_results)
        print(f"\nFound {len(results)} results.")
        for i, r in enumerate(results, 1):
            a = r.get('analysis', {})
            print(f"\n{i}. {r.get('title', '')}")
            print(f"   Score: {a.get('relevance_score', 0):.2f}")
            for pt in a.get('key_points', [])[:3]:
                print(f"   - {pt}")
    elif choice == '2':
        domain = input("Domain to allow: ").strip()
        web_controller.add_allowed_domain(domain)
        print(f"✓ Added {domain}")
    elif choice == '3':
        for d in web_controller.allowed_domains.get("allowed", []):
            print(f"  - {d}")


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main() -> int:
    print("\n=== AI Consulting System ===")
    print("1. Launch Analysis Interface (CLI)")
    print("2. Run Consulting Analysis")
    print("3. Run Diagnostics")
    print("4. Web Automation")
    print("5. Exit")

    try:
        choice = input("\nSelect an option (1-5): ").strip()

        if choice == '1':
            from core.interface.launch_aic import main as launch_interface
            launch_interface()
        elif choice == '2':
            run_consulting_analysis()
        elif choice == '3':
            run_diagnostics()
        elif choice == '4':
            run_web_automation()
        elif choice == '5':
            print("\nExiting…")
            sys.exit(0)
        else:
            print("\n❌ Invalid option.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"\n❌ Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    setup_environment()
    sys.exit(main())
