from pathlib import Path
import os
import sys
from core.engine.logic_engine import AICConsultingModel, create_aic_system
from core.interface.launch_aic import main as launch_interface
from core.web_automation.web_controller import WebController

def setup_environment():
    """Setup the AI Consult environment"""
    # Create necessary directories if they don't exist
    dirs = [
        'data/raw',
        'data/processed',
        'output/reports',
        'output/visualizations',
        'models/checkpoints',
        'core/web_automation/config'
    ]
    
    base_dir = Path(__file__).parent
    for dir_path in dirs:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    return base_dir

def main():
    """Main entry point for AI Consult system"""
    print("\n=== AI Consult System ===")
    print("1. Launch Analysis Interface")
    print("2. Create New AIC System")
    print("3. Run Diagnostics")
    print("4. Web Automation")
    print("5. Exit")
    
    try:
        choice = input("\nSelect an option (1-5): ")
        
        if choice == '1':
            launch_interface()
        elif choice == '2':
            print("\nCreating new AIC system...")
            aic_system = create_aic_system()
            print("✓ AIC system created successfully")
        elif choice == '3':
            run_diagnostics()
        elif choice == '4':
            run_web_automation()
        elif choice == '5':
            print("\nExiting AI Consult system...")
            sys.exit(0)
        else:
            print("\n❌ Invalid option selected")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0

def run_web_automation():
    """Run web automation tasks"""
    print("\n=== Web Automation Menu ===")
    print("1. Smart Web Search")
    print("2. Add Allowed Domain")
    print("3. View Allowed Domains")
    print("4. Back to Main Menu")
    
    web_controller = WebController()
    
    try:
        choice = input("\nSelect an option (1-4): ")
        
        if choice == '1':
            query = input("Enter search query: ")
            num_results = int(input("Number of results to analyze (default 5): ") or "5")
            print("\nPerforming smart search and analysis...")
            
            results = web_controller.smart_search(query, num_results)
            
            print(f"\nFound {len(results)} relevant results:")
            for i, result in enumerate(results, 1):
                analysis = result['analysis']
                print(f"\n{i}. {result['title']}")
                print(f"URL: {result['url']}")
                print(f"Relevance Score: {analysis['relevance_score']:.2f}")
                print("\nKey Points:")
                for point in analysis['key_points'][:3]:
                    print(f"- {point}")
                
                if i < len(results):
                    if input("\nPress Enter for next result or 'q' to quit: ").lower() == 'q':
                        break
            
        elif choice == '2':
            domain = input("Enter domain to allow (e.g., 'example.com'): ")
            web_controller.add_allowed_domain(domain)
            print(f"✓ Added {domain} to allowed domains")
            
        elif choice == '3':
            print("\nAllowed Domains:")
            for domain in web_controller.allowed_domains["allowed"]:
                print(f"- {domain}")
                
        elif choice == '4':
            return
            
        else:
            print("\n❌ Invalid option selected")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        
    # Ask if user wants to perform another action
    if input("\nWould you like to perform another web automation task? (y/n): ").lower() == 'y':
        run_web_automation()

def run_diagnostics():
    """Run system diagnostics"""
    print("\nRunning system diagnostics...")
    
    # Check core components
    components = [
        ('Logic Engine', 'core/engine/logic_engine.py'),
        ('ML Agents', 'core/ml_agents'),
        ('Data Processor', 'core/data/data_processor.py'),
        ('Interface', 'core/interface/launch_aic.py'),
        ('Web Controller', 'core/web_automation/web_controller.py')
    ]
    
    base_dir = Path(__file__).parent
    all_ok = True
    
    for name, path in components:
        component_path = base_dir / path
        if component_path.exists():
            print(f"✓ {name}: OK")
        else:
            print(f"❌ {name}: Missing ({path})")
            all_ok = False
    
    if all_ok:
        print("\n✓ All system components present")
    else:
        print("\n❌ Some components are missing. Please check the installation")

if __name__ == "__main__":
    setup_environment()
    sys.exit(main()) 