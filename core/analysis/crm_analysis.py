from data_analyzer import CompanyDataAnalyzer
from pathlib import Path
import json
import os
import time
from datetime import datetime

def setup_directories():
    """Setup necessary directories for analysis."""
    base_dir = Path(__file__).parent.parent
    dirs = {
        'data': base_dir / 'data',
        'output': base_dir / 'output',
        'reports': base_dir / 'output' / 'reports',
        'visualizations': base_dir / 'output' / 'visualizations'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def analyze_crm_data():
    """Analyze ECRM and SCRM data with comprehensive reporting."""
    print("\n🚀 Starting CRM Data Analysis")
    print("=" * 50)
    
    # Setup directories
    dirs = setup_directories()
    
    # Initialize analyzer
    analyzer = CompanyDataAnalyzer(processing_mode="Memory Efficient")
    
    # Define input files
    ecrm_file = dirs['data'] / 'ECRM.csv'
    scrm_file = dirs['data'] / 'SCRM.csv'
    
    if not (ecrm_file.exists() and scrm_file.exists()):
        print("❌ Error: Required files not found!")
        print(f"Please ensure both files exist in {dirs['data']}")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Analyze ECRM
        print("\n📊 Analyzing ECRM Data...")
        start_time = time.time()
        ecrm_analysis = analyzer.analyze_data(analyzer.load_data(ecrm_file))
        ecrm_time = time.time() - start_time
        print(f"✅ ECRM Analysis completed in {ecrm_time:.2f} seconds")
        
        # Save ECRM analysis
        ecrm_report = dirs['reports'] / f'ecrm_analysis_{timestamp}.json'
        analyzer.generate_report(ecrm_analysis, str(ecrm_report))
        print(f"📝 ECRM Report saved to {ecrm_report}")
        
        # Analyze SCRM
        print("\n📊 Analyzing SCRM Data...")
        start_time = time.time()
        scrm_analysis = analyzer.analyze_data(analyzer.load_data(scrm_file))
        scrm_time = time.time() - start_time
        print(f"✅ SCRM Analysis completed in {scrm_time:.2f} seconds")
        
        # Save SCRM analysis
        scrm_report = dirs['reports'] / f'scrm_analysis_{timestamp}.json'
        analyzer.generate_report(scrm_analysis, str(scrm_report))
        print(f"📝 SCRM Report saved to {scrm_report}")
        
        # Compare datasets
        print("\n🔄 Comparing ECRM and SCRM datasets...")
        start_time = time.time()
        comparison = analyzer.compare_datasets(str(ecrm_file), str(scrm_file))
        comparison_time = time.time() - start_time
        print(f"✅ Comparison completed in {comparison_time:.2f} seconds")
        
        # Save comparison
        comparison_report = dirs['reports'] / f'comparison_{timestamp}.json'
        analyzer.generate_report(comparison, str(comparison_report))
        print(f"📝 Comparison Report saved to {comparison_report}")
        
        # Generate visualizations
        print("\n📊 Generating visualizations...")
        try:
            # Power BI
            power_bi_url = analyzer.visualize_in_power_bi(
                comparison,
                workspace_name="CRM Analysis"
            )
            print(f"✨ Power BI report available at: {power_bi_url}")
        except Exception as e:
            print(f"⚠️ Power BI visualization error: {str(e)}")
        
        try:
            # Tableau
            tableau_url = analyzer.visualize_in_tableau(
                comparison,
                project_name="CRM Analysis",
                site_name=os.getenv('TABLEAU_SITE', '')
            )
            print(f"✨ Tableau workbook available at: {tableau_url}")
        except Exception as e:
            print(f"⚠️ Tableau visualization error: {str(e)}")
        
        # Generate summary
        summary = {
            'timestamp': timestamp,
            'files_analyzed': {
                'ecrm': str(ecrm_file),
                'scrm': str(scrm_file)
            },
            'processing_times': {
                'ecrm_analysis': ecrm_time,
                'scrm_analysis': scrm_time,
                'comparison': comparison_time
            },
            'reports_generated': {
                'ecrm': str(ecrm_report),
                'scrm': str(scrm_report),
                'comparison': str(comparison_report)
            },
            'visualizations': {
                'power_bi': power_bi_url if 'power_bi_url' in locals() else None,
                'tableau': tableau_url if 'tableau_url' in locals() else None
            }
        }
        
        # Save summary
        summary_file = dirs['reports'] / f'analysis_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n📝 Analysis summary saved to {summary_file}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        raise
    
    print("\n✨ Analysis complete!")
    print("=" * 50)

if __name__ == "__main__":
    analyze_crm_data() 