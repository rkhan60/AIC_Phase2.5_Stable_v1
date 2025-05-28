from pathlib import Path
import os
from ..engine.logic_engine import AICConsultingModel
from ..data.data_processor import CompanyDataProcessor as DataProcessor

def main():
    print("🚀 Launching AI Consulting System")
    print("----------------------------------------")
    
    # Initialize the system
    model = AICConsultingModel()
    processor = DataProcessor()
    
    # Get the data directory
    data_dir = Path(__file__).parent.parent.parent / "data/raw"
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        print("\n📁 Created 'data' directory for your files")
        print("Please place your data files in the 'data/raw' directory")
        return

    # List available files
    files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls"))
    
    if not files:
        print("\n❌ No data files found in the 'data/raw' directory")
        print("Please add your data files and run again")
        return

    print("\n📊 Available files:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file.name} ({file.stat().st_size / (1024*1024):.2f} MB)")

    # Get user input for file selection
    try:
        choice = int(input("\nSelect a file to analyze (enter number): ")) - 1
        if choice < 0 or choice >= len(files):
            print("❌ Invalid selection")
            return
    except ValueError:
        print("❌ Please enter a valid number")
        return

    selected_file = files[choice]
    print(f"\n🔍 Analyzing {selected_file.name}...")

    try:
        # Process the data
        data = processor.load_data(selected_file)
        processed_data = processor.process_data(data)
        
        # Run analysis
        results = model.consulting_inference(
            processed_data,
            client_context={
                'file_name': selected_file.name,
                'data_size': len(processed_data)
            }
        )
        
        # Save results
        output_dir = Path(__file__).parent.parent.parent / "output/reports"
        output_dir.mkdir(exist_ok=True)
        
        report_file = output_dir / f"analysis_{selected_file.stem}.json"
        processor.save_results(results, report_file)
        print(f"\n✅ Generated analysis report: {report_file}")

        # Generate visualizations
        viz_dir = Path(__file__).parent.parent.parent / "output/visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        viz_file = viz_dir / f"dashboard_{selected_file.stem}.html"
        processor.generate_visualization(results, viz_file)
        print(f"\n✅ Generated visualization: {viz_file}")

    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        return

    print("\n✨ Analysis complete! Check the output directory for results.")

if __name__ == "__main__":
    main() 