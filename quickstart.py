#!/usr/bin/env python
"""
quickstart.py
=============
One-click setup and run script for PayPal Financial Q&A System
Handles all setup steps automatically
"""

import os
import sys
import subprocess
from pathlib import Path
import time

class QuickStart:
    """Automated setup and execution"""
    
    def __init__(self):
        self.root_dir = Path.cwd()
        self.venv_dir = self.root_dir / "venv"
        self.data_dir = self.root_dir / "financial_data"
        
    def check_python_version(self):
        """Check Python version"""
        print("📌 Checking Python version...")
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        print(f"✅ Python {sys.version.split()[0]} detected")
        return True
    
    def setup_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating directories...")
        dirs = [
            "financial_data",
            "processed_data",
            "models",
            "indices",
            "evaluation_results"
        ]
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
        print("✅ Directories created")
    
    def check_data_files(self):
        """Check if PayPal reports are present"""
        print("\n📄 Checking for PayPal reports...")
        report_2023 = self.data_dir / "Paypal2023_report.pdf"
        report_2024 = self.data_dir / "Paypal2024_report.pdf"
        
        if not report_2023.exists() or not report_2024.exists():
            print("⚠️  PayPal reports not found!")
            print(f"Please place the following files in {self.data_dir}:")
            print("  - Paypal2023_report.pdf")
            print("  - Paypal2024_report.pdf")
            return False
        
        print("✅ Both PayPal reports found")
        return True
    
    def install_dependencies(self):
        """Install required packages"""
        print("\n📦 Installing dependencies...")
        print("This may take 5-10 minutes...")
        
        # Upgrade pip
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      capture_output=True)
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                      check=True)
        
        # Download NLTK data
        print("📚 Downloading NLTK data...")
        subprocess.run([sys.executable, "-c", 
                       "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"],
                      capture_output=True)
        
        print("✅ Dependencies installed")
    
    def run_pipeline(self):
        """Run the main pipeline"""
        print("\n🚀 Running PayPal Q&A Pipeline...")
        print("="*60)
        
        # Import and run main
        try:
            from main import PayPalQAOrchestrator
            
            orchestrator = PayPalQAOrchestrator()
            success = orchestrator.run_complete_pipeline()
            
            if success:
                print("\n✅ Pipeline completed successfully!")
                return True
            else:
                print("\n❌ Pipeline failed")
                return False
                
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Please ensure all Python scripts are in the current directory")
            return False
    
    def launch_interface(self):
        """Launch Streamlit interface"""
        print("\n🌐 Launching web interface...")
        print("The browser should open automatically")
        print("If not, navigate to: http://localhost:8501")
        print("\nPress Ctrl+C to stop the server")
        
        subprocess.run(["streamlit", "run", "interface.py"])
    
    def run(self):
        """Main execution"""
        print("="*60)
        print("🚀 PayPal Financial Q&A System - Quick Start")
        print("="*60)
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Setup directories
        self.setup_directories()
        
        # Check data files
        if not self.check_data_files():
            print("\n⏸️  Please add the PayPal reports and run again")
            return False
        
        # Check if dependencies are installed
        try:
            import torch
            import transformers
            import streamlit
            print("\n✅ Dependencies already installed")
        except ImportError:
            print("\n📦 Dependencies not found. Installing...")
            self.install_dependencies()
        
        # Run pipeline
        if not self.run_pipeline():
            return False
        
        # Ask if user wants to launch interface
        print("\n" + "="*60)
        response = input("Would you like to launch the web interface? (y/n): ")
        if response.lower() == 'y':
            self.launch_interface()
        else:
            print("\nTo launch the interface later, run:")
            print("  streamlit run interface.py")
        
        return True

def main():
    """Entry point"""
    quickstart = QuickStart()
    
    try:
        success = quickstart.run()
        if success:
            print("\n🎉 Setup complete! Your PayPal Q&A system is ready.")
        else:
            print("\n⚠️  Setup incomplete. Please check the errors above.")
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()