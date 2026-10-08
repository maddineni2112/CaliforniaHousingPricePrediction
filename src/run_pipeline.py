"""Main entry-point CLI for California Housing Machine Learning Project.

Usage:
    python run_pipeline.py --train       Train models, run RFE, and export evaluation figures
    python run_pipeline.py --app         Launch Streamlit interactive web application
    python run_pipeline.py --api         Launch Flask REST API & Web Dashboard
    python run_pipeline.py --predict     Run a sample prediction in the terminal
"""

import sys
import os
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="California Housing Price Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Execute end-to-end data loading, outlier cleaning, feature engineering, and model training.",
    )
    parser.add_argument(
        "--app",
        action="store_true",
        help="Launch the interactive Streamlit UI dashboard.",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Launch the Flask REST API & web dashboard.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run sample inference on a test house block.",
    )
    parser.add_argument(
        "--redfin",
        action="store_true",
        help="Download or update real California market data from Redfin Data Center.",
    )

    args = parser.parse_args()

    if args.train:
        from src.train import train_pipeline
        train_pipeline()
    elif args.app:
        app_file = os.path.join(PROJECT_ROOT, "app", "app.py")
        print(f"Launching Streamlit application from {app_file}...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_file])
    elif args.api:
        api_file = os.path.join(PROJECT_ROOT, "app", "web_server.py")
        print(f"Launching Flask API from {api_file}...")
        subprocess.run([sys.executable, api_file])
    elif args.redfin:
        from src.redfin_loader import fetch_redfin_california_state, fetch_redfin_california_counties, get_latest_county_summary
        print("Updating Redfin Data Center datasets...")
        fetch_redfin_california_state()
        county_df = fetch_redfin_california_counties()
        print("\nLatest California County Redfin Market Summary:")
        summary = get_latest_county_summary(county_df)
        print(summary[["County", "PERIOD_BEGIN", "MEDIAN_SALE_PRICE", "MEDIAN_PPSF", "MEDIAN_DOM", "AVG_SALE_TO_LIST"]].head(15).to_string(index=False))
    elif args.predict:
        from src.predict import sample_prediction
        sample_prediction()
    else:
        # Default behavior if no args given: show help or run sample prediction
        print("No arguments provided. Running sample prediction by default...\n")
        from src.predict import sample_prediction
        sample_prediction()
        print("\nTip: Run 'python run_pipeline.py --app' to open the interactive UI!")


if __name__ == "__main__":
    main()
