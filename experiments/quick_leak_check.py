"""
Quick Label Leak Diagnostic
============================
Run this script to quickly check for the most common label leak issues.
"""

import sys
import os

# Add parent to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from experiments.detect_label_leaks import LabelLeakDetector


def quick_diagnostic():
    """Run quick diagnostic checks."""
    print("=" * 80)
    print("QUICK LABEL LEAK DIAGNOSTIC")
    print("=" * 80)
    print()
    print("This script will check for common label leak issues:")
    print("  1. Train/val/test distribution similarity")
    print("  2. Sample overlap between splits")
    print("  3. Graph-label alignment issues")
    print()
    print("Starting checks...")
    print()

    detector = LabelLeakDetector()

    try:
        # Initialize
        detector.initialize()
        print()

        # Run critical checks
        print("Running distribution check...")
        detector.check_label_distribution()
        print()

        print("Running overlap check...")
        detector.check_sample_overlap()
        print()

        print("Running alignment check...")
        detector.check_graph_label_alignment()
        print()

        print("=" * 80)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 80)
        print()
        print("Check Logs/label_leak_detection.log for detailed results.")
        print()
        print("Summary of findings:")
        print("  - If no errors shown above, splits appear valid")
        print("  - If warnings or errors appear, review the log file")
        print("  - Expected: Some distribution differences between splits")
        print("  - Expected: Zero sample overlap")
        print("  - Expected: Perfect graph-label alignment")
        print()

    except Exception as e:
        print(f"\n❌ Error during diagnostic: {e}")
        import traceback

        traceback.print_exc()
        print("\nCheck if trainer and dataset are properly configured.")


if __name__ == "__main__":
    quick_diagnostic()
