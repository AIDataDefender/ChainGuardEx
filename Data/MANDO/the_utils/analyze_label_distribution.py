"""
Label Distribution Analysis Script for DAppSCAN Dataset
Analyzes the distribution of vulnerability labels in the Processed_Data directory
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. CSV export will use basic format.")

# Set up plotting style
if MATPLOTLIB_AVAILABLE:
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.style.use('default')


class LabelDistributionAnalyzer:
    """Analyzes the distribution of vulnerability labels in the dataset"""
    
    def __init__(self, processed_data_dir):
        self.processed_data_dir = Path(processed_data_dir)
        self.swc_counter = Counter()
        self.owasp_counter = Counter()
        self.swc_name_counter = Counter()
        self.project_count = 0
        self.contract_count = 0
        self.vulnerability_details = []
        
    def analyze(self):
        """Main analysis function"""
        print("=" * 80)
        print("DAppSCAN Dataset Label Distribution Analysis")
        print("=" * 80)
        print(f"\nAnalyzing directory: {self.processed_data_dir}")
        print(f"Ignoring: benign folder\n")
        
        # Iterate through all project directories
        for project_dir in sorted(self.processed_data_dir.iterdir()):
            if not project_dir.is_dir():
                continue
            
            # Skip benign folder
            if project_dir.name.lower() == 'benign':
                print(f"Skipping: {project_dir.name}")
                continue
            
            # Look for contract_level_vulnerabilities.json
            vuln_file = project_dir / 'contract_level_vulnerabilities.json'
            
            if vuln_file.exists():
                self.process_vulnerability_file(vuln_file, project_dir.name)
            else:
                print(f"Warning: No vulnerability file found in {project_dir.name}")
        
        self.print_statistics()
        self.create_visualizations()
        self.export_to_csv()
        
    def process_vulnerability_file(self, vuln_file, project_name):
        """Process a single vulnerability JSON file"""
        try:
            with open(vuln_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                return
            
            self.project_count += 1
            
            for contract_name, vuln_info in data.items():
                self.contract_count += 1
                
                swc_id = vuln_info.get('swc_id', 'Unknown')
                owasp_id = vuln_info.get('owasp_id', 'Unknown')
                swc_name = vuln_info.get('swc_name', 'Unknown')
                vuln_type = vuln_info.get('type', 'Unknown')
                
                # Update counters
                self.swc_counter[swc_id] += 1
                self.owasp_counter[owasp_id] += 1
                self.swc_name_counter[swc_name] += 1
                
                # Store detailed information
                self.vulnerability_details.append({
                    'project': project_name,
                    'contract': contract_name,
                    'swc_id': swc_id,
                    'swc_name': swc_name,
                    'owasp_id': owasp_id,
                    'type': vuln_type
                })
                
        except json.JSONDecodeError as e:
            print(f"Error parsing {vuln_file}: {e}")
        except Exception as e:
            print(f"Error processing {vuln_file}: {e}")
    
    def print_statistics(self):
        """Print detailed statistics"""
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Total projects analyzed: {self.project_count}")
        print(f"Total contracts with vulnerabilities: {self.contract_count}")
        print(f"Unique SWC IDs: {len(self.swc_counter)}")
        print(f"Unique OWASP IDs: {len(self.owasp_counter)}")
        print(f"Unique vulnerability types: {len(self.swc_name_counter)}")
        
        # SWC ID Distribution
        print("\n" + "=" * 80)
        print("SWC ID DISTRIBUTION (Top 20)")
        print("=" * 80)
        print(f"{'SWC ID':<15} {'Count':<10} {'Percentage':<12}")
        print("-" * 80)
        for swc_id, count in self.swc_counter.most_common(20):
            percentage = (count / self.contract_count) * 100
            print(f"{swc_id:<15} {count:<10} {percentage:>6.2f}%")
        
        # OWASP ID Distribution
        print("\n" + "=" * 80)
        print("OWASP ID DISTRIBUTION (Top 20)")
        print("=" * 80)
        print(f"{'OWASP ID':<15} {'Count':<10} {'Percentage':<12}")
        print("-" * 80)
        for owasp_id, count in self.owasp_counter.most_common(20):
            percentage = (count / self.contract_count) * 100
            print(f"{owasp_id:<15} {count:<10} {percentage:>6.2f}%")
        
        # Vulnerability Name Distribution
        print("\n" + "=" * 80)
        print("VULNERABILITY NAME DISTRIBUTION (Top 20)")
        print("=" * 80)
        print(f"{'Vulnerability Name':<50} {'Count':<10} {'Percentage':<12}")
        print("-" * 80)
        for name, count in self.swc_name_counter.most_common(20):
            percentage = (count / self.contract_count) * 100
            print(f"{name:<50} {count:<10} {percentage:>6.2f}%")
    
    def create_visualizations(self):
        """Create visualization plots"""
        if not MATPLOTLIB_AVAILABLE:
            print("\nSkipping visualizations (matplotlib not available)")
            return
            
        output_dir = self.processed_data_dir.parent / 'analysis_results'
        output_dir.mkdir(exist_ok=True)
        
        # 1. SWC ID Distribution (Top 15)
        plt.figure(figsize=(14, 8))
        swc_data = self.swc_counter.most_common(15)
        swc_ids, swc_counts = zip(*swc_data) if swc_data else ([], [])
        
        plt.barh(range(len(swc_ids)), swc_counts, color='steelblue')
        plt.yticks(range(len(swc_ids)), swc_ids)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel('SWC ID', fontsize=12)
        plt.title('Top 15 SWC ID Distribution', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add count labels
        for i, count in enumerate(swc_counts):
            plt.text(count, i, f' {count}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'swc_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {output_dir / 'swc_distribution.png'}")
        plt.close()
        
        # 2. OWASP ID Distribution (Top 15)
        plt.figure(figsize=(14, 8))
        owasp_data = self.owasp_counter.most_common(15)
        owasp_ids, owasp_counts = zip(*owasp_data) if owasp_data else ([], [])
        
        plt.barh(range(len(owasp_ids)), owasp_counts, color='coral')
        plt.yticks(range(len(owasp_ids)), owasp_ids)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel('OWASP ID', fontsize=12)
        plt.title('Top 15 OWASP ID Distribution', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add count labels
        for i, count in enumerate(owasp_counts):
            plt.text(count, i, f' {count}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'owasp_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir / 'owasp_distribution.png'}")
        plt.close()
        
        # 3. Vulnerability Name Distribution (Top 15)
        plt.figure(figsize=(16, 10))
        name_data = self.swc_name_counter.most_common(15)
        names, name_counts = zip(*name_data) if name_data else ([], [])
        
        # Truncate long names for display
        display_names = [name[:45] + '...' if len(name) > 45 else name for name in names]
        
        plt.barh(range(len(display_names)), name_counts, color='seagreen')
        plt.yticks(range(len(display_names)), display_names)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel('Vulnerability Name', fontsize=12)
        plt.title('Top 15 Vulnerability Name Distribution', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add count labels
        for i, count in enumerate(name_counts):
            plt.text(count, i, f' {count}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'vulnerability_name_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir / 'vulnerability_name_distribution.png'}")
        plt.close()
        
        # 4. Pie chart for top 10 SWC categories
        plt.figure(figsize=(12, 12))
        top_10_swc = self.swc_counter.most_common(10)
        other_count = sum(self.swc_counter.values()) - sum(count for _, count in top_10_swc)
        
        labels = [swc for swc, _ in top_10_swc]
        sizes = [count for _, count in top_10_swc]
        
        if other_count > 0:
            labels.append('Others')
            sizes.append(other_count)
        
        colors = plt.cm.Set3(range(len(labels)))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        plt.title('SWC ID Distribution (Pie Chart)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'swc_pie_chart.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir / 'swc_pie_chart.png'}")
        plt.close()
        
    def export_to_csv(self):
        """Export detailed data to CSV files"""
        output_dir = self.processed_data_dir.parent / 'analysis_results'
        output_dir.mkdir(exist_ok=True)
        
        if PANDAS_AVAILABLE:
            # Export detailed vulnerability data
            df_details = pd.DataFrame(self.vulnerability_details)
            details_file = output_dir / 'vulnerability_details.csv'
            df_details.to_csv(details_file, index=False)
            print(f"✓ Saved: {details_file}")
            
            # Export SWC distribution
            df_swc = pd.DataFrame(self.swc_counter.most_common(), columns=['SWC_ID', 'Count'])
            df_swc['Percentage'] = (df_swc['Count'] / self.contract_count * 100).round(2)
            swc_file = output_dir / 'swc_distribution.csv'
            df_swc.to_csv(swc_file, index=False)
            print(f"✓ Saved: {swc_file}")
            
            # Export OWASP distribution
            df_owasp = pd.DataFrame(self.owasp_counter.most_common(), columns=['OWASP_ID', 'Count'])
            df_owasp['Percentage'] = (df_owasp['Count'] / self.contract_count * 100).round(2)
            owasp_file = output_dir / 'owasp_distribution.csv'
            df_owasp.to_csv(owasp_file, index=False)
            print(f"✓ Saved: {owasp_file}")
            
            # Export vulnerability name distribution
            df_names = pd.DataFrame(self.swc_name_counter.most_common(), columns=['Vulnerability_Name', 'Count'])
            df_names['Percentage'] = (df_names['Count'] / self.contract_count * 100).round(2)
            names_file = output_dir / 'vulnerability_names_distribution.csv'
            df_names.to_csv(names_file, index=False)
            print(f"✓ Saved: {names_file}")
            
            # Export summary statistics
            summary_data = {
                'Metric': [
                    'Total Projects',
                    'Total Contracts',
                    'Unique SWC IDs',
                    'Unique OWASP IDs',
                    'Unique Vulnerability Types'
                ],
                'Value': [
                    self.project_count,
                    self.contract_count,
                    len(self.swc_counter),
                    len(self.owasp_counter),
                    len(self.swc_name_counter)
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            summary_file = output_dir / 'summary_statistics.csv'
            df_summary.to_csv(summary_file, index=False)
            print(f"✓ Saved: {summary_file}")
        else:
            # Manual CSV export without pandas
            import csv
            
            # Export detailed vulnerability data
            details_file = output_dir / 'vulnerability_details.csv'
            with open(details_file, 'w', newline='', encoding='utf-8') as f:
                if self.vulnerability_details:
                    writer = csv.DictWriter(f, fieldnames=self.vulnerability_details[0].keys())
                    writer.writeheader()
                    writer.writerows(self.vulnerability_details)
            print(f"✓ Saved: {details_file}")
            
            # Export SWC distribution
            swc_file = output_dir / 'swc_distribution.csv'
            with open(swc_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['SWC_ID', 'Count', 'Percentage'])
                for swc_id, count in self.swc_counter.most_common():
                    percentage = round((count / self.contract_count * 100), 2)
                    writer.writerow([swc_id, count, percentage])
            print(f"✓ Saved: {swc_file}")
            
            # Export OWASP distribution
            owasp_file = output_dir / 'owasp_distribution.csv'
            with open(owasp_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['OWASP_ID', 'Count', 'Percentage'])
                for owasp_id, count in self.owasp_counter.most_common():
                    percentage = round((count / self.contract_count * 100), 2)
                    writer.writerow([owasp_id, count, percentage])
            print(f"✓ Saved: {owasp_file}")
            
            # Export vulnerability name distribution
            names_file = output_dir / 'vulnerability_names_distribution.csv'
            with open(names_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Vulnerability_Name', 'Count', 'Percentage'])
                for name, count in self.swc_name_counter.most_common():
                    percentage = round((count / self.contract_count * 100), 2)
                    writer.writerow([name, count, percentage])
            print(f"✓ Saved: {names_file}")
            
            # Export summary statistics
            summary_file = output_dir / 'summary_statistics.csv'
            with open(summary_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Total Projects', self.project_count])
                writer.writerow(['Total Contracts', self.contract_count])
                writer.writerow(['Unique SWC IDs', len(self.swc_counter)])
                writer.writerow(['Unique OWASP IDs', len(self.owasp_counter)])
                writer.writerow(['Unique Vulnerability Types', len(self.swc_name_counter)])
            print(f"✓ Saved: {summary_file}")


def main():
    """Main entry point"""
    # Path to the Processed_Data directory
    script_dir = Path(__file__).parent
    processed_data_dir = script_dir / 'Processed_Data'
    
    if not processed_data_dir.exists():
        print(f"Error: Directory not found: {processed_data_dir}")
        print("Please ensure the Processed_Data directory exists.")
        return
    
    # Create analyzer and run analysis
    analyzer = LabelDistributionAnalyzer(processed_data_dir)
    analyzer.analyze()
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"\nResults saved in: {processed_data_dir.parent / 'analysis_results'}")
    print("\nGenerated files:")
    if MATPLOTLIB_AVAILABLE:
        print("  - swc_distribution.png")
        print("  - owasp_distribution.png")
        print("  - vulnerability_name_distribution.png")
        print("  - swc_pie_chart.png")
    print("  - vulnerability_details.csv")
    print("  - swc_distribution.csv")
    print("  - owasp_distribution.csv")
    print("  - vulnerability_names_distribution.csv")
    print("  - summary_statistics.csv")


if __name__ == '__main__':
    main()
