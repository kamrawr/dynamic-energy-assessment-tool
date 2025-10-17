#!/usr/bin/env python3
"""
Demo of Energy Trust Measures Integration
"""

import pandas as pd
import json
from pathlib import Path

def load_energy_trust_data():
    """Load and display Energy Trust measures"""
    try:
        df = pd.read_csv('data/energy_trust_measures.csv')
        print(f"🏠 Loaded {len(df)} Energy Trust measures")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def demo_income_classification():
    """Demo Oregon AMI income classification"""
    print("\n💰 Oregon AMI Income Classification Demo:")
    
    # Load AMI data
    ami_df = pd.read_csv('data/oregon_ami_2025.csv')
    
    # Example household in Multnomah County
    county = "Multnomah County"
    household_size = 4
    county_row = ami_df[ami_df['County'] == county].iloc[0]
    
    print(f"\n📍 Example: {household_size}-person household in {county}")
    print(f"   • 100% AMI: ${county_row['AMI_100_4Person']:,}")
    print(f"   • 60% AMI (Very Low Income): ${county_row['60%']:,}")  
    print(f"   • 80% AMI (Low Income): ${county_row['80%']:,}")
    print(f"   • 120% AMI (Moderate Income): ${county_row['120%']:,}")

def demo_program_eligibility(measures_df):
    """Demo program eligibility by income level"""
    print("\n🎯 Program Eligibility by Income Level:")
    
    programs = measures_df['program'].unique()
    for program in programs:
        count = len(measures_df[measures_df['program'] == program])
        print(f"   • {program}: {count} measures")

def demo_heat_pump_recommendations(measures_df):
    """Show heat pump recommendations with real incentives"""
    print("\n🔥 Heat Pump Recommendations with Real Energy Trust Incentives:")
    
    heat_pumps = measures_df[measures_df['measure'].str.contains('Heat Pump', na=False)]
    
    for program in ['Community Partner Funding', 'Savings Within Reach', 'Standard']:
        program_hps = heat_pumps[heat_pumps['program'] == program]
        if not program_hps.empty:
            print(f"\n   📋 {program}:")
            for _, hp in program_hps.head(3).iterrows():
                incentive = hp['incentive_value']
                cost = hp['expected_unit_price']
                customer_cost = max(0, cost - (float(str(incentive).replace('$', '').replace(',', '')) if isinstance(incentive, str) else incentive)) if program != 'Community Partner Funding' else 0
                
                print(f"      • {hp['measure']}")
                print(f"        Incentive: ${incentive} | Est. Cost: ${cost:,} | Customer Pays: ${customer_cost:,.0f}")
                print(f"        Requirements: {hp['requirements_summary'][:80]}...")

def demo_insulation_by_income(measures_df):
    """Show how insulation costs vary by income program"""
    print("\n🏘️ Attic Insulation Costs by Income Level:")
    
    attic = measures_df[measures_df['measure'].str.contains('Attic Insulation', na=False)]
    
    for program in ['Community Partner Funding', 'Savings Within Reach', 'Standard']:
        program_attic = attic[attic['program'] == program]
        if not program_attic.empty:
            measure = program_attic.iloc[0]
            incentive_per_sqft = measure['incentive_value']
            cost_per_sqft = measure['expected_unit_price']
            
            # Example: 1000 sq ft attic
            example_sqft = 1000
            total_incentive = float(str(incentive_per_sqft).replace('$', '').replace('/sqft', '')) * example_sqft
            total_cost = cost_per_sqft * example_sqft if cost_per_sqft else 2500  # estimate
            customer_cost = max(0, total_cost - total_incentive) if program != 'Community Partner Funding' else 0
            
            print(f"   • {program}:")
            print(f"     {incentive_per_sqft} incentive | 1000 sq ft attic")
            print(f"     Customer pays: ${customer_cost:,.0f}")

def main():
    print("🎯 Energy Trust Measures Integration Demo")
    print("=" * 50)
    
    # Load Energy Trust data
    measures_df = load_energy_trust_data()
    if measures_df is None:
        return
    
    # Show data overview
    print(f"\n📊 Data Overview:")
    print(f"   • Total measures: {len(measures_df)}")
    print(f"   • Programs: {', '.join(measures_df['program'].unique())}")
    print(f"   • Measure types: {len(measures_df['measure'].unique())} unique measures")
    
    # Demo income classification
    demo_income_classification()
    
    # Demo program eligibility
    demo_program_eligibility(measures_df)
    
    # Demo heat pump recommendations
    demo_heat_pump_recommendations(measures_df)
    
    # Demo insulation by income
    demo_insulation_by_income(measures_df)
    
    print("\n✅ Demo complete! This shows how the tool now uses real Energy Trust data")
    print("   to provide accurate, income-targeted recommendations with actual incentive amounts.")

if __name__ == "__main__":
    main()