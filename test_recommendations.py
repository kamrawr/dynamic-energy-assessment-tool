#!/usr/bin/env python3

from app import AssessmentData, get_targeted_recommendations, measures_database, load_measures_database, load_oregon_ami_data, load_incentives_database

# Load data
load_measures_database()
load_oregon_ami_data()
load_incentives_database()

print(f"Measures database loaded: {len(measures_database)} records")
print(f"Columns: {list(measures_database.columns) if not measures_database.empty else 'No columns'}")

# Create test assessment
test_assessment = AssessmentData(
    heating_type="electric_resistance",
    attic_insulation="poor",
    wall_insulation="none",
    window_type="single_pane",
    annual_income="low_income",  # Simple income option
    county=None,
    household_size=None
)

print(f"\nTest assessment:")
print(f"  Heating type: {test_assessment.heating_type}")
print(f"  Attic insulation: {test_assessment.attic_insulation}")
print(f"  Annual income: {test_assessment.annual_income}")

# Generate recommendations
recommendations = get_targeted_recommendations(test_assessment)

print(f"\nGenerated {len(recommendations)} recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec.title}")
    print(f"   Category: {rec.category}")
    print(f"   Priority: {rec.priority}")
    print(f"   Cost: ${rec.estimated_cost_low:.0f} - ${rec.estimated_cost_high:.0f}")
    if rec.rebates_available:
        print(f"   Rebates: {', '.join(rec.rebates_available)}")
    print()