#!/usr/bin/env python3
"""
Advanced Home Energy Assessment Tool with Dataset Integration
"""

from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Home Energy Assessment Tool",
    description="Advanced energy assessment with dataset integration, income classification, and cost matching",
    version="2.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Data models
class AssessmentData(BaseModel):
    housing_type: Optional[str] = None
    foundation_type: Optional[str] = None
    roof_condition: Optional[str] = None
    health_safety: List[str] = []
    attic_insulation: Optional[str] = None
    wall_insulation: Optional[str] = None
    crawlspace_insulation: Optional[str] = None
    ductwork_condition: Optional[str] = None
    window_type: Optional[str] = None
    door_sealing: Optional[str] = None
    heating_type: Optional[str] = None
    cooling_type: Optional[str] = None
    heating_system_age: Optional[str] = None
    cooling_system_age: Optional[str] = None
    # New fields for advanced features
    annual_income: Optional[str] = None
    household_size: Optional[int] = None
    county: Optional[str] = None
    zip_code: Optional[str] = None
    utility_bills_summer: Optional[float] = None
    utility_bills_winter: Optional[float] = None
    # Sizing fields for accurate cost calculations
    square_footage: Optional[int] = None
    attic_square_footage: Optional[int] = None
    wall_square_footage: Optional[int] = None
    conditioned_square_footage: Optional[int] = None
    rooms_needing_hvac: Optional[int] = None
    # Utility bill fields for savings calculations
    summer_electric_bill: Optional[float] = None
    winter_electric_bill: Optional[float] = None
    gas_bill: Optional[float] = None

class Recommendation(BaseModel):
    id: str
    category: str
    priority: int
    title: str
    description: str
    estimated_cost_low: Optional[float] = None
    estimated_cost_high: Optional[float] = None
    annual_savings_low: Optional[float] = None
    annual_savings_high: Optional[float] = None
    payback_period: Optional[float] = None
    rebates_available: List[str] = []
    financing_options: List[str] = []
    diy_friendly: bool = False
    contractor_required: bool = True

# Global data storage
measures_database = pd.DataFrame()
income_brackets = {}
regional_data = {}

def load_measures_database():
    """Load the Energy Trust measures database as primary source"""
    global measures_database
    
    # Try to load Energy Trust measures first (most comprehensive)
    energy_trust_path = Path("data/energy_trust_measures.csv")
    if energy_trust_path.exists():
        try:
            # Read with proper header row (row 1, 0-indexed)
            measures_database = pd.read_csv(energy_trust_path, header=1)
            # Clean up any empty columns
            measures_database = measures_database.dropna(axis=1, how='all')
            logger.info(f"Loaded {len(measures_database)} Energy Trust measures from database")
            logger.info(f"Columns: {list(measures_database.columns)}")
            return
        except Exception as e:
            logger.error(f"Error loading Energy Trust measures: {e}")
    
    # Fallback to template measures
    template_path = Path("data/measures_template.csv")
    if template_path.exists():
        try:
            measures_database = pd.read_csv(template_path)
            logger.info(f"Loaded {len(measures_database)} template measures from database")
            return
        except Exception as e:
            logger.error(f"Error loading template measures: {e}")
    
    # Last resort - create sample data
    logger.warning("No measures database found, creating sample data")
    create_sample_measures_database()

def load_oregon_ami_data():
    """Load Oregon AMI data for accurate income classification"""
    global regional_data
    try:
        ami_path = Path("data/oregon_ami_2025.csv")
        if ami_path.exists():
            ami_df = pd.read_csv(ami_path)
            # Create a dictionary for quick lookup by county
            regional_data['oregon_ami'] = {}
            for _, row in ami_df.iterrows():
                county = row['County'].replace(' County', '').lower()
                regional_data['oregon_ami'][county] = {
                    'ami_100': row['AMI_100_4Person'],
                    'ami_60': row['60%'],
                    'ami_80': row['80%'],
                    'ami_120': row['120%'],
                    'ami_150': row['150%']
                }
            logger.info(f"Loaded AMI data for {len(regional_data['oregon_ami'])} Oregon counties")
        
        # Load demographics data
        demo_path = Path("data/oregon_demographics.csv")
        if demo_path.exists():
            demo_df = pd.read_csv(demo_path)
            regional_data['oregon_demographics'] = demo_df.to_dict('records')
            logger.info(f"Loaded demographics data for {len(demo_df)} areas")
            
    except Exception as e:
        logger.error(f"Error loading Oregon AMI data: {e}")

def load_incentives_database():
    """Load Oregon incentives database for real rebate matching"""
    global regional_data
    try:
        incentives_path = Path("data/oregon_incentives.csv")
        if incentives_path.exists():
            incentives_df = pd.read_csv(incentives_path)
            regional_data['oregon_incentives'] = incentives_df.to_dict('records')
            logger.info(f"Loaded {len(incentives_df)} Oregon incentive programs")
        else:
            logger.warning("Oregon incentives database not found")
    except Exception as e:
        logger.error(f"Error loading Oregon incentives database: {e}")

def match_incentives_to_measure(measure_category: str, county: str, income_bracket: str) -> List[Dict]:
    """Match available incentives to a specific measure based on location and income"""
    matched_incentives = []
    
    if not regional_data.get('oregon_incentives'):
        return matched_incentives
    
    for incentive in regional_data['oregon_incentives']:
        # Check if incentive applies to this measure category
        if (incentive['measure_category'] == 'all' or 
            incentive['measure_category'] == measure_category):
            
            # Check income eligibility
            income_req = incentive['income_requirement']
            if (income_req == 'All incomes' or
                (income_req == '≤60% AMI' and income_bracket in ['very_low_income', 'low_income']) or
                (income_req == '≤80% AMI' and income_bracket in ['very_low_income', 'low_income', 'moderate_income'])):
                
                # Check county/territory restrictions
                county_restriction = incentive['county_restriction']
                if (county_restriction == 'All Oregon counties' or
                    county_restriction == 'All counties' or
                    county.lower() in county_restriction.lower()):
                    
                    matched_incentives.append({
                        'program_name': incentive['program_name'],
                        'program_type': incentive['program_type'],
                        'administrator': incentive['administrator'],
                        'incentive_amount': incentive['incentive_amount'],
                        'incentive_type': incentive['incentive_type'],
                        'application_required': incentive['application_required'],
                        'contact_info': incentive['contact_info'],
                        'website': incentive['website'],
                        'notes': incentive['notes']
                    })
    
    return matched_incentives

def create_sample_measures_database():
    """Create sample measures database for testing"""
    global measures_database
    sample_data = {
        'id': ['INS_001', 'INS_002', 'HVAC_001', 'HVAC_002', 'WIN_001'],
        'category': ['insulation', 'insulation', 'hvac', 'hvac', 'windows'],
        'measure': ['Attic Insulation Upgrade', 'Wall Insulation', 'Heat Pump Installation', 'Duct Sealing', 'Window Replacement'],
        'cost_low': [1500, 3000, 8000, 800, 6000],
        'cost_high': [3000, 6000, 15000, 1500, 12000],
        'savings_low': [200, 300, 800, 150, 400],
        'savings_high': [400, 600, 1500, 300, 800],
        'diy_friendly': [False, False, False, True, False],
        'priority_base': [1, 2, 1, 3, 3]
    }
    measures_database = pd.DataFrame(sample_data)
    logger.info("Created sample measures database")

def classify_income_bracket(annual_income: str, household_size: int, county: str = None) -> str:
    """Classify household into income bracket using Oregon AMI data when available"""
    
    # Handle simple income bracket options directly
    if annual_income in ['low_income', 'moderate_income', 'higher_income']:
        # Map to our internal classifications based on correct AMI thresholds
        if annual_income == 'low_income':  # <= 80% AMI
            return 'low_income'
        elif annual_income == 'moderate_income':  # 80-150% AMI
            return 'moderate_income'
        else:  # higher_income > 150% AMI
            return 'higher_income'
    
    income_mapping = {
        "under_25k": 25000,
        "25k_50k": 37500,
        "50k_75k": 62500,
        "75k_100k": 87500,
        "over_100k": 150000
    }
    
    income_value = income_mapping.get(annual_income, 50000)
    
    # Use Oregon AMI data if county is provided and data is available
    if county and regional_data.get('oregon_ami'):
        county_key = county.lower().replace(' county', '').replace(' ', '')
        county_ami = None
        
        # Try to find the county in AMI data
        for ami_county, ami_data in regional_data['oregon_ami'].items():
            if county_key in ami_county.replace(' ', '') or ami_county.replace(' ', '') in county_key:
                county_ami = ami_data
                break
        
        if county_ami:
            # Adjust AMI for household size (HUD adjustment factors)
            household_adjustments = {1: 0.7, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.08, 6: 1.16, 7: 1.24, 8: 1.32}
            adjustment = household_adjustments.get(household_size, 1.0)
            
            adjusted_ami_60 = county_ami['ami_60'] * adjustment
            adjusted_ami_80 = county_ami['ami_80'] * adjustment
            adjusted_ami_120 = county_ami['ami_120'] * adjustment
            
            # Classify based on Oregon AMI thresholds
            if income_value <= adjusted_ami_60:
                return "very_low_income"  # 60% AMI or below
            elif income_value <= adjusted_ami_80:
                return "low_income"  # 80% AMI or below
            elif income_value <= adjusted_ami_120:
                return "moderate_income"  # 120% AMI or below
            else:
                return "above_moderate_income"  # Above 120% AMI
    
    # Fallback to simple calculation if no county AMI data
    adjusted_income = income_value / household_size if household_size > 0 else income_value
    
    if adjusted_income < 30000:
        return "low_income"
    elif adjusted_income < 60000:
        return "moderate_income"
    else:
        return "above_moderate_income"

def get_targeted_recommendations(assessment: AssessmentData) -> List[Recommendation]:
    """Generate recommendations based on assessment data and income classification"""
    recommendations = []
    
    # Classify income if provided
    income_bracket = "moderate_income"  # default
    if assessment.annual_income:
        income_bracket = classify_income_bracket(
            assessment.annual_income, 
            assessment.household_size or 4,  # Use 4 as default household size
            assessment.county
        )
    
    logger.info(f"Income bracket classified as: {income_bracket}")
    logger.info(f"Measures database has {len(measures_database)} records")
    
    # Health & Safety first (highest priority)
    if assessment.health_safety and assessment.health_safety != ['none']:
        for factor in assessment.health_safety:
            if factor == "combustion_appliances":
                recommendations.append(Recommendation(
                    id="SAFETY_001",
                    category="health_safety",
                    priority=1,
                    title="Combustion Safety Testing",
                    description="Professional testing for CO spillage and backdraft before air sealing work",
                    estimated_cost_low=200,
                    estimated_cost_high=500,
                    contractor_required=True,
                    diy_friendly=False
                ))
    
    # Debug measures database
    logger.info(f"Measures database empty: {measures_database.empty}")
    if not measures_database.empty:
        logger.info(f"Measures database columns: {list(measures_database.columns)}")
        logger.info(f"Has 'program' column: {'program' in measures_database.columns}")
    
    # Use Energy Trust measures if available, otherwise fall back to generic recommendations
    if not measures_database.empty and 'program' in measures_database.columns:
        # Using Energy Trust measures database
        logger.info("Using Energy Trust measures database")
        energy_trust_recs = get_energy_trust_recommendations(assessment, income_bracket)
        recommendations.extend(energy_trust_recs)
        logger.info(f"Energy Trust generated {len(energy_trust_recs)} recommendations")
        
        # If no Energy Trust matches found, still provide generic recommendations as backup
        if len(energy_trust_recs) == 0:
            logger.info("No Energy Trust matches found, adding backup generic recommendations")
            # Add a few key generic recommendations based on assessment
            if assessment.heating_type == "electric_resistance":
                recommendations.append(Recommendation(
                    id="HVAC_GENERIC_001",
                    category="hvac",
                    priority=1,
                    title="Heat Pump Installation",
                    description="Replace electric resistance heating with efficient heat pump for major energy savings.",
                    estimated_cost_low=4000,
                    estimated_cost_high=12000,
                    annual_savings_low=800,
                    annual_savings_high=1500,
                    payback_period=6.0,
                    rebates_available=["Energy Trust incentives up to $1,800", "Federal tax credits 30%"],
                    financing_options=["Energy Trust financing", "Federal tax credits"],
                    diy_friendly=False,
                    contractor_required=True
                ))
    else:
        logger.info("Falling back to generic recommendations")
        # Always provide some basic recommendations based on assessment
        
        # Insulation recommendations
        if assessment.attic_insulation in ["none", "poor", "fair"]:
            recommendations.append(Recommendation(
                id="INSUL_001",
                category="insulation",
                priority=2,
                title="Attic Insulation Upgrade",
                description=f"Current attic insulation is {assessment.attic_insulation}. Upgrading to R-38+ can reduce heating/cooling costs.",
                estimated_cost_low=1500,
                estimated_cost_high=3500,
                annual_savings_low=300,
                annual_savings_high=600,
                payback_period=4.0,
                rebates_available=["Energy Trust incentives available"],
                financing_options=["On-bill financing", "PACE loans"],
                diy_friendly=False,
                contractor_required=True
            ))
        
        # Wall insulation
        if assessment.wall_insulation in ["none", "partial"]:
            recommendations.append(Recommendation(
                id="INSUL_002",
                category="insulation",
                priority=2,
                title="Wall Insulation",
                description=f"Wall insulation is {assessment.wall_insulation}. Dense-pack cellulose or spray foam can improve comfort.",
                estimated_cost_low=3000,
                estimated_cost_high=7000,
                annual_savings_low=400,
                annual_savings_high=800,
                payback_period=5.5,
                rebates_available=["Energy Trust incentives available"],
                financing_options=["PACE loans", "Energy loan programs"],
                diy_friendly=False,
                contractor_required=True
            ))
        
        # Heat pump recommendations
        if assessment.heating_type == "electric_resistance":
            recommendations.append(Recommendation(
                id="HVAC_001",
                category="hvac",
                priority=1,
                title="Heat Pump Installation",
                description="Replace electric resistance heating with efficient heat pump for major energy savings.",
                estimated_cost_low=4000,
                estimated_cost_high=12000,
                annual_savings_low=800,
                annual_savings_high=1500,
                payback_period=6.0,
                rebates_available=["Energy Trust incentives up to $1,800", "Federal tax credits 30%"],
                financing_options=["Energy Trust financing", "Federal tax credits"],
                diy_friendly=False,
                contractor_required=True
            ))
        
        # Window upgrades
        if assessment.window_type == "single_pane":
            recommendations.append(Recommendation(
                id="WIN_001",
                category="windows",
                priority=3,
                title="Window Upgrades",
                description="Single-pane windows lose significant energy. Upgrade to ENERGY STAR double/triple-pane.",
                estimated_cost_low=4000,
                estimated_cost_high=12000,
                annual_savings_low=200,
                annual_savings_high=500,
                payback_period=15.0,
                rebates_available=["Energy Trust incentives available"],
                financing_options=["PACE loans", "Traditional financing"],
                diy_friendly=False,
                contractor_required=True
            ))
    
    logger.info(f"Total recommendations generated: {len(recommendations)}")
    return sorted(recommendations, key=lambda x: x.priority)

def get_energy_trust_recommendations(assessment: AssessmentData, income_bracket: str) -> List[Recommendation]:
    """Generate recommendations using Energy Trust measures database"""
    recommendations = []
    
    if measures_database.empty:
        return recommendations
    
    county = assessment.county or 'Unknown'
    
    # Determine program eligibility based on income (corrected thresholds)
    if income_bracket == 'low_income':  # <= 80% AMI - qualifies for no-cost programs
        eligible_programs = ['Community Partner Funding', 'Savings Within Reach', 'Standard']
    elif income_bracket == 'moderate_income':  # 80-150% AMI - qualifies for reduced-cost programs  
        eligible_programs = ['Savings Within Reach', 'Standard']
    else:  # higher_income > 150% AMI - standard programs only
        eligible_programs = ['Standard']
    
    # Heat pump recommendations - select BEST option for income level
    if (assessment.heating_type == 'electric_resistance' or 
        assessment.cooling_type == 'none' or 
        assessment.heating_system_age in ['15-20', '20+']):
        
        logger.info(f"Looking for heat pump measures. Heating type: {assessment.heating_type}")
        logger.info(f"Available programs for {income_bracket}: {eligible_programs}")
        
        heat_pump_measures = measures_database[
            measures_database['measure'].str.contains('Heat Pump', na=False) &
            measures_database['program'].isin(eligible_programs)
        ]
        
        logger.info(f"Found {len(heat_pump_measures)} heat pump measures for programs: {eligible_programs}")
        
        # Select the BEST heat pump option for this income level
        if len(heat_pump_measures) > 0:
            # Priority order: Community Partner Funding > Savings Within Reach > Standard
            # Also prefer ductless for flexibility and lower cost
            best_measure = None
            
            # First try Community Partner Funding (no cost for low income)
            cpf_measures = heat_pump_measures[heat_pump_measures['program'] == 'Community Partner Funding']
            if len(cpf_measures) > 0:
                # Prefer ductless for flexibility
                ductless_cpf = cpf_measures[cpf_measures['measure'].str.contains('Ductless', na=False)]
                best_measure = ductless_cpf.iloc[0] if len(ductless_cpf) > 0 else cpf_measures.iloc[0]
            
            # If no CPF, try Savings Within Reach
            elif 'Savings Within Reach' in eligible_programs:
                swr_measures = heat_pump_measures[heat_pump_measures['program'] == 'Savings Within Reach']
                if len(swr_measures) > 0:
                    ductless_swr = swr_measures[swr_measures['measure'].str.contains('Ductless', na=False)]
                    best_measure = ductless_swr.iloc[0] if len(ductless_swr) > 0 else swr_measures.iloc[0]
            
            # Otherwise use Standard program
            else:
                standard_measures = heat_pump_measures[heat_pump_measures['program'] == 'Standard']
                if len(standard_measures) > 0:
                    ductless_std = standard_measures[standard_measures['measure'].str.contains('Ductless', na=False)]
                    best_measure = ductless_std.iloc[0] if len(ductless_std) > 0 else standard_measures.iloc[0]
            
            if best_measure is not None:
                recommendations.append(create_energy_trust_recommendation(best_measure, assessment, income_bracket))
    
    # Attic Insulation - select BEST option for income level
    if assessment.attic_insulation in ['none', 'poor', 'fair']:
        insulation_measures = measures_database[
            measures_database['measure'].str.contains('Attic Insulation', na=False) &
            measures_database['program'].isin(eligible_programs)
        ]
        
        logger.info(f"Found {len(insulation_measures)} attic insulation measures for programs: {eligible_programs}")
        
        if len(insulation_measures) > 0:
            # Select best program for income level
            best_insulation = None
            if 'Community Partner Funding' in eligible_programs:
                cpf_insulation = insulation_measures[insulation_measures['program'] == 'Community Partner Funding']
                best_insulation = cpf_insulation.iloc[0] if len(cpf_insulation) > 0 else None
            
            if best_insulation is None and 'Savings Within Reach' in eligible_programs:
                swr_insulation = insulation_measures[insulation_measures['program'] == 'Savings Within Reach']
                best_insulation = swr_insulation.iloc[0] if len(swr_insulation) > 0 else None
            
            if best_insulation is None:
                standard_insulation = insulation_measures[insulation_measures['program'] == 'Standard']
                best_insulation = standard_insulation.iloc[0] if len(standard_insulation) > 0 else None
            
            if best_insulation is not None:
                recommendations.append(create_energy_trust_recommendation(best_insulation, assessment, income_bracket))
    
    # Wall Insulation - select BEST option for income level
    if assessment.wall_insulation in ['none', 'partial']:
        wall_measures = measures_database[
            measures_database['measure'].str.contains('Wall Insulation', na=False) &
            measures_database['program'].isin(eligible_programs)
        ]
        
        if len(wall_measures) > 0:
            # Select best program for income level
            best_wall_insulation = None
            if 'Community Partner Funding' in eligible_programs:
                cpf_wall = wall_measures[wall_measures['program'] == 'Community Partner Funding']
                best_wall_insulation = cpf_wall.iloc[0] if len(cpf_wall) > 0 else None
            
            if best_wall_insulation is None and 'Savings Within Reach' in eligible_programs:
                swr_wall = wall_measures[wall_measures['program'] == 'Savings Within Reach']
                best_wall_insulation = swr_wall.iloc[0] if len(swr_wall) > 0 else None
            
            if best_wall_insulation is None:
                standard_wall = wall_measures[wall_measures['program'] == 'Standard']
                best_wall_insulation = standard_wall.iloc[0] if len(standard_wall) > 0 else None
            
            if best_wall_insulation is not None:
                recommendations.append(create_energy_trust_recommendation(best_wall_insulation, assessment, income_bracket))
    
    # Window recommendations - select BEST option for income level
    if assessment.window_type == 'single_pane':
        window_measures = measures_database[
            measures_database['measure'].str.contains('Windows', na=False) &
            measures_database['program'].isin(eligible_programs)
        ]
        
        if len(window_measures) > 0:
            # Select best program for income level
            best_window = None
            if 'Community Partner Funding' in eligible_programs:
                cpf_windows = window_measures[window_measures['program'] == 'Community Partner Funding']
                best_window = cpf_windows.iloc[0] if len(cpf_windows) > 0 else None
            
            if best_window is None and 'Savings Within Reach' in eligible_programs:
                swr_windows = window_measures[window_measures['program'] == 'Savings Within Reach']
                best_window = swr_windows.iloc[0] if len(swr_windows) > 0 else None
            
            if best_window is None:
                standard_windows = window_measures[window_measures['program'] == 'Standard']
                best_window = standard_windows.iloc[0] if len(standard_windows) > 0 else None
            
            if best_window is not None:
                recommendations.append(create_energy_trust_recommendation(best_window, assessment, income_bracket))
    
    return recommendations

def create_energy_trust_recommendation(measure: pd.Series, assessment: AssessmentData, income_bracket: str) -> Recommendation:
    """Create a recommendation object from Energy Trust measure data with actual sizing"""
    
    # Calculate effective cost based on program type and actual sizing
    expected_unit_price_raw = measure.get('expected_unit_price', 0) or 0
    incentive_value_raw = measure.get('incentive_value', 0) or 0
    unit = measure.get('unit', '$')
    
    # Parse expected unit price - handle URL encoding and malformed data
    if isinstance(expected_unit_price_raw, str):
        import urllib.parse
        cleaned_price = urllib.parse.unquote(str(expected_unit_price_raw))
        if '$' in cleaned_price:
            parts = cleaned_price.split('$')
            if len(parts) > 1:
                numeric_part = parts[-1].strip().replace(',', '')
                try:
                    expected_unit_price = float(numeric_part)
                except:
                    expected_unit_price = 0
            else:
                expected_unit_price = 0
        else:
            try:
                expected_unit_price = float(cleaned_price)
            except:
                expected_unit_price = 0
    else:
        try:
            expected_unit_price = float(expected_unit_price_raw)
        except:
            expected_unit_price = 0
    
    # Parse incentive value - handle URL encoding and malformed data
    if isinstance(incentive_value_raw, str):
        # Clean up URL encoding and extract numeric value
        cleaned_value = str(incentive_value_raw)
        # Handle URL encoded data
        import urllib.parse
        cleaned_value = urllib.parse.unquote(cleaned_value)
        # Extract just the numeric part after the last $
        if '$' in cleaned_value:
            # Find the last occurrence of $ and get the number after it
            parts = cleaned_value.split('$')
            if len(parts) > 1:
                numeric_part = parts[-1].strip().replace(',', '')
                try:
                    incentive_per_unit = float(numeric_part)
                except:
                    incentive_per_unit = 0
            else:
                incentive_per_unit = 0
        else:
            try:
                incentive_per_unit = float(cleaned_value)
            except:
                incentive_per_unit = 0
    else:
        try:
            incentive_per_unit = float(incentive_value_raw)
        except:
            incentive_per_unit = 0
    
    # Calculate costs based on measure type and square footage
    measure_name = measure['measure']
    quantity = 1
    
    if 'Attic Insulation' in measure_name and assessment.attic_square_footage:
        quantity = assessment.attic_square_footage
        expected_cost = expected_unit_price * quantity
        total_incentive = incentive_per_unit * quantity
    elif 'Wall Insulation' in measure_name and assessment.wall_square_footage:
        quantity = assessment.wall_square_footage
        expected_cost = expected_unit_price * quantity
        total_incentive = incentive_per_unit * quantity
    elif 'Heat Pump' in measure_name:
        # Use rooms or conditioned space for heat pump sizing
        if assessment.rooms_needing_hvac and 'Ductless' in measure_name:
            quantity = assessment.rooms_needing_hvac
        elif assessment.conditioned_square_footage:
            quantity = max(1, assessment.conditioned_square_footage / 1000)  # Rough sizing
        expected_cost = expected_unit_price * quantity
        total_incentive = incentive_per_unit * quantity
    else:
        # Fixed cost measures
        expected_cost = expected_unit_price
        total_incentive = incentive_per_unit
    
    # Calculate customer cost based on program
    if measure['program'] == 'Community Partner Funding':
        customer_cost_low = 0  # No customer payment
        customer_cost_high = 50  # Minimal for display
    elif measure['program'] == 'Savings Within Reach':
        customer_cost_low = max(0, expected_cost - total_incentive) * 0.2
        customer_cost_high = max(0, expected_cost - total_incentive) * 0.4
    else:  # Standard
        customer_cost_low = max(0, expected_cost - total_incentive)
        customer_cost_high = expected_cost
    
    # Ensure reasonable costs
    if customer_cost_low <= 0:
        customer_cost_low = 0 if measure['program'] == 'Community Partner Funding' else 200
    if customer_cost_high <= customer_cost_low:
        customer_cost_high = customer_cost_low * 1.2 if customer_cost_low > 0 else 100
    
    # Calculate savings based on utility bills and measure type
    annual_savings = calculate_realistic_savings(measure_name, expected_cost, assessment)
    
    # Create rebate description
    if unit == '$/sqft' and quantity > 1:
        rebate_text = f"{measure['program']}: ${incentive_per_unit:.2f}/sqft (${total_incentive:.0f} total)"
    else:
        rebate_text = f"{measure['program']}: ${total_incentive:.0f}"
    
    return Recommendation(
        id=f"ET_{measure.name}",
        category=categorize_energy_trust_measure(measure_name),
        priority=get_measure_priority(measure_name, measure['program']),
        title=f"{measure_name} ({measure['program']})",
        description=f"{measure.get('requirements_summary', 'Energy Trust measure')}",
        estimated_cost_low=customer_cost_low,
        estimated_cost_high=customer_cost_high,
        annual_savings_low=annual_savings * 0.8,
        annual_savings_high=annual_savings * 1.2,
        payback_period=customer_cost_low / annual_savings if annual_savings > 0 else 0,
        rebates_available=[rebate_text],
        financing_options=get_financing_options(measure['program'], income_bracket),
        diy_friendly=False,
        contractor_required=True
    )

def categorize_energy_trust_measure(measure_name: str) -> str:
    """Categorize Energy Trust measures"""
    if 'Heat Pump' in measure_name:
        return 'hvac'
    elif 'Insulation' in measure_name:
        return 'insulation'
    elif 'Windows' in measure_name:
        return 'windows'
    elif 'Thermostat' in measure_name:
        return 'controls'
    elif 'Water Heater' in measure_name:
        return 'water_heating'
    else:
        return 'other'

def get_measure_priority(measure_name: str, program: str) -> int:
    """Assign priority based on measure type and program"""
    if program == 'Community Partner Funding':
        return 1  # Highest priority - no cost
    elif 'Heat Pump' in measure_name:
        return 1  # High impact
    elif 'Attic Insulation' in measure_name:
        return 2  # High ROI
    elif 'Wall Insulation' in measure_name:
        return 2
    else:
        return 3

def calculate_realistic_savings(measure_name: str, cost: float, assessment: AssessmentData) -> float:
    """Calculate realistic annual savings based on actual utility bills and measure type"""
    # Get current annual bill estimate
    current_annual_cost = 0
    if assessment.summer_electric_bill and assessment.winter_electric_bill:
        # Estimate annual electric cost: 3 months each of summer/winter, 6 months average
        avg_bill = (assessment.summer_electric_bill + assessment.winter_electric_bill) / 2
        current_annual_cost = (assessment.summer_electric_bill * 3 + 
                             assessment.winter_electric_bill * 3 + 
                             avg_bill * 6)
    
    # Add gas bill if applicable
    if assessment.gas_bill:
        current_annual_cost += assessment.gas_bill * 12
    
    # If no bill data, fall back to cost-based estimate
    if current_annual_cost == 0:
        return estimate_annual_savings(measure_name, cost)
    
    # Calculate savings percentage based on measure type and current bills
    if 'Heat Pump' in measure_name:
        if assessment.heating_type == 'electric_resistance':
            # Heat pumps can save 30-50% on heating costs for electric resistance replacement
            heating_portion = current_annual_cost * 0.6  # Assume 60% of bill is heating/cooling
            return heating_portion * 0.40  # 40% savings on heating portion
        else:
            return current_annual_cost * 0.15  # 15% overall savings
    
    elif 'Attic Insulation' in measure_name:
        # Good attic insulation can save 10-20% on heating/cooling
        heating_cooling_portion = current_annual_cost * 0.6
        return heating_cooling_portion * 0.15  # 15% savings on HVAC portion
    
    elif 'Wall Insulation' in measure_name:
        # Wall insulation saves 8-15% on heating/cooling
        heating_cooling_portion = current_annual_cost * 0.6
        return heating_cooling_portion * 0.12  # 12% savings on HVAC portion
    
    elif 'Windows' in measure_name:
        # Window upgrades save 5-10% on heating/cooling
        heating_cooling_portion = current_annual_cost * 0.6
        return heating_cooling_portion * 0.08  # 8% savings on HVAC portion
    
    else:
        # Default fallback
        return current_annual_cost * 0.10

def estimate_annual_savings(measure_name: str, cost: float) -> float:
    """Estimate annual energy savings based on measure type (fallback when no bill data)"""
    if 'Ductless Heat Pump' in measure_name:
        return cost * 0.15  # ~15% of cost as annual savings
    elif 'Ducted Heat Pump' in measure_name:
        return cost * 0.12
    elif 'Attic Insulation' in measure_name:
        return cost * 0.20  # Higher ROI for insulation
    elif 'Wall Insulation' in measure_name:
        return cost * 0.18
    elif 'Windows' in measure_name:
        return cost * 0.08
    else:
        return cost * 0.10  # Default 10%

def get_financing_options(program: str, income_bracket: str) -> List[str]:
    """Get financing options based on program and income"""
    if program == 'Community Partner Funding':
        return ['No-cost program', 'Grant funding']
    elif program == 'Savings Within Reach':
        return ['Income-qualified financing', 'On-bill financing', 'PACE financing']
    else:
        if income_bracket in ['very_low_income', 'low_income']:
            return ['Energy Trust rebates', 'Federal tax credits', 'On-bill financing']
        else:
            return ['Energy Trust rebates', 'Federal tax credits', 'Traditional financing']

def get_measure_recommendations(category: str, measure_type: str, income_bracket: str, assessment: AssessmentData) -> List[Recommendation]:
    """Get specific recommendations from measures database"""
    global measures_database
    
    if measures_database.empty:
        return []
    
    # Check if this is Energy Trust format (has 'program' column) or template format (has 'category' column)
    if 'program' in measures_database.columns:
        # Energy Trust format - use get_energy_trust_recommendations instead
        return []
    
    # Template format - check for category column
    if 'category' not in measures_database.columns:
        return []
    
    # Filter measures by category
    relevant_measures = measures_database[measures_database['category'] == category]
    
    recommendations = []
    for _, measure in relevant_measures.iterrows():
        # Adjust costs and financing based on income bracket
        cost_multiplier = 1.0
        financing_options = []
        rebates = []
        
        if income_bracket == "very_low_income":
            rebates = ["Weatherization Assistance Program", "LIHEAP", "Energy Trust incentives", "Community Action Agency programs"]
            financing_options = ["No-cost weatherization", "On-bill financing", "Grant funding"]
            cost_multiplier = 0.1  # Most assistance available - many programs are free
        elif income_bracket == "low_income":
            rebates = ["Weatherization Assistance Program", "LIHEAP", "Energy Trust incentives", "Utility low-income programs"]
            financing_options = ["On-bill financing", "PACE financing", "Heat pump rebates"]
            cost_multiplier = 0.3  # Significant assistance available
        elif income_bracket == "moderate_income":
            rebates = ["Federal tax credits (30%)", "Energy Trust incentives", "Utility rebates", "State tax credits"]
            financing_options = ["PACE financing", "Utility financing", "Energy loan programs"]
            cost_multiplier = 0.7  # Some assistance available
        else:
            rebates = ["Federal tax credits (30%)", "Energy Trust incentives", "Utility rebates"]
            financing_options = ["Traditional loans", "HELOC", "Cash purchase"]
            cost_multiplier = 1.0  # Full cost
        
        # Get real incentives for this measure
        county = assessment.county or 'Unknown'
        real_incentives = match_incentives_to_measure(measure['category'], county, income_bracket)
        
        # Use real incentives if available, otherwise fall back to generic ones
        if real_incentives:
            rebates = [inc['program_name'] + ' (' + inc['incentive_amount'] + ')' for inc in real_incentives]
            financing_options = list(set([inc['program_type'] for inc in real_incentives if inc['program_type'] in ['Grant', 'Rebate', 'Tax Credit']]))
        
        recommendations.append(Recommendation(
            id=measure['id'],
            category=measure['category'],
            priority=measure['priority_base'],
            title=measure['measure'],
            description=f"Recommended {measure['measure'].lower()} based on your assessment",
            estimated_cost_low=measure['cost_low'] * cost_multiplier,
            estimated_cost_high=measure['cost_high'] * cost_multiplier,
            annual_savings_low=measure['savings_low'],
            annual_savings_high=measure['savings_high'],
            payback_period=(measure['cost_low'] * cost_multiplier) / measure['savings_low'] if measure['savings_low'] > 0 else None,
            rebates_available=rebates,
            financing_options=financing_options,
            diy_friendly=measure['diy_friendly'],
            contractor_required=not measure['diy_friendly']
        ))
    
    return recommendations

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    load_measures_database()
    load_oregon_ami_data()
    load_incentives_database()
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("uploads").mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with assessment form"""
    return templates.TemplateResponse("assessment_form.html", {"request": request})

@app.post("/assess", response_class=HTMLResponse)
async def assess(request: Request):
    """Process assessment form submission and render HTML report"""
    try:
        form = await request.form()
        # Extract multi-select health & safety factors
        health_safety_raw = form.getlist("health_safety_factors") or form.getlist("health_safety") or []
        # Build AssessmentData from form values (fallbacks as needed)
        # Handle simple vs detailed income classification
        use_simple_income = form.get("use_simple_income") == "true"
        if use_simple_income:
            # Use simple income bracket directly
            annual_income = form.get("simple_income_bracket")
            household_size = None
            county = None
        else:
            # Use detailed demographics
            annual_income = form.get("annual_income")
            household_size = int(form.get("household_size")) if form.get("household_size") else None
            county = form.get("county")
        
        assessment = AssessmentData(
            housing_type=form.get("housing_type"),
            foundation_type=form.get("foundation_type"),
            roof_condition=form.get("roof_condition"),
            health_safety=list(health_safety_raw),
            attic_insulation=form.get("attic_insulation"),
            wall_insulation=form.get("wall_insulation"),
            crawlspace_insulation=form.get("crawlspace_insulation"),
            ductwork_condition=form.get("ductwork_condition"),
            window_type=form.get("window_type"),
            door_sealing=form.get("door_sealing"),
            heating_type=form.get("heating_type"),
            cooling_type=form.get("cooling_type"),
            heating_system_age=form.get("heating_system_age"),
            cooling_system_age=form.get("cooling_system_age"),
            annual_income=annual_income,
            household_size=household_size,
            county=county,
            zip_code=form.get("zipcode") or form.get("zip_code"),
            utility_bills_summer=float(form.get("utility_bills_summer")) if form.get("utility_bills_summer") else None,
            utility_bills_winter=float(form.get("utility_bills_winter")) if form.get("utility_bills_winter") else None,
            # Sizing fields
            square_footage=int(form.get("square_footage")) if form.get("square_footage") else None,
            attic_square_footage=int(form.get("attic_square_footage")) if form.get("attic_square_footage") else None,
            wall_square_footage=int(form.get("wall_square_footage")) if form.get("wall_square_footage") else None,
            conditioned_square_footage=int(form.get("conditioned_square_footage")) if form.get("conditioned_square_footage") else None,
            rooms_needing_hvac=int(form.get("rooms_needing_hvac")) if form.get("rooms_needing_hvac") else None,
            # Utility bill fields
            summer_electric_bill=float(form.get("summer_electric_bill")) if form.get("summer_electric_bill") else None,
            winter_electric_bill=float(form.get("winter_electric_bill")) if form.get("winter_electric_bill") else None,
            gas_bill=float(form.get("gas_bill")) if form.get("gas_bill") else None,
        )

        recommendations = get_targeted_recommendations(assessment)
        
        # Calculate summary statistics
        total_recs = len(recommendations)
        total_cost_low = sum(r.estimated_cost_low or 0 for r in recommendations)
        total_cost_high = sum(r.estimated_cost_high or 0 for r in recommendations)
        total_savings_low = sum(r.annual_savings_low or 0 for r in recommendations)
        total_savings_high = sum(r.annual_savings_high or 0 for r in recommendations)
        
        summary = {
            "total_recommendations": total_recs,
            "estimated_total_cost_low": total_cost_low,
            "estimated_total_cost_high": total_cost_high,
            "estimated_annual_savings_low": total_savings_low,
            "estimated_annual_savings_high": total_savings_high,
            "estimated_payback_years": (total_cost_low / total_savings_low) if total_savings_low > 0 else None,
        }

        return templates.TemplateResponse(
            "assessment_report.html",
            {
                "request": request,
                "assessment": assessment.dict(),
                "recommendations": [r.dict() for r in recommendations],
                "summary": summary,
            },
        )
    except Exception as e:
        logger.error(f"Error in assessment: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload custom measures dataset"""
    if not file.filename.endswith(('.csv', '.xlsx', '.json')):
        return {"success": False, "error": "File must be CSV, Excel, or JSON format"}
    
    try:
        # Save uploaded file
        file_path = Path(f"uploads/{file.filename}")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load the dataset
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:  # JSON
            df = pd.read_json(file_path)
        
        # Validate required columns
        required_columns = ['id', 'category', 'measure', 'cost_low', 'cost_high']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return {
                "success": False, 
                "error": f"Missing required columns: {missing_columns}",
                "required_columns": required_columns,
                "found_columns": list(df.columns)
            }
        
        # Update global measures database
        global measures_database
        measures_database = df
        logger.info(f"Updated measures database with {len(df)} records from {file.filename}")
        
        return {
            "success": True,
            "message": f"Successfully loaded {len(df)} measures from {file.filename}",
            "columns": list(df.columns),
            "sample": df.head(3).to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/measures")
async def get_measures():
    """Get current measures database"""
    global measures_database
    if measures_database.empty:
        return {"success": False, "error": "No measures database loaded"}
    
    return {
        "success": True,
        "count": len(measures_database),
        "measures": measures_database.to_dict('records')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)