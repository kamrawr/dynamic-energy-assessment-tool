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

def load_measures_database(file_path: str = "data/measures_template.csv"):
    """Load the measures/recommendations database"""
    global measures_database
    try:
        if Path(file_path).exists():
            measures_database = pd.read_csv(file_path)
            logger.info(f"Loaded {len(measures_database)} measures from database")
        else:
            logger.warning(f"Measures database not found at {file_path}")
            # Create sample data
            create_sample_measures_database()
    except Exception as e:
        logger.error(f"Error loading measures database: {e}")
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
    if assessment.annual_income and assessment.household_size:
        income_bracket = classify_income_bracket(
            assessment.annual_income, 
            assessment.household_size, 
            assessment.county
        )
    
    # Health & Safety first (highest priority)
    if assessment.health_safety:
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
    
    # Insulation recommendations
    if assessment.attic_insulation in ["none", "poor", "fair"]:
        recs = get_measure_recommendations("insulation", "attic", income_bracket, assessment)
        recommendations.extend(recs)
    
    # HVAC recommendations with heat pump logic
    if assessment.heating_type == "electric_resistance" or assessment.cooling_type == "none":
        recs = get_measure_recommendations("hvac", "heat_pump", income_bracket, assessment)
        recommendations.extend(recs)
    
    return sorted(recommendations, key=lambda x: x.priority)

def get_measure_recommendations(category: str, measure_type: str, income_bracket: str, assessment: AssessmentData) -> List[Recommendation]:
    """Get specific recommendations from measures database"""
    global measures_database
    
    if measures_database.empty:
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
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("uploads").mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with assessment form"""
    return templates.TemplateResponse("assessment_form.html", {"request": request})

@app.post("/assess")
async def assess(assessment: AssessmentData):
    """Process assessment and return recommendations"""
    try:
        recommendations = get_targeted_recommendations(assessment)
        
        # Calculate summary statistics
        total_recs = len(recommendations)
        total_cost_low = sum(r.estimated_cost_low or 0 for r in recommendations)
        total_cost_high = sum(r.estimated_cost_high or 0 for r in recommendations)
        total_savings_low = sum(r.annual_savings_low or 0 for r in recommendations)
        total_savings_high = sum(r.annual_savings_high or 0 for r in recommendations)
        
        return {
            "success": True,
            "assessment": assessment.dict(),
            "recommendations": [r.dict() for r in recommendations],
            "summary": {
                "total_recommendations": total_recs,
                "estimated_total_cost_low": total_cost_low,
                "estimated_total_cost_high": total_cost_high,
                "estimated_annual_savings_low": total_savings_low,
                "estimated_annual_savings_high": total_savings_high,
                "estimated_payback_years": total_cost_low / total_savings_low if total_savings_low > 0 else None
            }
        }
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