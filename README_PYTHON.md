# 🏠 Advanced Home Energy Assessment Tool (Python Version)

**Dataset-driven energy assessment with income classification, cost matching, and rebate integration.**

## 🎯 Advanced Features

- **📊 Dataset Integration** - Upload your own measures database (CSV/Excel/JSON)
- **💰 Income Classification** - Targeted recommendations based on household income
- **💵 Cost Matching** - Real cost estimates with regional adjustments
- **🎁 Rebate Integration** - Automatic rebate and financing option matching
- **🔬 API-First** - RESTful API for integration with other systems
- **📈 Analytics Ready** - Built for data analysis and reporting

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Access the Application
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## 📊 Dataset Integration

### Upload Your Measures Database

The system accepts CSV, Excel, or JSON files with your measures database.

**Required Columns:**
- `id` - Unique measure identifier
- `category` - Measure category (insulation, hvac, windows, etc.)
- `measure` - Measure name/title
- `cost_low` - Low cost estimate
- `cost_high` - High cost estimate

**Optional Columns:**
- `description` - Detailed description
- `savings_low` - Low annual savings estimate
- `savings_high` - High annual savings estimate
- `diy_friendly` - Boolean for DIY feasibility
- `priority_base` - Base priority score (1-5)
- `prerequisites` - Required conditions
- `rebates_federal` - Federal rebate information
- `rebates_state` - State rebate information
- `rebates_utility` - Utility rebate information

### Sample Dataset
See `data/measures_template.csv` for a complete example with real measures and costs.

### API Endpoints

#### Upload Dataset
```bash
curl -X POST "http://localhost:8000/upload_dataset" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_measures.csv"
```

#### Get Current Dataset
```bash
curl -X GET "http://localhost:8000/api/measures"
```

#### Submit Assessment
```bash
curl -X POST "http://localhost:8000/assess" \
     -H "Content-Type: application/json" \
     -d '{
       "housing_type": "single_family",
       "attic_insulation": "poor",
       "annual_income": "50k_75k",
       "household_size": 3
     }'
```

## 💰 Income Classification System

The system automatically classifies households into income brackets for targeted recommendations:

### Income Brackets
- **Low Income** - Adjusted income < $30,000
- **Moderate Income** - Adjusted income $30,000 - $60,000  
- **Above Moderate Income** - Adjusted income > $60,000

### Income-Based Adjustments
- **Low Income**: 
  - Cost multiplier: 0.7 (more assistance assumed)
  - Rebates: Weatherization Assistance Program, LIHEAP
  - Financing: On-bill financing, PACE
  
- **Moderate Income**:
  - Cost multiplier: 1.0
  - Rebates: Utility rebates, State tax credits
  - Financing: PACE financing, Utility programs
  
- **Above Moderate Income**:
  - Cost multiplier: 1.0  
  - Rebates: Federal tax credits, Utility rebates
  - Financing: Traditional loans, HELOC

## 🏗️ Architecture

```
dynamic-energy-assessment-tool/
├── app.py                      # FastAPI application
├── requirements.txt            # Python dependencies
├── data/
│   └── measures_template.csv   # Sample measures database
├── uploads/                    # Uploaded datasets
├── templates/                  # HTML templates (if needed)
└── static/                     # Static files (if needed)
```

### Key Components

- **FastAPI Framework** - Modern, fast web API framework
- **Pydantic Models** - Data validation and serialization
- **Pandas Integration** - Dataset manipulation and analysis
- **Income Classifier** - Household income bracket classification
- **Recommendation Engine** - Dataset-driven recommendation matching
- **File Upload System** - Support for CSV, Excel, JSON datasets

## 🔧 Customization

### Adding New Assessment Fields
Update the `AssessmentData` model in `app.py`:

```python
class AssessmentData(BaseModel):
    # Add your new fields
    solar_potential: Optional[str] = None
    electric_vehicle: Optional[bool] = None
    # ... existing fields
```

### Custom Income Classification
Modify the `classify_income_bracket()` function to match your regional Area Median Income (AMI) thresholds.

### Regional Cost Adjustments
Extend the recommendation engine to include regional cost multipliers based on ZIP code or location data.

## 📈 API Response Format

```json
{
  "success": true,
  "assessment": { /* input data */ },
  "recommendations": [
    {
      "id": "INS_001",
      "category": "insulation", 
      "priority": 1,
      "title": "Attic Insulation Upgrade",
      "description": "Add or upgrade attic insulation to R-49",
      "estimated_cost_low": 1500,
      "estimated_cost_high": 3000,
      "annual_savings_low": 200,
      "annual_savings_high": 400,
      "payback_period": 7.5,
      "rebates_available": ["Tax Credit 30%", "Utility rebates"],
      "financing_options": ["PACE financing"],
      "diy_friendly": false,
      "contractor_required": true
    }
  ],
  "summary": {
    "total_recommendations": 5,
    "estimated_total_cost_low": 12000,
    "estimated_total_cost_high": 22000,
    "estimated_annual_savings_low": 1200,
    "estimated_annual_savings_high": 2200,
    "estimated_payback_years": 10
  }
}
```

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
- `DATABASE_URL` - Database connection string (if using database)
- `UPLOAD_DIR` - Custom upload directory
- `LOG_LEVEL` - Logging level (INFO, DEBUG, etc.)

## 🤝 Contributing

This version is designed for organizations needing:
- Custom measure databases
- Income-qualified program delivery  
- Regional cost adjustments
- API integration capabilities
- Advanced analytics and reporting

---

**Ready to integrate your energy efficiency datasets and deliver targeted recommendations!**