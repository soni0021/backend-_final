from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import os
import numpy as np
from mangum import Mangum

app = FastAPI(title="NEET Rank Predictor API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, you should restrict this
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
@app.middleware("http")
async def remove_stage_prefix(request: Request, call_next):
    # Remove stage prefix from path if present (like /dev/ or /prod/)
    path = request.url.path
    if path.startswith('/dev/'):
        request.scope["path"] = path.replace('/dev', '', 1)
    elif path.startswith('/prod/'):
        request.scope["path"] = path.replace('/prod', '', 1)
    
    response = await call_next(request)
    return response

# Define separate paths for rank prediction and state datasets
rank_file_path = "./Corrected_Marks_vs_Rank.xlsx"  # Rank vs Marks file
state_data_path = "./cleaned_data"  # Folder where all state Excel files are stored

# Load the rank vs marks dataset
df_rank = pd.read_excel(rank_file_path)
df_rank = df_rank.sort_values(by="Marks").reset_index(drop=True)

# Mapping of state names to actual file names (without .xlsx extension)
STATE_TO_FILE_MAP = {
    "Andhra Pradesh": "andhra_pradesh",
    "Arunachal Pradesh": "arunchal_pradesh",  # Fixed spelling to match actual file
    "Assam": "assam",
    "Bihar": "bihar",
    "Chhattisgarh": "chattisgarh", 
    "Goa": "open",  # Fallback to open for states without specific files
    "Gujarat": "gujrat", 
    "Haryana": "haryana",
    "Himachal Pradesh": "himachal_pradesh", 
    "Jharkhand": "jharkhand",
    "Karnataka": "karnataka",
    "Kerala": "kerela",  
    "Madhya Pradesh": "madhya_pradesh",
    "Maharashtra": "maharashtra",
    "Odisha": "odisha",
    "Punjab": "punjab",
    "Rajasthan": "rajasthan",
    "Tamil Nadu": "tamil_nadu",
    "Telangana": "telangana",
    "Tripura": "tripura",
    "Uttar Pradesh": "uttar_pradesh",
    "Uttarakhand": "uttrakhand", 
    "All India": "all_india",
    "Jammu and Kashmir": "jammu_and_kashmir",
    "Delhi": "delhi",
    "Chandigarh": "chandigarh",
    "Pondicherry": "pondicherry"
}

# List of all Indian states
STATES_LIST = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"]

def predict_rank(marks, category):
    """Predicts NEET rank based on marks & category adjustments."""
    if marks < 0 or marks > 720:
        return {"predicted_rank": 2000000}  # Default worst rank for out-of-range marks

    # Get base predicted rank using interpolation
    predicted_rank = np.interp(marks, df_rank["Marks"], df_rank["Predicted Rank"])
    predicted_rank = int(round(predicted_rank))
    
    # Calculate rank ranges
    lower_bound = int(round(predicted_rank * 0.9))
    upper_bound = int(round(predicted_rank * 1.1))
    
    result = {
        "predicted_rank": predicted_rank,
        "rank_range": [lower_bound, upper_bound],
        "category_rank_range": None
    }
    
    # Category-specific adjustments using the new formula
    if category.lower() not in ["general", "open"]:
        if category.lower() == "obc-ncl" or category.lower() == "obc":
            cat_rank = int(round(0.25 * (predicted_rank ** 1.01)))
        elif category.lower() == "sc":
            cat_rank = int(round(0.01716 * (predicted_rank ** 1.02338)))
        elif category.lower() == "st":
            cat_rank = int(round(0.007465 * (predicted_rank ** 1.04465)))
        else:
            # Use the old adjustment method as fallback for other categories
            category_adjustment = {
                "ews": 1.1,
                "obc": 1.3,
                "sc": 1.6,
                "st": 1.9
            }
            cat_rank = int(predicted_rank * category_adjustment.get(category.lower(), 1.0))
        
        cat_lb = int(round(cat_rank * 0.9))
        cat_ub = int(round(cat_rank * 1.1))
        result["category_rank_range"] = [cat_lb, cat_ub]
        result["predicted_rank"] = cat_rank
    
    return result

# Pydantic models for request validation
class PredictRequest(BaseModel):
    marks: int
    category: str

class StateRequest(BaseModel):
    state: str

class CollegeRequest(BaseModel):
    state: str
    category: str
    round: str
    rank: int

@app.get("/")
def root():
    return {"message": "NEET Rank Predictor API"}

@app.post("/predict", response_model=Dict[str, Any])
def predict(request: PredictRequest):
    """Predict rank based on marks and category."""
    try:
        marks = request.marks
        category = request.category.lower()
        
        if marks is None or category is None:
            raise HTTPException(status_code=400, detail="Missing 'marks' or 'category'")
            
        result = predict_rank(marks, category)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_states", response_model=Dict[str, List[str]])
def get_states():
    """Returns the list of Indian states."""
    return {"states": STATES_LIST}

@app.post("/get_categories", response_model=Dict[str, List[str]])
def get_categories(request: StateRequest):
    """Returns unique categories based on selected state dataset."""
    state = request.state
    
    print(f"Received request for categories with state: {state}")
    
    # Use the mapping to get the correct filename
    file_base = STATE_TO_FILE_MAP.get(state)
    
    # If state not in mapping, try a case-insensitive search as fallback
    if not file_base:
        lowercase_state = state.lower().replace(' ', '_')
        for key, value in STATE_TO_FILE_MAP.items():
            if key.lower().replace(' ', '_') == lowercase_state:
                file_base = value
                break
    
    # If still not found, use open.xlsx as fallback
    if not file_base:
        print(f"State '{state}' not found in mapping, using 'open' as fallback")
        file_base = "open"
    
    file_name = f"{file_base}.xlsx"
    file_path = os.path.join(state_data_path, file_name)
    
    print(f"Looking for file: {file_path}")
    
    # If file still doesn't exist, try the generic open.xlsx file
    if not os.path.exists(file_path):
        print(f"File {file_path} not found, using open.xlsx as fallback")
        file_path = os.path.join(state_data_path, "open.xlsx")
        
        if not os.path.exists(file_path):
            print("Fallback file open.xlsx also not found")
            # Return default categories when file not found
            default_categories = ["Open", "OBC", "SC", "ST", "EWS"]
            return {"categories": default_categories}
    
    try:
        df = pd.read_excel(file_path)
        
        if "category" not in df.columns:
            print(f"Category column not found in {file_path}. Available columns: {df.columns.tolist()}")
            # Return default categories if category column not found
            default_categories = ["Open", "OBC", "SC", "ST", "EWS"]
            return {"categories": default_categories}
        
        unique_categories = df["category"].dropna().unique().tolist()
        print(f"Found categories: {unique_categories}")
        return {"categories": unique_categories}
        
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        # Return default categories on error
        default_categories = ["Open", "OBC", "SC", "ST", "EWS"]
        print(f"Returning default categories: {default_categories}")
        return {"categories": default_categories}

@app.post("/find_colleges")
def find_colleges(request: CollegeRequest):
    """Returns college names, states, and closing ranks based on user selection."""
    state = request.state
    category = request.category
    round_num = request.round
    rank = request.rank
    
    print(f"Finding colleges for state: {state}, category: {category}, round: {round_num}, rank: {rank}")
    
    # All India may have more rounds
    max_rounds = 6 if state == "All India" else 3
    
    # Use the mapping to get the correct filename
    file_base = STATE_TO_FILE_MAP.get(state)
    
    # If state not in mapping, try a case-insensitive search as fallback
    if not file_base:
        lowercase_state = state.lower().replace(' ', '_')
        for key, value in STATE_TO_FILE_MAP.items():
            if key.lower().replace(' ', '_') == lowercase_state:
                file_base = value
                break
    
    # If still not found, use open.xlsx as fallback
    if not file_base:
        print(f"State '{state}' not found in mapping, using 'open' as fallback")
        file_base = "open"
    
    file_name = f"{file_base}.xlsx"
    file_path = os.path.join(state_data_path, file_name)
    
    print(f"Looking for file: {file_path}")
    
    # If file still doesn't exist, try the generic open.xlsx file
    if not os.path.exists(file_path):
        print(f"File {file_path} not found, using open.xlsx as fallback")
        file_path = os.path.join(state_data_path, "open.xlsx")
        
        if not os.path.exists(file_path):
            print("Fallback file open.xlsx also not found")
            raise HTTPException(status_code=400, detail="Dataset not found and no fallback available")
    
    try:
        df = pd.read_excel(file_path)
        print(f"Columns in dataset: {df.columns.tolist()}")
        
        # Try different possible column name formats for rounds
        possible_column_formats = [
            f"cr_2023_{round_num}",  # Standard format
            f"cr_{round_num}",        # Without year
            f"rank_{round_num}",      # Alternative naming
            f"round_{round_num}",     # Another alternative
            f"r{round_num}",          # Shorter version
            f"closing_rank_{round_num}" # Full name
        ]
        
        round_column = None
        for col_format in possible_column_formats:
            if col_format in df.columns:
                round_column = col_format
                print(f"Found round column: {round_column}")
                break
        
        # If not found, try to find any column that might contain the round number
        if not round_column:
            print(f"Standard column formats not found, searching for alternatives")
            alternative_columns = [col for col in df.columns if f"_{round_num}" in col or f"r{round_num}" in col.lower()]
            
            if alternative_columns:
                round_column = alternative_columns[0]
                print(f"Using alternative column: {round_column}")
            elif int(round_num) > 3 and state != "All India":
                # For non-All India, try to use round 3 data if higher rounds aren't available
                print(f"Round {round_num} not found, trying to fall back to round 3")
                for col_format in [f"cr_2023_3", f"cr_3", f"rank_3", f"round_3", f"r3", f"closing_rank_3"]:
                    if col_format in df.columns:
                        round_column = col_format
                        print(f"Using round 3 data instead: {round_column}")
                        break
            
            if not round_column:
                raise HTTPException(status_code=400, detail=f"Round {round_num} data not found in dataset")
        
        # Map common category names to actual values in the dataset
        category_mapping = {
            "general": "Open",
            "open": "Open",
            "obc": "OBC",
            "sc": "SC",
            "st": "ST",
            "ews": "EWS"
        }
        
        # Try to map the category to a known one, or use as is if not in mapping
        mapped_category = category_mapping.get(category.lower(), category)
        
        # Check if category column exists
        if "category" not in df.columns:
            print(f"Category column not found in {file_path}. Available columns: {df.columns.tolist()}")
            # Try to find alternative category column
            alt_category_columns = [col for col in df.columns if "categ" in col.lower() or "type" in col.lower()]
            
            if alt_category_columns:
                category_col = alt_category_columns[0]
                print(f"Using alternative category column: {category_col}")
                filtered_df = df[df[category_col].str.contains(mapped_category, case=False) & (df[round_column] >= rank)]
                filtered_df = filtered_df.sort_values(by=round_column)
            else:
                # If no category column, return all colleges sorted by closing rank
                filtered_df = df.sort_values(by=round_column)
                result = filtered_df.head(30)[["college_name", "state", round_column]].rename(columns={round_column: "closing_rank"}).to_dict(orient="records")
                return {"colleges": result}
        else:
            # Using case-insensitive comparison for category
            try:
                filtered_df = df[df["category"].str.contains(mapped_category, case=False) & (df[round_column] >= rank)]
                filtered_df = filtered_df.sort_values(by=round_column)
            except Exception as e:
                print(f"Error filtering by category: {str(e)}")
                # If filtering fails, return all colleges sorted by closing rank
                filtered_df = df[df[round_column] >= rank].sort_values(by=round_column)
        
        # Make sure college_name and state columns exist, use defaults if not
        columns_to_return = []
        rename_dict = {}
        
        if "college_name" in df.columns:
            columns_to_return.append("college_name")
        elif "name" in df.columns:
            columns_to_return.append("name")
            rename_dict["name"] = "college_name"
        elif "college" in df.columns:
            columns_to_return.append("college")
            rename_dict["college"] = "college_name"
        else:
            # If no college name column, create a generic one
            filtered_df["college_name"] = "College " + filtered_df.index.astype(str)
            columns_to_return.append("college_name")
        
        if "state" in df.columns:
            columns_to_return.append("state")
        else:
            # If no state column, use the selected state
            filtered_df["state"] = state
            columns_to_return.append("state")
        
        columns_to_return.append(round_column)
        rename_dict[round_column] = "closing_rank"
        
        # Limit to 20 colleges max
        result = filtered_df[columns_to_return].rename(columns=rename_dict).head(20).to_dict(orient="records")
        return {"colleges": result}
        
    except Exception as e:
        print(f"Error processing colleges request: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")

# Create Lambda handler
handler = Mangum(app)

#if __name__ == '__main__':
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8081) 