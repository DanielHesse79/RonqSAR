# RonqSAR UI Startup Guide

## Problem Analysis

The original startup attempt failed due to several issues:

1. **Python PATH Issue**: Python wasn't in the system PATH, requiring full path specification
2. **Missing Dependencies**: The UI required additional packages like `loguru`, `sentence-transformers`, and `faiss-cpu`
3. **Streamlit Date Widget Issue**: The UI had a compatibility issue with pandas Timestamp objects in Streamlit sliders
4. **Port Conflict**: The original port 8501 was already in use

## Solutions Implemented

### 1. Fixed Python Path Issues
- Created startup scripts that use the full Python path: `C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe`
- Added proper error checking for Python installation

### 2. Resolved Dependencies
- Updated startup scripts to install all required dependencies:
  - `streamlit` - Web UI framework
  - `pandas` - Data manipulation
  - `numpy` - Numerical computing
  - `plotly` - Interactive visualizations
  - `loguru` - Logging
  - `sentence-transformers` - Text embeddings
  - `faiss-cpu` - Vector similarity search

### 3. Fixed Streamlit Date Widget
- **Problem**: `st.slider` doesn't support pandas Timestamp objects
- **Solution**: Replaced with `st.date_input` and added proper type conversion
- **Files Modified**:
  - `NarrowSearchAI/ui/app.py` - Changed date slider to date input
  - `NarrowSearchAI/src/filters.py` - Added Timestamp conversion for date filtering

### 4. Changed Port Configuration
- Updated all scripts to use port 8502 instead of 8501
- Updated all startup scripts and documentation

## Available Startup Scripts

### 1. `start_ui.bat` (Recommended)
- **Purpose**: Simple, reliable startup for the UI only
- **Features**: 
  - Automatic dependency installation
  - Browser auto-opening
  - Proper error handling
  - Clear status messages

### 2. `simple_startup.bat`
- **Purpose**: Basic UI startup with minimal dependencies
- **Features**: Focuses only on UI components

### 3. `startup.bat` (Full System)
- **Purpose**: Complete system startup including QSAR API
- **Note**: Requires RDKit which may not be available on Windows

### 4. `startup.ps1` (PowerShell Version)
- **Purpose**: Advanced startup with comprehensive logging
- **Features**: Detailed error logging and dependency checking

## How to Start the UI

### Quick Start (Recommended)
```bash
# Double-click or run:
start_ui.bat
```

### Manual Start
```bash
# Navigate to project directory
cd C:\Users\t14\Documents\GitHub\ronQSAR\RonqSAR

# Run the startup script
.\start_ui.bat
```

### Command Line Start
```bash
# Install dependencies
"C:\Users\t14\AppData\Local\Programs\Python\Python313\python.exe" -m pip install streamlit pandas numpy plotly loguru sentence-transformers faiss-cpu

# Start the UI
cd NarrowSearchAI
"C:\Users\t14\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run ui/app.py --server.port 8502
```

## What You'll See

1. **Terminal Output**: Status messages showing startup progress
2. **Browser**: Automatically opens to `http://localhost:8502`
3. **UI Features**:
   - **Filters**: Category, tags, date range, score range
   - **Semantic Search**: Text-based similarity search
   - **Analytics**: Interactive charts and visualizations
   - **Export**: CSV export functionality

## Troubleshooting

### If the browser doesn't open automatically:
- Manually navigate to `http://localhost:8502`
- Wait a few seconds for the server to start

### If you get dependency errors:
- Run: `"C:\Users\t14\AppData\Local\Programs\Python\Python313\python.exe" -m pip install --upgrade streamlit pandas numpy plotly loguru sentence-transformers faiss-cpu`

### If the UI shows errors:
- Check the terminal for error messages
- Ensure all dependencies are installed
- Try refreshing the browser page

### If port 8502 is in use:
- Change the port in the startup script to another port (e.g., 8503)
- Update the browser URL accordingly

## Testing

Run the test script to verify everything works:
```bash
"C:\Users\t14\AppData\Local\Programs\Python\Python313\python.exe" test_ui.py
```

This will test:
- ✓ Import functionality
- ✓ Data loading
- ✓ Filter operations
- ✓ UI components

## Success Indicators

✅ **UI loads without errors**
✅ **All filters work correctly**
✅ **Date picker functions properly**
✅ **Search functionality works**
✅ **Analytics charts display**
✅ **Export functionality works**

## Next Steps

Once the UI is running successfully, you can:
1. Explore the sample data
2. Test the filtering capabilities
3. Try the semantic search features
4. View the analytics dashboard
5. Export filtered results

The UI provides a comprehensive interface for exploring and analyzing structured datasets with both traditional filtering and modern semantic search capabilities.
