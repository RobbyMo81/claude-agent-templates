#!/bin/bash
# Helios Python Virtual Environment Setup Script for Unix/Linux/macOS
# This script creates and configures a Python virtual environment for the backend

echo " Helios Python Virtual Environment Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "Python is not available or not in PATH"
    echo "Please install Python from https://python.org"
    echo ""
    exit 1
fi

print_info "Python is available"
$PYTHON_CMD --version

# Check if we're in the correct directory
if [ ! -d "backend" ]; then
    print_error "Backend directory not found"
    echo "Please run this script from the Helios project root directory"
    echo ""
    exit 1
fi

print_success "Backend directory found"
echo ""

# Check if virtual environment already exists
if [ -d "backend/venv" ]; then
    print_info "Virtual environment already exists"
    read -p "Do you want to recreate it? (y/N): " choice
    case "$choice" in 
        y|Y ) 
            print_info "Removing existing virtual environment..."
            rm -rf backend/venv
            ;;
        * ) 
            print_info "Using existing virtual environment"
            ;;
    esac
fi

# Create virtual environment if it doesn't exist
if [ ! -d "backend/venv" ]; then
    print_info "Creating Python virtual environment..."
    $PYTHON_CMD -m venv backend/venv
    
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment"
        echo "Make sure you have python3-venv installed:"
        echo "  Ubuntu/Debian: sudo apt install python3-venv"
        echo "  CentOS/RHEL: sudo yum install python3-venv"
        echo "  macOS: pip3 install virtualenv"
        echo ""
        exit 1
    fi
    
    print_success "Virtual environment created successfully"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source backend/venv/bin/activate

if [ $? -ne 0 ]; then
    print_error "Failed to activate virtual environment"
    echo "Try running: source backend/venv/bin/activate"
    echo ""
    exit 1
fi

print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
python -m pip install --upgrade pip

if [ $? -ne 0 ]; then
    print_warning "Failed to upgrade pip, continuing anyway..."
fi

# Install dependencies
if [ -f "backend/requirements.txt" ]; then
    print_info "Installing Python dependencies..."
    pip install -r backend/requirements.txt
    
    if [ $? -ne 0 ]; then
        print_error "Failed to install some dependencies"
        echo "Check the output above for details"
        echo ""
        exit 1
    fi
    
    print_success "Dependencies installed successfully"
else
    print_warning "No requirements.txt found, skipping dependency installation"
fi

# Verify installation
echo ""
print_info "Verifying installation..."
echo -e "${CYAN}Python version:${NC}"
python --version
echo ""
echo -e "${CYAN}Pip version:${NC}"
pip --version
echo ""
echo -e "${CYAN}Installed packages:${NC}"
pip list
echo ""

print_success " Virtual environment setup completed!"
echo ""
echo -e "${CYAN} Next Steps:${NC}"
echo "================"
echo -e "${NC}1. To activate the virtual environment manually:${NC}"
echo -e "   ${YELLOW}source backend/venv/bin/activate${NC}"
echo ""
echo -e "${NC}2. To start the backend server:${NC}"
echo -e "   ${YELLOW}cd backend${NC}"
echo -e "   ${YELLOW}python server.py${NC}"
echo ""
echo -e "${NC}3. To deactivate when done:${NC}"
echo -e "   ${YELLOW}deactivate${NC}"
echo ""
echo -e "${NC}4. Or use npm scripts:${NC}"
echo -e "   ${YELLOW}npm run python:dev    # Start backend${NC}"
echo -e "   ${YELLOW}npm run dev          # Start frontend${NC}"
echo ""

# Create activation shortcut
print_info "Creating activation shortcut..."
cat > activate-venv.sh << 'EOF'
#!/bin/bash
# Helios Virtual Environment Activation Script

echo "Activating Helios Python environment..."
source backend/venv/bin/activate
echo ""
echo "[SUCCESS] Virtual environment activated!"
echo "To start the backend server: cd backend && python server.py"
echo "To deactivate: deactivate"
echo ""
EOF

chmod +x activate-venv.sh
print_success "Created 'activate-venv.sh' for easy activation"

echo ""
read -p "Would you like to test the backend server now? (y/N): " test_server

if [[ $test_server =~ ^[Yy]$ ]]; then
    print_info "Starting backend server test..."
    cd backend
    python server.py
    # Note: The server will run until manually stopped with Ctrl+C
fi
