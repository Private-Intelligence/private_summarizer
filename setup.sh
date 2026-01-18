#!/bin/bash
#
# Private Summarizer - Setup & Run Script
# Uses uv for fast, reliable Python environment management
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Private Summarizer${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${BLUE}Installing uv package manager...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source uv
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi

    if ! command -v uv &> /dev/null; then
        echo ""
        echo -e "${RED}Please restart your terminal and run this script again.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}uv installed:${NC} $(uv --version)"
echo ""

# Sync dependencies (creates venv if needed)
echo -e "${BLUE}Setting up environment...${NC}"
uv sync --no-install-project

echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "To start the server:"
echo -e "  ${BLUE}uv run python backend.py${NC}"
echo ""
echo "Then open summarizer.html in your browser."
echo ""
