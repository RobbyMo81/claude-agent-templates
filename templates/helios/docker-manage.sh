#!/bin/bash

# Helios Docker Management Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose > /dev/null 2>&1; then
        print_error "Docker Compose is not installed or not in PATH."
        exit 1
    fi
}

# Function to start production environment
start_production() {
    print_status "Starting Helios in production mode..."
    docker-compose up -d
    print_success "Helios is running in production mode"
    print_status "Frontend: http://localhost"
    print_status "Backend API: http://localhost:5001"
}

# Function to start development environment
start_development() {
    print_status "Starting Helios in development mode..."
    docker-compose --profile dev up -d
    print_success "Helios is running in development mode"
    print_status "Frontend: http://localhost:5173"
    print_status "Backend API: http://localhost:5001"
}

# Function to stop services
stop_services() {
    print_status "Stopping Helios services..."
    docker-compose down
    print_success "All services stopped"
}

# Function to view logs
view_logs() {
    service=${1:-""}
    if [[ -z "$service" ]]; then
        print_status "Showing logs for all services..."
        docker-compose logs -f
    else
        print_status "Showing logs for $service..."
        docker-compose logs -f "$service"
    fi
}

# Function to rebuild services
rebuild_services() {
    service=${1:-""}
    if [[ -z "$service" ]]; then
        print_status "Rebuilding all services..."
        docker-compose build --no-cache
    else
        print_status "Rebuilding $service..."
        docker-compose build --no-cache "$service"
    fi
    print_success "Rebuild completed"
}

# Function to show service status
show_status() {
    print_status "Service Status:"
    docker-compose ps
    echo ""
    print_status "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
}

# Function to cleanup
cleanup() {
    print_warning "This will remove all containers, volumes, and images. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v --rmi all
        docker system prune -f
        print_success "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Function to show help
show_help() {
    echo "Helios Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start-prod          Start production environment"
    echo "  start-dev           Start development environment"
    echo "  stop                Stop all services"
    echo "  restart             Restart all services"
    echo "  logs [service]      View logs (optionally for specific service)"
    echo "  status              Show service status and resource usage"
    echo "  rebuild [service]   Rebuild services (optionally specific service)"
    echo "  cleanup             Remove all containers, volumes, and images"
    echo "  health              Check service health"
    echo "  help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start-prod       # Start in production mode"
    echo "  $0 start-dev        # Start in development mode"
    echo "  $0 logs frontend    # View frontend logs"
    echo "  $0 rebuild backend  # Rebuild only backend"
    echo ""
}

# Function to check health
check_health() {
    print_status "Checking service health..."

    # Check if services are running
    if ! docker-compose ps | grep -q "Up"; then
        print_error "No services are running"
        return 1
    fi

    # Check frontend health
    if curl -s http://localhost/ > /dev/null; then
        print_success "Frontend is healthy"
    else
        print_error "Frontend is not responding"
    fi

    # Check backend health
    if curl -s http://localhost:5001/health > /dev/null; then
        print_success "Backend is healthy"
    else
        print_error "Backend is not responding"
    fi
}

# Main script logic
main() {
    # Check prerequisites
    check_docker
    check_docker_compose

    # Parse command
    case "${1:-help}" in
        "start-prod")
            start_production
            ;;
        "start-dev")
            start_development
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            start_production
            ;;
        "logs")
            view_logs "$2"
            ;;
        "status")
            show_status
            ;;
        "rebuild")
            rebuild_services "$2"
            ;;
        "cleanup")
            cleanup
            ;;
        "health")
            check_health
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"
