.PHONY: help dev test lint format clean build deploy

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

## help: Show this help message
help:
	@echo "${BLUE}Agent Army - Development Commands${NC}"
	@echo ""
	@grep -E '^## .*:' $(MAKEFILE_LIST) | sed 's/^## /  /' | column -t -s ':'

## dev: Start local development environment
dev:
	@echo "${GREEN}Starting local development environment...${NC}"
	docker-compose up --build

## dev-bg: Start development environment in background
dev-bg:
	@echo "${GREEN}Starting development environment in background...${NC}"
	docker-compose up -d --build
	@echo "${GREEN}Services started. Access:${NC}"
	@echo "  - API: http://localhost:8000"
	@echo "  - Docs: http://localhost:8000/docs"
	@echo "  - Temporal UI: http://localhost:8080"
	@echo "  - Database UI: http://localhost:8081"

## logs: Tail application logs
logs:
	docker-compose logs -f app

## stop: Stop all services
stop:
	@echo "${YELLOW}Stopping all services...${NC}"
	docker-compose down

## clean: Stop services and remove volumes (fresh start)
clean:
	@echo "${YELLOW}Cleaning up all services and volumes...${NC}"
	docker-compose down -v
	rm -rf .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

## test: Run test suite
test:
	@echo "${GREEN}Running tests...${NC}"
	docker-compose exec app pytest -v

## test-cov: Run tests with coverage report
test-cov:
	@echo "${GREEN}Running tests with coverage...${NC}"
	docker-compose exec app pytest --cov=src --cov-report=html --cov-report=term -v
	@echo "${GREEN}Coverage report generated: htmlcov/index.html${NC}"

## lint: Run linters (black, isort, flake8, mypy)
lint:
	@echo "${GREEN}Running linters...${NC}"
	docker-compose exec app black --check src tests
	docker-compose exec app isort --check-only src tests
	docker-compose exec app flake8 src tests
	docker-compose exec app mypy src

## format: Auto-format code (black, isort)
format:
	@echo "${GREEN}Formatting code...${NC}"
	docker-compose exec app black src tests
	docker-compose exec app isort src tests

## shell: Open Python shell in app container
shell:
	@echo "${GREEN}Opening Python shell...${NC}"
	docker-compose exec app python

## db-shell: Open PostgreSQL shell
db-shell:
	@echo "${GREEN}Opening database shell...${NC}"
	docker-compose exec db psql -U agent_army

## db-migrate: Run database migrations
db-migrate:
	@echo "${GREEN}Running database migrations...${NC}"
	docker-compose exec app alembic upgrade head

## db-reset: Reset database (WARNING: deletes all data)
db-reset:
	@echo "${YELLOW}Resetting database...${NC}"
	docker-compose down -v
	docker-compose up -d db
	@echo "${GREEN}Database reset complete${NC}"

## redis-shell: Open Redis CLI
redis-shell:
	@echo "${GREEN}Opening Redis CLI...${NC}"
	docker-compose exec redis redis-cli

## build: Build Docker image
build:
	@echo "${GREEN}Building Docker image...${NC}"
	docker build -t agent-army:latest .

## push: Push Docker image to registry
push:
	@echo "${GREEN}Pushing Docker image...${NC}"
	docker tag agent-army:latest ghcr.io/robbym081/claude-agent-templates:latest
	docker push ghcr.io/robbym081/claude-agent-templates:latest

## deploy-staging: Deploy to staging environment
deploy-staging:
	@echo "${GREEN}Deploying to staging...${NC}"
	gcloud run deploy agent-army-staging \
		--image ghcr.io/robbym081/claude-agent-templates:latest \
		--platform managed \
		--region us-central1 \
		--allow-unauthenticated \
		--set-env-vars "APP_ENV=staging"

## deploy-prod: Deploy to production (requires confirmation)
deploy-prod:
	@echo "${YELLOW}WARNING: Deploying to PRODUCTION${NC}"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		gcloud run deploy agent-army \
			--image ghcr.io/robbym081/claude-agent-templates:latest \
			--platform managed \
			--region us-central1 \
			--allow-unauthenticated \
			--set-env-vars "APP_ENV=production"; \
	fi

## install: Install dependencies with Poetry
install:
	@echo "${GREEN}Installing dependencies...${NC}"
	poetry install

## update: Update dependencies
update:
	@echo "${GREEN}Updating dependencies...${NC}"
	poetry update

## security-scan: Run security vulnerability scan
security-scan:
	@echo "${GREEN}Running security scan...${NC}"
	docker-compose exec app bandit -r src
	docker-compose exec app safety check

## performance-test: Run performance tests
performance-test:
	@echo "${GREEN}Running performance tests...${NC}"
	k6 run tests/performance/load-test.js

## docs: Generate API documentation
docs:
	@echo "${GREEN}Generating API documentation...${NC}"
	docker-compose exec app python -c "from src.main import app; import json; print(json.dumps(app.openapi()))" > api/openapi.json
	@echo "${GREEN}Documentation generated: api/openapi.json${NC}"

## backup-db: Backup database to file
backup-db:
	@echo "${GREEN}Backing up database...${NC}"
	docker-compose exec db pg_dump -U agent_army agent_army > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "${GREEN}Backup complete${NC}"

## restore-db: Restore database from backup (requires BACKUP_FILE variable)
restore-db:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "${YELLOW}Usage: make restore-db BACKUP_FILE=backup_20231119_120000.sql${NC}"; \
		exit 1; \
	fi
	@echo "${YELLOW}Restoring database from $(BACKUP_FILE)...${NC}"
	docker-compose exec -T db psql -U agent_army agent_army < $(BACKUP_FILE)
	@echo "${GREEN}Restore complete${NC}"

## status: Show service status
status:
	@echo "${BLUE}Service Status:${NC}"
	docker-compose ps

## version: Show version information
version:
	@echo "${BLUE}Agent Army Version Information${NC}"
	@echo "Python: $(shell python --version)"
	@echo "Poetry: $(shell poetry --version)"
	@echo "Docker: $(shell docker --version)"
	@echo "Docker Compose: $(shell docker-compose --version)"
