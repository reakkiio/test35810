# =============================================================================
# Webscout Docker Makefile
# Simplifies Docker operations and development workflows
# Supports enhanced authentication system with no-auth mode
# =============================================================================

# Variables
DOCKER_IMAGE_NAME := webscout-api
DOCKER_TAG := latest
DOCKER_REGISTRY :=
COMPOSE_FILE := docker-compose.yml
CONTAINER_NAME := webscout-api

# Build arguments
BUILD_DATE := $(shell date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
VERSION := $(shell git describe --tags --always 2>/dev/null || echo "dev")

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# Help
# =============================================================================

.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)Webscout Docker Management$(NC)"
	@echo "=========================="
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make build          # Build the Docker image"
	@echo "  make up             # Start services with authentication"
	@echo "  make no-auth        # Start in no-auth mode (development)"
	@echo "  make mongodb        # Start with MongoDB database"
	@echo "  make generate-key   # Generate a test API key"
	@echo "  make logs           # View logs"
	@echo "  make clean          # Clean up everything"

# =============================================================================
# Build Operations
# =============================================================================

.PHONY: build
build: ## Build the Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build \
		--build-arg BUILD_DATE="$(BUILD_DATE)" \
		--build-arg VCS_REF="$(VCS_REF)" \
		--build-arg VERSION="$(VERSION)" \
		-t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		-t $(DOCKER_IMAGE_NAME):$(VCS_REF) \
		.
	@echo "$(GREEN)Build completed!$(NC)"

.PHONY: build-no-cache
build-no-cache: ## Build the Docker image without cache
	@echo "$(GREEN)Building Docker image (no cache)...$(NC)"
	docker build --no-cache \
		--build-arg BUILD_DATE="$(BUILD_DATE)" \
		--build-arg VCS_REF="$(VCS_REF)" \
		--build-arg VERSION="$(VERSION)" \
		-t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		.

.PHONY: build-dev
build-dev: ## Build development image
	@echo "$(GREEN)Building development image...$(NC)"
	docker build \
		--target builder \
		--build-arg BUILD_DATE="$(BUILD_DATE)" \
		--build-arg VCS_REF="$(VCS_REF)" \
		--build-arg VERSION="$(VERSION)-dev" \
		-t $(DOCKER_IMAGE_NAME):dev \
		.

# =============================================================================
# Container Operations
# =============================================================================

.PHONY: up
up: ## Start services with docker-compose
	@echo "$(GREEN)Starting services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Services started!$(NC)"
	@make status

.PHONY: up-build
up-build: ## Build and start services
	@echo "$(GREEN)Building and starting services...$(NC)"
	docker-compose up -d --build

.PHONY: down
down: ## Stop and remove containers
	@echo "$(YELLOW)Stopping services...$(NC)"
	docker-compose down

.PHONY: restart
restart: ## Restart services
	@echo "$(YELLOW)Restarting services...$(NC)"
	docker-compose restart
	@make status

.PHONY: stop
stop: ## Stop services
	@echo "$(YELLOW)Stopping services...$(NC)"
	docker-compose stop

.PHONY: start
start: ## Start stopped services
	@echo "$(GREEN)Starting services...$(NC)"
	docker-compose start
	@make status

# =============================================================================
# Development Operations
# =============================================================================

.PHONY: dev
dev: ## Start development environment
	@echo "$(GREEN)Starting development environment...$(NC)"
	docker-compose --profile development up -d
	@make status

.PHONY: prod
prod: ## Start production environment
	@echo "$(GREEN)Starting production environment...$(NC)"
	docker-compose --profile production up -d
	@make status

.PHONY: nginx
nginx: ## Start with Nginx reverse proxy
	@echo "$(GREEN)Starting with Nginx proxy...$(NC)"
	docker-compose --profile nginx up -d
	@make status

.PHONY: monitoring
monitoring: ## Start with monitoring stack
	@echo "$(GREEN)Starting with monitoring...$(NC)"
	docker-compose --profile monitoring up -d
	@make status

.PHONY: no-auth
no-auth: ## Start in no-auth mode (development/demo)
	@echo "$(GREEN)Starting in no-auth mode...$(NC)"
	docker-compose -f docker-compose.yml -f docker-compose.no-auth.yml up -d
	@make status

.PHONY: mongodb
mongodb: ## Start with MongoDB database
	@echo "$(GREEN)Starting with MongoDB...$(NC)"
	docker-compose --profile mongodb up -d
	@make status

# =============================================================================
# Monitoring and Debugging
# =============================================================================

.PHONY: status
status: ## Show container status
	@echo "$(BLUE)Container Status:$(NC)"
	@docker-compose ps

.PHONY: logs
logs: ## Show logs
	docker-compose logs -f

.PHONY: logs-api
logs-api: ## Show API logs only
	docker-compose logs -f webscout-api

.PHONY: health
health: ## Check health status
	@echo "$(BLUE)Health Check:$(NC)"
	@curl -f http://localhost:$${WEBSCOUT_PORT:-8000}/health > /dev/null 2>&1 && echo "$(GREEN)API is healthy$(NC)" || echo "$(RED)Health check failed$(NC)"

.PHONY: shell
shell: ## Open shell in container
	docker exec -it $(CONTAINER_NAME) bash

.PHONY: shell-root
shell-root: ## Open root shell in container
	docker exec -it --user root $(CONTAINER_NAME) bash

.PHONY: stats
stats: ## Show container resource usage
	docker stats $(CONTAINER_NAME)

.PHONY: inspect
inspect: ## Inspect container configuration
	docker inspect $(CONTAINER_NAME)

# =============================================================================
# Testing
# =============================================================================

.PHONY: test
test: ## Run API tests
	@echo "$(GREEN)Running API tests...$(NC)"
	@curl -s http://localhost:$${WEBSCOUT_PORT:-8000}/v1/models > /dev/null && echo "$(GREEN)API is responding$(NC)" || echo "$(RED)API is not responding$(NC)"

.PHONY: test-health
test-health: ## Test health endpoint
	@echo "$(BLUE)Testing health endpoint...$(NC)"
	@curl -f http://localhost:$${WEBSCOUT_PORT:-8000}/health > /dev/null 2>&1 && echo "$(GREEN)Health check passed$(NC)" || echo "$(RED)Health check failed$(NC)"

.PHONY: test-endpoints
test-endpoints: ## Test all endpoints
	@echo "$(BLUE)Testing endpoints...$(NC)"
	@echo "Root endpoint:"
	@curl -s -o /dev/null -w "  Status: %{http_code}\n" http://localhost:$${WEBSCOUT_PORT:-8000}/
	@echo "Health endpoint:"
	@curl -s -o /dev/null -w "  Status: %{http_code}\n" http://localhost:$${WEBSCOUT_PORT:-8000}/health
	@echo "Models endpoint:"
	@curl -s -o /dev/null -w "  Status: %{http_code}\n" http://localhost:$${WEBSCOUT_PORT:-8000}/v1/models
	@echo "TTI Models endpoint:"
	@curl -s -o /dev/null -w "  Status: %{http_code}\n" http://localhost:$${WEBSCOUT_PORT:-8000}/v1/TTI/models
	@echo "Docs endpoint:"
	@curl -s -o /dev/null -w "  Status: %{http_code}\n" http://localhost:$${WEBSCOUT_PORT:-8000}/docs

.PHONY: test-auth
test-auth: ## Test authentication endpoints
	@echo "$(BLUE)Testing authentication...$(NC)"
	@echo "Generating API key:"
	@curl -s -X POST "http://localhost:$${WEBSCOUT_PORT:-8000}/v1/auth/generate-key" \
		-H "Content-Type: application/json" \
		-d '{"username": "test_user", "telegram_id": "123456789", "name": "Test Key"}' \
		-w "  Status: %{http_code}\n" || echo "  Failed to generate API key"

.PHONY: generate-key
generate-key: ## Generate a test API key
	@echo "$(GREEN)Generating API key...$(NC)"
	@curl -s -X POST "http://localhost:$${WEBSCOUT_PORT:-8000}/v1/auth/generate-key" \
		-H "Content-Type: application/json" \
		-d '{"username": "makefile_user", "telegram_id": "987654321", "name": "Makefile Test Key"}' \
		| python3 -m json.tool 2>/dev/null || echo "Failed to generate API key"

# =============================================================================
# Cleanup Operations
# =============================================================================

.PHONY: clean
clean: ## Clean up containers, images, and volumes
	@echo "$(YELLOW)Cleaning up...$(NC)"
	docker-compose down -v --remove-orphans
	docker image prune -f
	docker volume prune -f
	@echo "$(GREEN)Cleanup completed!$(NC)"

.PHONY: clean-all
clean-all: ## Clean up everything including images
	@echo "$(RED)Cleaning up everything...$(NC)"
	docker-compose down -v --remove-orphans --rmi all
	docker system prune -af --volumes
	@echo "$(GREEN)Complete cleanup finished!$(NC)"

.PHONY: clean-volumes
clean-volumes: ## Clean up volumes only
	@echo "$(YELLOW)Cleaning up volumes...$(NC)"
	docker-compose down -v
	docker volume prune -f

.PHONY: clean-images
clean-images: ## Clean up images
	@echo "$(YELLOW)Cleaning up images...$(NC)"
	docker image prune -af
	docker rmi $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) 2>/dev/null || true

# =============================================================================
# Registry Operations
# =============================================================================

.PHONY: tag
tag: ## Tag image for registry
	@if [ -z "$(DOCKER_REGISTRY)" ]; then \
		echo "$(RED)DOCKER_REGISTRY not set$(NC)"; \
		exit 1; \
	fi
	docker tag $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
	docker tag $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):$(VCS_REF)

.PHONY: push
push: tag ## Push image to registry
	@if [ -z "$(DOCKER_REGISTRY)" ]; then \
		echo "$(RED)DOCKER_REGISTRY not set$(NC)"; \
		exit 1; \
	fi
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):$(VCS_REF)

.PHONY: pull
pull: ## Pull image from registry
	@if [ -z "$(DOCKER_REGISTRY)" ]; then \
		echo "$(RED)DOCKER_REGISTRY not set$(NC)"; \
		exit 1; \
	fi
	docker pull $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

# =============================================================================
# Utility Operations
# =============================================================================

.PHONY: config
config: ## Show docker-compose configuration
	docker-compose config

.PHONY: ps
ps: ## Show running containers
	docker-compose ps

.PHONY: top
top: ## Show running processes in containers
	docker-compose top

.PHONY: images
images: ## Show Docker images
	docker images | grep $(DOCKER_IMAGE_NAME)

.PHONY: version
version: ## Show version information
	@echo "$(BLUE)Version Information:$(NC)"
	@echo "  Build Date: $(BUILD_DATE)"
	@echo "  VCS Ref: $(VCS_REF)"
	@echo "  Version: $(VERSION)"
	@echo "  Docker Image: $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)"

# =============================================================================
# Environment Setup
# =============================================================================

.PHONY: env
env: ## Create example environment file
	@if [ ! -f .env ]; then \
		echo "$(GREEN)Creating .env file...$(NC)"; \
		echo "# Webscout Environment Configuration" > .env; \
		echo "" >> .env; \
		echo "# Server Settings" >> .env; \
		echo "WEBSCOUT_HOST=0.0.0.0" >> .env; \
		echo "WEBSCOUT_PORT=8000" >> .env; \
		echo "WEBSCOUT_WORKERS=1" >> .env; \
		echo "WEBSCOUT_LOG_LEVEL=info" >> .env; \
		echo "WEBSCOUT_DEBUG=false" >> .env; \
		echo "WEBSCOUT_API_TITLE=Webscout OpenAI API" >> .env; \
		echo "WEBSCOUT_API_DESCRIPTION=OpenAI API compatible interface for various LLM providers with enhanced authentication" >> .env; \
		echo "WEBSCOUT_API_VERSION=0.2.0" >> .env; \
		echo "WEBSCOUT_API_DOCS_URL=/docs" >> .env; \
		echo "WEBSCOUT_API_REDOC_URL=/redoc" >> .env; \
		echo "WEBSCOUT_API_OPENAPI_URL=/openapi.json" >> .env; \
		echo "" >> .env; \
		echo "# Authentication Settings" >> .env; \
		echo "WEBSCOUT_NO_AUTH=false" >> .env; \
		echo "WEBSCOUT_NO_RATE_LIMIT=false" >> .env; \
		echo "WEBSCOUT_API_KEY=your-secret-api-key" >> .env; \
		echo "" >> .env; \
		echo "# Database Settings" >> .env; \
		echo "# MONGODB_URL=mongodb://localhost:27017/webscout" >> .env; \
		echo "WEBSCOUT_DATA_DIR=./data" >> .env; \
		echo "" >> .env; \
		echo "# Provider Settings" >> .env; \
		echo "WEBSCOUT_DEFAULT_PROVIDER=ChatGPT" >> .env; \
		echo "WEBSCOUT_BASE_URL=" >> .env; \
		echo "$(GREEN).env file created! Please edit it with your configuration.$(NC)"; \
	else \
		echo "$(YELLOW).env file already exists$(NC)"; \
	fi

# =============================================================================
# Special Targets
# =============================================================================

.PHONY: all
all: build up test ## Build, start, and test

.PHONY: quick-start
quick-start: build up ## Quick start for new users
	@echo "$(GREEN)Quick start completed!$(NC)"
	@echo "$(BLUE)Access the API at: http://localhost:$${WEBSCOUT_PORT:-8000}$(NC)"
	@echo "$(BLUE)View docs at: http://localhost:$${WEBSCOUT_PORT:-8000}/docs$(NC)"
	@make test-endpoints
