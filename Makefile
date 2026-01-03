.PHONY: help build up down logs clean

help:
	@echo "F1 AI Race Engineer - Docker Commands"
	@echo "======================================"
	@echo "make build    - Build Docker images"
	@echo "make up       - Start all services"
	@echo "make down     - Stop all services"
	@echo "make logs     - View logs"
	@echo "make clean    - Remove everything"

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend: http://localhost:8000"

down:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	docker system prune -f