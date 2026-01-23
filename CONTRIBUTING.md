# Contributing to DataPilot AI Pro

Thank you for your interest in contributing to DataPilot AI Pro! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/DataPilotAI.git
   cd DataPilotAI
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 📋 Contribution Guidelines

### Code Style

- Follow **PEP 8** for Python code
- Use **type hints** for function signatures
- Write **docstrings** for all public functions and classes
- Keep functions focused and under 50 lines where possible

### Commit Messages

Use conventional commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

Example: `feat: add SHAP waterfall plots to explainer agent`

### Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with tests
3. Run tests: `pytest tests/`
4. Push and create a Pull Request
5. Ensure CI passes and request review

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_agents.py
```

## 📁 Project Structure

When adding new features, follow the existing structure:

- `src/agents/` - New agents inherit from `BaseAgent`
- `src/rl_selector/` - RL-related components
- `src/pipelines/` - Pipeline orchestration
- `src/api/` - FastAPI endpoints
- `src/ui/` - Streamlit components

## 🤝 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow

## 📧 Questions?

Open an issue or reach out to the maintainers.

---

Thank you for contributing! 🎉
