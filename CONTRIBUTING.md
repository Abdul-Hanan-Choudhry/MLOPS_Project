# Contributing to MLOPS_Project

## Branching Strategy

This project follows a strict branching model for quality control:

```
feature/* ──► dev ──► test ──► master
```

### Branch Descriptions

| Branch | Purpose | Protection Level |
|--------|---------|------------------|
| `master` | Production-ready code | Highest - Requires PR approval + CI pass |
| `test` | Staging/Testing environment | High - Requires PR approval + model validation |
| `dev` | Development integration | Medium - Requires PR approval + tests pass |
| `feature/*` | New features | None - Free to push |

### Workflow

1. **Create a Feature Branch**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   ```

2. **Develop and Commit**
   ```bash
   # Make changes
   git add .
   git commit -m "feat: description of your changes"
   ```

3. **Push and Create PR to `dev`**
   ```bash
   git push origin feature/your-feature-name
   # Create PR from feature/* -> dev on GitHub
   ```

4. **Merge to `test` for Staging**
   - Create PR from `dev` -> `test`
   - CI will run model training and CML report
   - Review CML metrics comparison

5. **Deploy to Production**
   - Create PR from `test` -> `master`
   - CI will build and push Docker image
   - Automatic deployment verification

## Commit Message Convention

Use conventional commits:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting)
- `refactor:` - Code refactoring
- `test:` - Adding or modifying tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add LSTM model for price prediction
fix: resolve data preprocessing null handling
docs: update API documentation
test: add unit tests for transform module
```

## CI/CD Pipeline

### Feature → Dev
- **Triggered**: PR to `dev` branch
- **Checks**:
  - Code linting (black, isort, flake8)
  - Unit tests (pytest)
  - Security scan (bandit)

### Dev → Test
- **Triggered**: PR to `test` branch
- **Checks**:
  - Full test suite
  - Model retraining
  - CML metrics report posted to PR
  - Performance comparison vs baseline

### Test → Master
- **Triggered**: Merge to `master` branch
- **Actions**:
  - Fetch best model from MLflow
  - Build Docker image
  - Push to Docker Hub
  - Deployment verification

## Setting Up Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/Abdul-Hanan-Choudhry/MLOPS_Project.git
   cd MLOPS_Project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Run tests locally**
   ```bash
   pytest tests/ -v
   ```

5. **Run linting locally**
   ```bash
   black src/ tests/ --check
   isort src/ tests/ --check-only
   flake8 src/ tests/
   ```

## Required GitHub Secrets

For CI/CD to work, configure these secrets in GitHub repository settings:

| Secret | Description |
|--------|-------------|
| `DAGSHUB_TOKEN` | DagsHub API token for MLflow |
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub password/token |

## Branch Protection Rules

Configure these in GitHub Settings → Branches:

### `master` branch:
- ✅ Require pull request before merging
- ✅ Require approvals (1+)
- ✅ Require status checks to pass
- ✅ Require branches to be up to date

### `test` branch:
- ✅ Require pull request before merging
- ✅ Require approvals (1+)
- ✅ Require status checks to pass

### `dev` branch:
- ✅ Require pull request before merging
- ✅ Require status checks to pass

## API Documentation

Once deployed, access API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Questions?

Open an issue on GitHub for any questions or suggestions.
