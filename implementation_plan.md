<userMessage>
I have decided I want to migrate my code to follow these good principles. Packaging with pyproject, a venv and the uv tool. I want to organize my code into modules, add testing, add the ruff linter, documentation (I already have git) and set up different logging levels. I also want to standardize a code style, and summarize the use of each module of code into a .cursorrules file so AI code ediitng tools can quickly get the important context on the project. Can you help me determine what order I should do these things in?
</userMessage>

<assistantMessage>
Okay, migrating to a modern setup is a great goal! Doing it step-by-step will make it manageable. Here's a logical order to tackle these improvements, building foundational pieces first:
1 Environment Setup (venv + uv):
Action: Create a virtual environment (python -m venv .venv), activate it, and install uv (pip install uv). uv is a fast, modern replacement for pip and pip-tools.
Why: This isolates your project dependencies immediately, ensuring subsequent steps work in a clean, controlled environment. uv will be used for all package management going forward.
2 Core Packaging (pyproject.toml):
Action: Create a pyproject.toml file. Define the build backend (e.g., setuptools) in [build-system]. Move basic project metadata (name, version, author, description) from setup.py to the [project] table. List your core runtime dependencies (from requirements.txt) under [project.dependencies].
Why: This establishes the modern configuration standard. It tells Python and uv how your project is structured and what it needs to run.
3 Install Dependencies with uv:
Action: Run uv pip install -e . (or uv sync if you also list dev dependencies under [project.optional-dependencies]). This installs your package in editable mode and its dependencies using the new pyproject.toml.
Why: Verifies that your basic pyproject.toml setup is correct and that your core code is installable within the venv.
4 Linter/Formatter Setup (ruff):
Action: Add ruff to your development dependencies (e.g., in pyproject.toml under [project.optional-dependencies], group name dev). Install it (uv pip install -e ".[dev]" or uv pip install ruff). Configure ruff within pyproject.toml under [tool.ruff] and [tool.ruff.format]. Run uv run ruff format . and uv run ruff check --fix . to apply initial formatting and linting fixes.
Why: Enforces code style and catches potential errors early. Doing this before major refactoring means you're working with cleaner code and establishing the style standard. uv run ... executes commands using the environment's installed packages.
5 Code Organization (Modularity):
Action: Refactor the code within src/hep_foundation. Break down large files or related functions/classes into smaller, more focused modules (.py files) or sub-packages (directories with __init__.py). Aim for logical grouping based on functionality (e.g., data loading, analysis steps, utilities).
Why: This is the core structural improvement. It enhances readability, maintainability, and makes testing much easier.
6 Testing Setup (pytest):
Action: Add pytest as a development dependency and install it. Create a top-level tests/ directory. Configure pytest (e.g., under [tool.pytest.ini_options] in pyproject.toml). Start writing basic tests for your newly organized modules.
Why: Tests verify that your refactoring works and prevent future regressions. Test structure often mirrors the src structure.
7 Refine Logging:
Action: Review your logging_config.py and how logging is used throughout the refactored code. Ensure consistent use of levels (INFO for progress, DEBUG for details, WARNING/ERROR for issues). Standardize formatting if needed.
Why: Good logging is crucial for debugging and understanding program flow, especially in research code. This step ensures it aligns with the new structure.
8 Documentation (Docstrings & Structure):
Action: Write clear docstrings (e.g., Google or NumPy style) for all public modules, classes, and functions in your refactored code. If desired, set up a documentation generator like Sphinx or MkDocs (add as dev dependencies, create config files, e.g., docs/conf.py or mkdocs.yml). Update your main README.md.
Why: Makes the code understandable. Docstrings are the primary source for auto-generated documentation.
9 Create .cursorrules:
Action: Create a .cursorrules file in the root of your project. Summarize the purpose of each main module/sub-package you created during the organization step.
Why: Provides context for AI tools about your project's structure and intent, improving the quality of AI-assisted coding. This is best done after the structure is finalized.
10 Cleanup:
Action: Remove the old setup.py (if all its info is migrated to pyproject.toml). Remove requirements.txt if dependencies are fully managed by pyproject.toml. Ensure your .gitignore is up-to-date (ignoring .venv/, __pycache__/, .ruff_cache/, *.egg-info/, build artifacts, etc.).
Why: Keeps the project repository clean and avoids confusion between old and new configuration methods.
This order starts with the essential environment and packaging, applies automated style/linting, performs the core code reorganization, adds tests to validate it, and finishes with documentation and tool-specific configurations.
</assistantMessage>

<instructions>
Follow the implementation plan to help refactor our code. Use the scratch pad to write down what we have completed so far. After each step, update the readme and scratchpad. Then update the cursorrules if necessary.
</instructions>

<scratchpad>
Completed step 1 and updated readme.
Completed step 2 and updated readme.
Completed step 3 and updated readme.
Completed step 4 (Ruff setup & fixes): Added ruff as dev dependency, configured in pyproject.toml, ran initial format. Fixed `E721` (type check) and `F821` (undefined name) errors reported by `ruff check`. Updated README with Ruff usage.
</scratchpad>