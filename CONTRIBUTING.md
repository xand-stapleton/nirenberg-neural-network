## Environment setup
1. Install `uv`:
    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. Navigate to project directory, and download a compatible version of Python:
    ```
    cd ainstein-KW
    uv venv --python 3.11
    ```
3. Activate virtual environment:
    ```
    source .venv/bin/activate
    ```
4. Sync the packages:
   ```
   uv sync
   ```
## Running code
Before running code, remember to source the virtual environment:
    ```
    source .venv/bin/activate
    ```


## Making contributions
- Keep your local `main` branch **clean and up to date**. Never commit directly to `main`.
- Create a **feature branch** from `main` for each feature or fix:
    ```
    git checkout main
    git pull
    git checkout -b <your-initials>/<feature-name>
    ```
- Make your changes, commit, and push:
    ```
    git add .
    git commit -m "Description of changes"
    git push -u origin <your-initials>/<feature-name>
    ```
- Open a Pull Request on GitHub and have someone review your changes before merging into `main`.
