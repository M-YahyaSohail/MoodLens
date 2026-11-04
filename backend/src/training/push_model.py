"""Push the trained model weights to the Hugging Face Hub.

Usage (choose one auth method):
    - `uv run huggingface-cli login` once (token is cached afterwards), then:
        uv run python -m src.training.push_model
    - or set the HF_TOKEN environment variable with a write token.

Note: only the model file is uploaded - not the dataset or venv.
"""

import os

from huggingface_hub import create_repo, upload_file

REPO_ID = os.environ.get("MOODLENS_MODEL_REPO", "zeoxd/moodlens")
MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "src", "deployment", "model", "model.pth",
    )
)


def main():
    token = os.environ.get("HF_TOKEN") or None
    create_repo(REPO_ID, repo_type="model", exist_ok=True, token=token)
    upload_file(
        path_or_fileobj=str(MODEL_PATH),
        path_in_repo="model.pth",
        repo_id=REPO_ID,
        repo_type="model",
        token=token,
    )
    print(f"Pushed {MODEL_PATH} -> {REPO_ID}/model.pth")


if __name__ == "__main__":
    main()
