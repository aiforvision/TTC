hydra configuration files adapted and modified from


url https://github.com/ashleve/lightning-hydra-template 

# Configuration

This directory contains configurations for the binary learning project.

## Environment Variables

Before running the project, you need to set the following environment variables:

1. `INAT21_DATA_PATH` - Path to the iNat21 dataset
2. `CARDIAC_DATA_PATH` - Path to the cardiac dataset file
3. `WANDB_PROJECT_NAME` - (Optional) Your Weights & Biases project name (defaults to "binary-learning")

Example:
```bash
export INAT21_DATA_PATH="/path/to/iNat21"
export CARDIAC_DATA_PATH="/path/to/cardiac/data.pt"
export WANDB_PROJECT_NAME="your-project-name"
```

You can also create a `.env` file in the project root directory and use an environment loader. 