import csv
import os

from reasoning.llm_output import json_2_csv_double_vllm


def create_file_paths(config):
    """Create and return all required file paths"""
    dataset_dir = config["dataset_dir"]
    model_name = config["model_name"]

    paths = {
        "vllm_input": os.path.join(dataset_dir, "Reasoning.json"),
        "vllm_output": os.path.join(dataset_dir, f"vllm_csv_output_{model_name}.csv"),
    }

    # Ensure output directory exists
    os.makedirs(dataset_dir, exist_ok=True)

    return paths


def run_vllm_processing(file_paths, config):
    """Run vLLM processing"""
    for _ in range(config["vllm_retry_attempts"]):
        json_2_csv_double_vllm(
            config["vllm_ports"],
            file_paths["vllm_input"],
            file_paths["vllm_output"],
            config["vllm_base_urls"],
            config["base_models"],
            batch_size=config["vllm_batch_size"],
            max_thread=config["vllm_max_thread"]
        )


def main():
    # Set CSV field size limit
    csv.field_size_limit(100 * 1024 * 1024)

    config = {
        "model_name": "model_name",  # Model name for output files
        "dataset_dir": "dataset_dir",  # Directory for inputs/outputs

        # vLLM distributed inference settings
        "base_models": ["model_name_1", "model_name_2"],  # Models to use
        "vllm_base_urls": ["IP_1", "IP_2"],  # Server addresses
        "vllm_ports": ["port_1", "port_2"],  # Server ports
        "vllm_batch_size": 100,  # Batch size
        "vllm_max_thread": 100,  # Thread count
    }

    # Create file paths
    file_paths = create_file_paths(config)

    run_vllm_processing(file_paths, config)


if __name__ == "__main__":
    main()
