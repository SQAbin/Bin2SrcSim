# Bin2SrcSim 🔍💻

**A Novel Approach for Binary Code Similarity Detection Using LLMs**

<p align="center">
    <a href="#"><img src="https://img.shields.io/badge/BCSD-Framework-blue.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/LLM-Powered-green.svg"></a>
</p>

<p align="center">
  <a href="#overview">📖Overview</a> •
  <a href="#key-features">✨Key Features</a> •
  <a href="#prepare-environment">🧪Environment</a> •
  <a href="#quick-start">🚀Quick Start</a> •
  <a href="#prompt">Prompt</a> •
  <a href="#citation">📝Citation</a>
</p>


## 📖Overview

As IoT manufacturers increasingly adopt open-source third-party components to enhance development efficiency, embedded device firmware has inadvertently introduced numerous known vulnerabilities. Binary Code Similarity Detection (BCSD) is essential for identifying vulnerable functions within closed-source embedded device firmware.

Bin2SrcSim addresses two major challenges faced by existing BCSD approaches:
1. The irreversible loss of semantic and structural information during binary code compilation
2. The inability to directly perform similarity detection between binary code and source code

Our framework employs a Large Language Model (LLM) to convert binary code into source code representations, improving similarity detection performance across architectures, compilers, and optimization levels.

## ✨Key Features

- **Source Code Transformation**: Fine-tuned LLM converts assembly code and pseudocode into high-quality source code
- **Enhanced Semantic Recovery**: Supplements and enhances high-level semantic information lost during compilation
- **Cross-Format Detection**: Enables similarity detection between binary code and source code

## 🧪Prepare Environment

Bin2SrcSim is developed on Ubuntu 24.04 LTS. 
Please follow these steps to set up the Python environment:

```bash
conda create -n Bin2SrcSim python=3.11.11
conda activate Bin2SrcSim
pip install -r requirements.txt
```
We recommend using vLLM to launch the model for better multi-threading performance and efficiency.


## 🚀Quick Start

Use the following command to perform distributed inference with vLLM:

```bash
python command.py
```

The script uses a configuration dictionary that controls the distributed inference:

```python
config = {
    "model_name": "model_name",  # Model name for output files
    "dataset_dir": "dataset_dir",                # Directory for inputs/outputs
    
    # vLLM distributed inference settings
    "base_models": ["model_name_1", "model_name_2"],     # Models to use
    "vllm_base_urls": ["IP_1", "IP_2"],  # Server addresses
    "vllm_ports": ["port_1", "port_2"],                      # Server ports 
    "vllm_batch_size": 100,                              # Batch size
    "vllm_max_thread": 100,                              # Thread count
}
```

## 💬Prompt

The LLM prompts in our fine-tuning dataset, which is stored at data/Fine-Tuning.json, follow the format illustrated in the figure below.

```text
Translate the following into a functional C/C++ program:
Assembly Code:
{Asc}
Pseudocode:
{Psc}
Please generate complete, compilable C/C++ source code that implements the same logic.
```

## 📄Data Format

The required input format for llm_output.py is as follows:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Prompt with assembly code and pseudocode"
    },
    {
      "from": "gpt",
      "value": "Generated C/C++ source code implementation"
    }
  ],
  "filepath": "architecture-compiler-optimization-binary_name/function_name"
}
```

The filepath helps with constructing cross-architecture (XA), cross-compiler (XC), cross-optimization (XO), and cross-matching (XM) code pairs for evaluation. When processed, the dataset generates three columns:
- **filepath**: Path indicating origin of the code sample
- **sourceCode**: Original function source code 
- **translateCode**: Source code generated from pseudocode and assembly


## 📝Citation

If you find Bin2SrcSim helpful, please cite our paper.
