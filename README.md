# exploring-phi-4-gguf-with-llama-cpp

This repository contains a Jupyter Notebook (`phi_4_llama-cpp.ipynb`) demonstrating how to load, run, and perform basic performance comparisons for various Phi-4 GGUF (quantized) models using the `llama-cpp-python` library. The primary focus is on a Named Entity Recognition (NER) task with a strict JSON-only output requirement.

## Notebook Contents

The `phi_4_llama-cpp.ipynb` notebook includes:

1. **Dependency Installation:**
   * Installs `llama-cpp-python`.

2. **Model Loading & Inference:**
   * Demonstrates loading three different Phi-4 GGUF models:
     * `microsoft/phi-4-gguf` (filename: `phi-4-q4.gguf`)
     * `unsloth/Phi-4-mini-instruct-GGUF` (filename: `Phi-4-mini-instruct-Q4_K_M.gguf`)
     * `unsloth/phi-4-GGUF` (filename: `phi-4-Q2_K.gguf`)
   * All models are configured to attempt full GPU offloading (`n_gpu_layers=-1`).
   * The `microsoft/phi-4-gguf` model explicitly sets `n_ctx=16384`.

3. **Prompting for JSON NER:**
   * Utilizes a specific **system prompt** to enforce JSON-only output and a defined schema for address extraction:
   ```
   You are a JSON-only response system. Follow these rules absolutely:
   1. ONLY output valid, parseable JSON
   2. NEVER include text before or after the JSON
   3. NEVER include markdown code blocks or formatting
   4. NEVER include explanations
   5. If you can't fulfill a request, return {"error": "error message"}
   6. Output should always be a single JSON object
   
   For address requests, use this format:
   {
     "address": {
       "license": "B1231241",
       "Address": "X City",
       "Sex": "Male",
       "Weight": "X",
       "Height": "X"
     }
   }
   ```
   * A consistent **test instruction** is used for NER extraction from driver's license-like text:
   ```
   extract NER: California DRIVER LICENSe dl 11234568 CLASS C EXP 08/31/2014 END NONE LNCARDHOLDER FNIMA 2570 24TH STREET ANYTOWN, CA 95818 doB 08/31/1977 RSTR NONE 08311977 VETERAN Cordhslde SEX F HGT 5'-05" HAIR BRN WGT 125 lb EYES BRN DD 00/00/0000NNNAN/ANFD/YY ISS 08/31/2009
   ```

4. **Performance Measurement:**
   * Captures and prints the time taken for chat completion (prompt processing + generation) for each model.

## Prerequisites

* Python 3.x
* `pip`
* A C++ compiler and other build tools required by `llama-cpp-python`. Refer to the [official llama-cpp-python installation guide](https://github.com/abetlen/llama-cpp-python#installation) for system-specific requirements, especially if you encounter build issues.
* (Optional but Recommended) NVIDIA GPU with CUDA drivers for GPU offloading.

## Setup

1. **Clone the repository:**
```bash
git clone https://github.com/SakibAhmedShuva/exploring-phi-4-gguf-with-llama-cpp.git
cd exploring-phi-4-gguf-with-llama-cpp
```

2. **Create and activate a virtual environment (recommended):**
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

3. **Install Jupyter:**
```bash
pip install jupyterlab
# or jupyter notebook
```
The main dependency, `llama-cpp-python`, will be installed by the notebook itself.

4. **Run the Jupyter Notebook:**
```bash
jupyter lab phi_4_llama-cpp.ipynb
# or
# jupyter notebook phi_4_llama-cpp.ipynb
```
Execute the cells in the notebook sequentially.

## Models Tested & Initial Observations

| Model Repository | GGUF Filename | Quantization | `n_ctx` (used) | Output Structure (Observed) | Approx. Time (s) | Tokens (Total) | Notes |
| :--------------------------------- | :---------------------------------- | :----------- | :------------- | :----------------------------------------------------------- | :--------------- | :------------- | :---------------------------------- |
| `microsoft/phi-4-gguf` | `phi-4-q4.gguf` | Q4_K_M | 16384 | `{"address": {...}, "additional_info": {...}}` | ~409.94 | 460 | More detailed, follows prompt closely |
| `unsloth/Phi-4-mini-instruct-GGUF` | `Phi-4-mini-instruct-Q4_K_M.gguf` | Q4_K_M | 512 (default) | `{"entities": {"Location": [...], "License": [...] ...}}` | ~187.43 | 511 | Different schema, faster |
| `unsloth/phi-4-GGUF` | `phi-4-Q2_K.gguf` | Q2_K | 512 (default) | `{"address": {...}}` (simpler) | ~349.61 | 364 | Slower than Q4 Mini, simpler output |

**Important Notes:**
* **Model Downloading:** GGUF model files are downloaded automatically by `Llama.from_pretrained()` from Hugging Face Hub. They are cached locally, typically in `~/.cache/huggingface/hub/`.
* **GPU Offloading:** The notebook attempts to offload all layers to the GPU (`n_gpu_layers=-1`). If a compatible GPU is not found or `llama-cpp-python` is not built with GPU support, it will fall back to CPU, which will be significantly slower.
* **Performance Variability:** Inference times are highly dependent on your specific hardware (CPU, GPU, RAM, VRAM), `llama-cpp-python` build flags, and background system load. The times reported are from a single run environment (as seen in the notebook outputs) and should be considered indicative.
* **Output Consistency:** While the system prompt aims for strict JSON, LLMs can sometimes deviate. The observed outputs show that different fine-tunes/quantizations of Phi-4 might interpret the "address request" format part of the prompt differently or prioritize other learned behaviors.
* The `phi-4-Q2_K.gguf` model from `unsloth/phi-4-GGUF` was noted as "Slow for unknown reason" in the notebook during initial testing for its quantization level.

## Potential Future Work

* More rigorous and systematic benchmarking with multiple runs.
* Testing other Phi-4 quantization levels (e.g., Q5_K_M, Q8_0).
* Evaluating different `n_ctx` settings for models where it's not explicitly set.
* Expanding the range of tasks beyond NER.
* Testing different `llama-cpp-python` parameters (e.g., `n_batch`, `n_threads`).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
