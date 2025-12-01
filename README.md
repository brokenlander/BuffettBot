# BuffettBot

A fine-tuned language model that distills Warren Buffett's investment philosophy through an optimized pipeline: intelligent preprocessing of historical documents, LoRA fine-tuning using [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) as the base model, and FP8 dynamic quantization for efficient deployment. The model captures both Buffett's deep investment insights and his characteristic communication style from decades of written and spoken wisdom, preserving his unique perspective on business, markets, and long-term value creation.

## Dataset

The training data draws from three primary sources that capture Buffett's investment philosophy and decision-making process.

### 1. Berkshire Hathaway Shareholder Letters (1977-2023)
- Annual letters written by Warren Buffett to Berkshire Hathaway shareholders
- Contains detailed investment rationale, business principles, and market insights

### 2. Berkshire Annual Meeting Q&A Transcripts (1994-2022)
- Transcribed questions and answers from annual shareholder meetings
- Features Buffett's direct responses to shareholder inquiries
- Includes valuable insights on market conditions, investment decisions, and business philosophy

### 3. "The Essays of Warren Buffett: Lessons for Corporate America"
- Curated collection edited by Lawrence Cunningham
- Thematically organized writings that highlight key principles and teachings
- Provides structured context to Buffett's investment and management philosophy

## Data Preprocessing

The preprocessing pipeline ([`DataPreprocessing.ipynb`](DataPreprocessing.ipynb)) transforms source materials into high-quality training data while preserving Buffett's unique insights and communication style.

### Document Processing Strategy

Two optimized chunking strategies are implemented:

#### 1. Numbered Section Strategy
- Splits text based on numbered Q&A sections (e.g., "1.", "2.")
- Designed for meeting transcripts to preserve dialogue context
- Splits long sections at sentence midpoints when exceeding max chunk size

#### 2. Sentence Overlap Strategy
- Uses [spaCy](https://spacy.io/) for sentence-based segmentation with overlap
- Max chunk: 3000 chars, min: 800 chars, 2-sentence overlap
- Optimized for narrative documents like letters and essays

### Content Processing Pipeline

The pipeline uses Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) for sophisticated content processing:

#### 1. Content Validation
Filters content to ensure quality:
- Approves broadly applicable business philosophy and market insights
- Removes transaction specifics, isolated decisions, and raw financial data
- Focuses on enduring principles over temporal details

#### 2. Conversation Generation
Transforms validated content into training pairs:
- Generates 2 substantive questions per chunk about key themes
- Constructs detailed answers (400-600 words) using Buffett's characteristic Q&A style
- Maintains his conversational, accessible speaking voice
- Outputs in ShareGPT format for training compatibility

## Model Training

Fine-tuning ([`Training.ipynb`](Training.ipynb)) performed using [Unsloth's](https://github.com/unslothai/unsloth) optimized implementation:

### Base Model
- [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) base model
- Max sequence length: 8192 tokens
- Full precision training (no 4-bit quantization)
- Custom ChatML template without `<think>` tags for direct responses

### LoRA Configuration
- Rank: 32
- Alpha: 32
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Dropout: 0
- RSLoRA (Rank-Stabilized LoRA) enabled for improved training stability
- Gradient checkpointing: enabled with Unsloth optimization

### Training Parameters
- Batch size: 2 per device
- Gradient accumulation: 4 steps (effective batch size: 8)
- Learning rate: 1e-4
- Weight decay: 0.01
- Scheduler: Cosine with 5% warmup
- Epochs: 2
- Mixed precision: bfloat16
- Optimizer: AdamW (fused implementation)
- Train/validation split: 85/15

### Training Strategy
- **Response-only training**: Loss computed only on assistant responses, not user prompts
- This preserves the model's instruction-following capabilities while teaching Buffett's response style
- Custom chat template uses ChatML format without reasoning tags

## Model Quantization

Post-training quantization ([`Quantization.ipynb`](Quantization.ipynb)) using [LLM-Compressor](https://github.com/vllm-project/llm-compressor):

### Quantization Strategy
- Method: FP8 Dynamic quantization
- Target: All Linear layers except `lm_head`
- No calibration data required
- Single-pass quantization (oneshot)

### Implementation
- Applied to LoRA-merged model
- Maintains model quality with reduced precision
- Optimized for inference deployment
- Weights and tokenizer saved in HuggingFace format

## Models

Models available at HuggingFace Hub:
- Full model: [`andreamoccia/BuffettBot`](https://huggingface.co/andreamoccia/BuffettBot)
- LoRA adapter: [`andreamoccia/BuffettBot-lora`](https://huggingface.co/andreamoccia/BuffettBot-lora)
- Quantized model: [`andreamoccia/BuffettBot-FP8-Dynamic`](https://huggingface.co/andreamoccia/BuffettBot-FP8-Dynamic)

## Environment Setup

The project uses two separate conda environments due to dependency conflicts.

### Training Environment (Python 3.11)

Used for data preprocessing and model training:

```bash
conda create -n unsloth python=3.11 -y
conda activate unsloth

pip install unsloth
pip install anthropic pymupdf spacy datasets trl peft accelerate huggingface_hub
python -m spacy download en_core_web_sm
```

### Quantization Environment (Python 3.12)

Used for FP8 quantization:

```bash
conda create -n llmcompressor python=3.12 -y
conda activate llmcompressor

pip install llmcompressor huggingface_hub
```

## Usage

### Inference Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "andreamoccia/BuffettBot",
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("andreamoccia/BuffettBot")

messages = [{"role": "user", "content": "What makes a great business?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.85, top_p=0.95)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
