from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class ModelLoader:
    def __init__(
        self, model_id: str = None, model_path: str = None, tokenizer_path: str = None
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._load_model()

    def _load_model(self):
        if self.model_id is None:
            token_path = self.tokenizer_path
            model_path = self.model_path
        else:
            token_path = self.model_id
            model_path = self.model_id

        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Set pad_token_id if not already set
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = max(self.model.config.eos_token_id)
            print(
                f"Set pad_token_id to eos_token_id: {self.tokenizer.pad_token_id}"
            )
        # Important: Set padding to the left for decoder-only models
        self.tokenizer.padding_side = "left"
