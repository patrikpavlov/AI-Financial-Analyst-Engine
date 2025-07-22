import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class EndpointHandler:
    def __init__(self, path=""):
        """
        Initializes the model and tokenizer.
        """
        model_name = "patrikpavlov/llama-finance-sentiment"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

    def __call__(self, data: dict) -> list:
        """
        Handles an incoming request, runs inference, and returns the response.
        """
        # Get the prompt from the input data
        inputs = data.pop("inputs", "")
        if not inputs:
            return [{"error": "Input 'inputs' is required."}]

        # Get parameters or use defaults
        parameters = data.pop("parameters", {})
        
        # Tokenize the input text
        input_tokens = self.tokenizer(inputs, return_tensors="pt")
        # Move tokens to the same device as the model
        input_tokens = {k: v.to(self.model.device) for k, v in input_tokens.items()}

        # Generate text
        with torch.no_grad():
            output_tokens = self.model.generate(**input_tokens, **parameters)
        
        # Decode the generated tokens to a string
        generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        return [{"generated_text": generated_text}]