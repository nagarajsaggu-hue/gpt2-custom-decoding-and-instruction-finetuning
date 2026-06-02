import logging
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from helper import *

logger = setup_logging(GROUP)  
set_seed(42)  # For reproducibility, do not change

class CustomDecoder:
    def __init__(self, model_name: str = 'gpt2', device: str = None): 

        self.logger = logging.getLogger(__name__)  

        self.logger.info(f"Initializing CustomDecoder with model: {model_name}")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.logger.info(f"Loaded tokenizer and model: {model_name}")

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        self.model.to(self.device)
        self.model.eval()
        self.logger.info("Model moved to device and set to evaluation mode")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.info("Set pad_token to eos_token")

        self.logger.info(f"CustomDecoder initialized successfully")
        print(f"CustomDecoder initialized with model: {model_name} on device: {self.device}")

    def _sample_token(self, logits: torch.Tensor,
                      temperature: float = 1.0,
                      top_k: int = 0,
                      top_p: float = 0.0,
                      do_sample: bool = True) -> torch.Tensor:
        """
        Samples a token from the logits.
        
        Args:
            logits: Raw logits from the model (batch_size, vocab_size).
            temperature: Controls randomness. 0 means greedy.
            top_k: Filters to top_k most likely tokens. 0 means no top-k filtering.
            top_p: Nucleus sampling. Filters to the smallest set of tokens whose cumulative probability exceeds top_p. 0.0 means no top-p filtering.
            do_sample: If False, forces greedy sampling regardless of other parameters.
        Returns:
            Sampled token IDs (batch_size, 1).
        """
        self.logger.debug(f"Sampling token with params - temp: {temperature}, top_k: {top_k}, top_p: {top_p}, do_sample: {do_sample}")
        
        if not do_sample or temperature == 0:
            self.logger.debug("Using greedy sampling")
            # TODO: Implement greedy sampling
            predicted_token = logits.argmax(dim=-1, keepdim=True)  
            next_token = predicted_token # Your code here
            return next_token

        # TODO: Implement temperature scaling to logits
        logits = torch.div(logits, temperature)  # apply temperature control to logits

        self.logger.debug(f"Applied temperature scaling: {temperature}")

        # TODO: Implement top-k filtering
        if top_k > 0:
            top_k_actual = min(top_k, logits.size(-1)) # Ensure top_k is not larger than vocab size
            if top_k_actual > 0:
                # Your code here
                 top_logits, top_indices = torch.topk(logits, top_k_actual, dim=-1)
                 filtered_logits = torch.full_like(logits, float('-inf'))
                 filtered_logits.scatter_(1, top_indices, top_logits)
                 logits = filtered_logits  # keep only top-k values

                 self.logger.debug(f"Applied top-k filtering with k={top_k_actual}")

        # TODO: Implement top-p (nucleus) filtering
        if 0.0 < top_p < 1.0:
            self.logger.debug(f"Applying nucleus sampling with top_p={top_p}")
            # Your code here 
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = probs.cumsum(dim=-1)

            remove_mask = cumprobs > top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False

            final_mask = remove_mask.scatter(1, sorted_indices, remove_mask)
            logits = logits.masked_fill(final_mask, float('-inf'))
            # hints: 
            # 1. Sort logits and get cumulative probabilities
            # 2. Create a mask for tokens to remove based on cumulative probabilities
            # 3. Apply the mask to logits

            self.logger.debug("Applied nucleus filtering")
            
        # TODO: Implement ancestral sampling
        # Your code here
            logits = torch.clamp(logits, min=-100, max=100)
            final_probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(final_probs, num_samples=1)
                                                                                                                     
            
        return next_token

    def generate(self, prompt: str,
                 max_new_tokens: int = 50,
                 temperature: float = 1.0,
                 top_k: int = 50, 
                 top_p: float = 0.9, 
                 do_sample: bool = True,
                 **kwargs # keep for potential future use or to pass to model.generate if used
                ):
        """
        Generates text from a prompt.
        Args:
            prompt: The initial text to start generation from.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Softmax temperature for sampling.
            top_k: Top-k filtering.
            top_p: Nucleus (top-p) filtering.
            do_sample: Whether to sample or use greedy decoding.
        Returns:
            The generated text string.
        """
        # This method was already implemented for you
        self.logger.info("=== STARTING TEXT GENERATION ===")
        self.logger.info(f"Generation parameters:")
        self.logger.info(f"  Prompt length: {len(prompt)} characters")
        self.logger.info(f"  Max new tokens: {max_new_tokens}")
        self.logger.info(f"  Temperature: {temperature}")
        self.logger.info(f"  Top-k: {top_k}")
        self.logger.info(f"  Top-p: {top_p}")
        self.logger.info(f"  Do sample: {do_sample}")
        self.logger.info(f"  Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        self.model.eval() # Ensure model is in eval mode
        self.logger.debug("Model set to evaluation mode")

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        self.logger.info(f"Input tokenized to {input_ids.shape[1]} tokens")
        generated_ids = input_ids.clone()

        tokens_generated = 0
        early_stop = False

        with torch.no_grad():
            self.logger.info("Starting token-by-token generation...")
            for step in range(max_new_tokens):
                outputs = self.model(input_ids=generated_ids)
                next_token_logits = outputs.logits[:, -1, :] 
                current_do_sample = do_sample and temperature > 0.0 

                next_token = self._sample_token(
                    next_token_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=current_do_sample
                )
                
                generated_ids = torch.cat((generated_ids, next_token), dim=1)
                tokens_generated += 1
                if step % 10 == 0:
                    self.logger.debug(f"Generated {step} tokens so far")

                if next_token.item() == self.tokenizer.eos_token_id:
                    self.logger.info(f"EOS token reached at step {step}")
                    early_stop = True
                    break
        
        self.logger.info(f"Generation completed:")
        self.logger.info(f"  Tokens generated: {tokens_generated}")
        self.logger.info(f"  Early stop (EOS): {early_stop}")
        self.logger.info(f"  Total sequence length: {generated_ids.shape[1]}")
        
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.logger.info(f"Generated text length: {len(generated_text)} characters")
        self.logger.info(f"Generated text preview: {generated_text[:200]}{'...' if len(generated_text) > 200 else ''}")
        self.logger.info("=== TEXT GENERATION COMPLETED ===")
        
        return generated_text


def test_custom_decoder(model_name: str="gpt2", # use your fine-tuned model here or original gpt2  
                        max_new_tokens: int=50, 
                        use_intruction_dataset: bool=False):
    """Test function for CustomDecoder with logging"""
    logger = setup_logging(GROUP)
    logger.info("=== STARTING CUSTOM DECODER TEST ===")
    
    decoder = CustomDecoder(model_name=model_name)
    # decoder = CustomDecoder(model_name=f"./fine_tuned_gpt2_GROUP_{GROUP}") # use your fine-tuned model here
    if not use_intruction_dataset:
        test_prompts = [
            "The future of artificial intelligence is",
            "Once upon a time in a distant galaxy",
            "The key to happiness lies in"
        ]
    else: 
        dataset = load_and_split_dataset("tatsu-lab/alpaca", small=True)
        test_prompts = dataset['test']['text'][:5]

    
    # Generation parameters to test
    test_configs = [
        {"temperature": 0.5}, 
        {"do_sample": False},
        {"top_k": 10},
        {"top_k": 50},
        {"top_p": 0.5},
        {"top_p": 0.95},
        {"temperature": 0.7, "top_k": 50, "top_p": 0.9, "do_sample": True},
        {"temperature": 1.0, "top_k": 40, "top_p": 0.8, "do_sample": True},
        
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts):
        logger.info(f"Testing prompt {i+1}/{len(test_prompts)}: {prompt}")
        
        for j, config in enumerate(test_configs):
            logger.info(f"Using configuration {j+1}/{len(test_configs)}: {config}")
            
            try:
                generated_text = decoder.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    **config
                )
                
                result = {
                    "prompt": prompt,
                    "config": config,
                    "generated": generated_text,
                    "success": True
                }
                results.append(result)
                
                logger.info(f"Generation successful for prompt {i+1}, config {j+1}")
                
            except Exception as e:
                logger.error(f"Generation failed for prompt {i+1}, config {j+1}: {str(e)}")
                result = {
                    "prompt": prompt,
                    "config": config,
                    "error": str(e),
                    "success": False
                }
                results.append(result)
    
    # Save results
    results_file = f"custom_decoder_results_GROUP_{GROUP}.txt"
    with open(results_file, "w") as f:
        f.write("=== CUSTOM DECODER TEST RESULTS ===\n\n")
        for i, result in enumerate(results):
            f.write(f"Test {i+1}:\n")
            f.write(f"Prompt: {result['prompt']}\n")
            f.write(f"Config: {result['config']}\n")
            if result['success']:
                f.write(f"Generated: {result['generated']}\n")
            else:
                f.write(f"Error: {result['error']}\n")
            f.write("-" * 50 + "\n\n")
    
    logger.info(f"Test results saved to {results_file}")
    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Successful: {sum(1 for r in results if r['success'])}")
    logger.info(f"Failed: {sum(1 for r in results if not r['success'])}")
    logger.info("=== CUSTOM DECODER TEST COMPLETED ===")
    
    return results

if __name__ == "__main__":

     
    # TODO: Analyze the results of the test_custom_decoder function on the instruction dataset
    
    # example usage
    test_results = test_custom_decoder(use_intruction_dataset=True)
    
    # or use directly    
    # decoder = CustomDecoder(model_name='gpt2')
    # result = decoder.generate(
    #     prompt="The meaning of life is",
    #     max_new_tokens=50,
    #     temperature=0.8,
    #     top_k=50,
    #     top_p=0.9
    # )
    # print(result)