
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup, BatchEncoding
from datasets import Dataset
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import os
import numpy as np
import re
from helper import *

logger = setup_logging(GROUP)

# for reproducibility, do NOT change
set_seed(42)

class GPT2Model:
    def __init__(self, model_name="gpt2", max_length=128, device=None):
        """
        Initialize custom GPT-2 model for fine-tuning and text generation.
        
        Args:
            model_name: Pre-trained model name (e.g., "gpt2", "gpt2-medium")
            max_length: Maximum sequence length
            device: Device to use for training
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing GPT2Model with model: {model_name}")
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_length = max_length
        

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.logger.info(f"Loaded model and tokenizer: {model_name}")
                
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.info("Set pad_token to eos_token")
            
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Max length: {max_length}")
        
    
        """
        Tokenize the text data
        
        Args:
            examples: Dataset examples with "text" feature
            
        Returns:
            Tokenized BatchEncoding object
        """
        # TODO: Tokenize the input examples
        # Use tokenizer to tokenize examples["text"]
        # Truncate and pad the sequences to max_length
        # Your code here
        

        #  TODO: For language modeling, labels are the same as input_ids
        # Set labels equal to input_ids for language modeling
        # Your code here

        # TODO: Return the tokenized examples
        # Your code here
    def tokenize_function(self, examples: Dataset) -> BatchEncoding:
            encoded = self.tokenizer(
               examples["text"],
               padding="max_length",
               truncation=True,
               max_length=self.max_length
            )
            encoded["labels"] = encoded["input_ids"].copy()
            return encoded


    
    def prepare_dataset(self, dataset: Dataset, batch_size: int=4) -> DataLoader:
        """
        Prepare the dataset for training
        
        Args:
            dataset: HuggingFace Dataset
            batch_size: Batch size for training
        
        Returns:
            DataLoader object
        """
        # This function was already implemented for you
        self.logger.info(f"Preparing dataset with batch size: {batch_size}")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        self.logger.info("Dataset tokenized successfully")
        
        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.data_collator
        )
        self.logger.info(f"DataLoader created with {len(dataloader)} batches")
        return dataloader
    
    def data_collator(self, batch: list) -> dict:
        """
        Custom data collator to handle batching
        Takes a list of examples and pads them to the maximum length in the batch.
        This is necessary for the model to accept variable-length sequences.
        The attention mask is to indicate which tokens are padding.

        Args:
            batch: List of examples from the dataset
        Returns:
            Dictionary with padded input_ids, attention_mask, and labels
        """
        # This function was already implemented for you
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def train(self, dataset: Dataset, epochs: int=3, 
              batch_size: int=4, learning_rate: float=5e-5, 
              warmup_steps: int=100, save_path: str="./fine_tuned_gpt2"):
        """
        Fine-tune the GPT-2 model
        
        Args:
            dataset: HuggingFace Dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            warmup_steps: Number of warmup steps for scheduler
            save_path: Path to save the fine-tuned model
        """
        self.logger.info("=== STARTING TRAINING ===")
        self.logger.info(f"Training parameters:")
        self.logger.info(f"  Epochs: {epochs}")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Learning rate: {learning_rate}")
        self.logger.info(f"  Warmup steps: {warmup_steps}")
        self.logger.info(f"  Save path: {save_path}")
        self.logger.info(f"  Dataset size: {len(dataset)}")
        
        dataloader = self.prepare_dataset(dataset, batch_size)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(dataloader) * epochs
        self.logger.info(f"Total training steps: {total_steps}")
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # TODO: Complete the training loop
        self.model.train()
        total_loss = 0
        
        # iterate through epochs
        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            # iterate through batches
            for batch_idx, batch in enumerate(progress_bar):
                # (hints)
                # Move batch to device
                # Reset optimizer gradients to 0
                # Pass batch to model
                # Calculate the loss
                # Perform backward pass
                # Clip gradients to prevent exploding gradients
                # Update weights using optimizer
                # Update learning rate using scheduler

                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
    
                epoch_loss += loss.item()

                # tracking the training progress, do not change
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
                })

        
                # Log every 10 batches
                if batch_idx % 10 == 0:
                    self.logger.info(f"Epoch {epoch+1}, Batch {batch_idx}: loss={loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

        avg_total_loss = total_loss/(len(dataloader)*epochs)
        self.logger.info(f"Training completed. Average loss: {avg_total_loss:.4f}")

        self.save_model(save_path)
        
    def save_model(self, save_path: str=f"./fine_tuned_gpt2_GROUP_{GROUP}") -> None:
        """Save the fine-tuned model and tokenizer
        Args:
            save_path: Path to save the model
        Returns:
            None
        """
        # This function was already implemented for you
        self.logger.info(f"Saving model to: {save_path}")
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.logger.info(f"Model saved successfully to {save_path}")
    
    def generate_text(self, prompt, max_length=128):
        """
        Generate text using the fine-tuned model
        
        Tokenizes the input prompt and generates text using the model.
        The generated text is decoded and returned as a list of strings.
        The model is set to evaluation mode to disable dropout and other training-specific behaviors.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            num_return_sequences: Number of sequences to generate
        
        Returns:
            List of generated texts
        """
        # This function was already implemented for you
        self.model.eval()
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts

    
    def _calculate_perplexity(self, test_dataset: Dataset, batch_size: int=4):
        """
        Calculate perplexity on test dataset
        
        Returns:
            Perplexity score (float)
        """

        self.logger.info("Calculating perplexity...")
        self.model.eval()
        
        test_dataloader = self.prepare_dataset(test_dataset, batch_size)
        total_loss = 0
        total_tokens = 0
        # TODO: Complete the method 
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Calculating perplexity")):
                # Your code here
                 batch = {k: v.to(self.device) for k, v in batch.items()}
                 outputs = self.model(**batch)
                 loss = outputs.loss
                 total_loss += loss.item() * batch["input_ids"].size(0)
                 total_tokens += batch["input_ids"].size(0)

        perplexity = np.exp(total_loss / total_tokens)
        return perplexity
    

    def _calculate_repetition_score(self, generated_text):
        """Calculate repetition score 
        (percentage of repeated bi-grams and tri-grams-grams in the generated text)"""
        
        words = set(re.findall(r'\w+', generated_text.lower()))
        if len(words) < 4:
            return 0.0

        # TODO: Implement the method
        # Your code here
        tokens = generated_text.lower().split()
        bigrams = list(zip(tokens, tokens[1:]))
        trigrams = list(zip(tokens, tokens[1:], tokens[2:]))

        rep_bi = len(bigrams) - len(set(bigrams))
        rep_tri = len(trigrams) - len(set(trigrams))
        total = len(bigrams) + len(trigrams)
        return (rep_bi + rep_tri) / total if total > 0 else 0.0


    def _calculate_diversity_score(self, generated_text):
        """Calculate lexical diversity (unique words / total words)"""
        
        words = set(re.findall(r'\w+', generated_text.lower()))
        # TODO: Implement the method
        # Your code here
        words = re.findall(r'\w+', generated_text.lower())
        return len(set(words)) / len(words) if words else 0.0

    
    def _calculate_jaccard_similarity(self, true_text, generated_text):
        """Calculate Jaccard similarity between two texts"""
        true_words = set(re.findall(r'\w+', true_text.lower()))
        generated_words = set(re.findall(r'\w+', generated_text.lower()))
        
        # TODO: Implement the method
        # Your code here
        intersection = true_words.intersection(generated_words)
        union = true_words.union(generated_words)
        return len(intersection) / len(union) if union else 0.0


    def _calculate_generation_metrics(self, test_dataset):
        """Calculate generation quality metrics"""
        # This method was already implemented for you
        self.logger.info("Calculating generation metrics...")
        repetition_scores = []
        diversity_scores = []
        jaccard_scores = []
        
        outputs = []
        for idx, example in enumerate(tqdm(test_dataset, desc="Calculating generation metrics")):
            prompt = example["text"]
            true_output = example['output']
            
            try:
                generated_texts = self.generate_text(
                    prompt=prompt,
                    max_length=self.max_length
                )
                
                if generated_texts:
                    generated = generated_texts[0]
                    outputs.append(generated)
                    repetition = self._calculate_repetition_score(generated)
                    repetition_scores.append(repetition)
                    
                    diversity = self._calculate_diversity_score(generated)
                    diversity_scores.append(diversity)

                    jaccard_similarity = self._calculate_jaccard_similarity(true_output, generated)
                    jaccard_scores.append(jaccard_similarity)
                    
                    if idx % 100 == 0:
                        self.logger.info(f"Processed {idx} examples for generation metrics")
            
            except Exception as e:
                self.logger.warning(f"Error processing example {idx}: {str(e)}")
                continue
        
        outputs_file = f"generated_outputs_{self.model_name}_{GROUP}.txt"
        with open(outputs_file, "w") as f:
            for output in outputs:
                f.write(output + "\n")
        self.logger.info(f"Saved generated outputs to {outputs_file}")
        
        metrics = {
            "avg_repetition_score": np.mean(repetition_scores) if repetition_scores else 0.0,
            "avg_diversity_score": np.mean(diversity_scores) if diversity_scores else 0.0,
            "avg_jaccard_similarity": np.mean(jaccard_scores) if jaccard_scores else 0.0
        }
        self.logger.info(f"Generation metrics calculated: {metrics}")
        return metrics

    def evaluate(self, test_dataset: Dataset, batch_size: int=4):
        """
        Evaluate the model on test dataset
        
        Args:
            test_dataset: HuggingFace Dataset for evaluation
            batch_size: Batch size for evaluation
            max_gen_length: Maximum length for text generation (for BLEU calculation)
            num_samples_bleu: Number of samples to use for BLEU score calculation
        
        Returns:
            Dictionary containing perplexity and BLEU score
        """
        # This method was already implemented for you
        self.logger.info("=== STARTING EVALUATION ===")
        self.logger.info(f"Test dataset size: {len(test_dataset)}")
        self.logger.info(f"Batch size: {batch_size}")
        
        generation_metrics = self._calculate_generation_metrics(test_dataset)
        results = {
            "perplexity": self._calculate_perplexity(test_dataset, batch_size),
            **generation_metrics
        }
        
        self.logger.info("=== EVALUATION RESULTS ===")
        print(f"Evaluation Results:")
        for k, v in results.items():
            self.logger.info(f"{k}: {v:.4f}")
            print(f"{k}: {v:.4f}")    
        
        results_file = f"evaluation_results_{self.model_name}_{GROUP}.txt"
        with open(results_file, "w") as f:
            for k, v in results.items():
                f.write(f"{k}: {v:.4f}\n")
        self.logger.info(f"Evaluation results saved to {results_file}")
        print(f"Evaluation results saved to {results_file}")

        return results


def fine_tune(model_name: str='gpt2',
              dataset: Dataset=None,
              max_length: int=128,
              epochs: int=1,
              batch_size: int=8,
              lr: float=5e-5,
              warmup_steps: int=100,
              ):
    """
    Fine-tune the GPT-2 model on the provided dataset.
    Args:
        model_name: Pre-trained model name (e.g., "gpt2", "gpt2-medium")
        dataset: HuggingFace Dataset for fine-tuning
        max_length: Maximum sequence length for text generation
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate for optimizer
        warmup_steps: Number of warmup steps for scheduler
    """
    logger = setup_logging(GROUP)
    logger.info("=== STARTING FINE-TUNING PROCESS ===")
    logger.info(f"Parameters:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Max length: {max_length}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    
    train_dataset = dataset['train']
    model = GPT2Model(model_name=model_name, max_length=max_length)

    logger.info(f"Fine-tuning {model_name} model...")
    print(f"Fine-tuning {model_name} model...")
    model.train(
        dataset=dataset['train'],
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps
    )
    
    logger.info("Evaluating fine-tuned GPT-2 model...")
    print("Evaluating fine-tuned GPT-2 model...")
    model.evaluate(
            test_dataset=train_dataset,
            batch_size=batch_size
        )    
    logger.info("=== FINE-TUNING PROCESS COMPLETED ===")


def generate_and_evaluate(model_name: str='gpt2',
              max_length: int=128,
              batch_size: int=8,
              small: bool=False,
              ):
    """
    Generate and evaluate the GPT-2 model on the test dataset.
    Args:
        model_name: Pre-trained or fine-tuned model name (e.g., "gpt2", "fine_tuned_gpt2_GROUP_XX")
        max_length: Maximum sequence length for text generation
        batch_size: Batch size for evaluation
        small: If True, use a smaller subset of the dataset for testing purposes.
    """
    logger = setup_logging(GROUP)
    logger.info("=== STARTING GENERATE AND EVALUATE ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Max length: {max_length}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Small dataset: {small}")
    
    dataset = load_and_split_dataset(small=small)
    test_dataset = dataset['test']

    model = GPT2Model(model_name=model_name, max_length=max_length)

    logger.info(f"Evaluating {model_name} model...")
    print(f"Evaluating {model_name} model...")
    model.evaluate(
            test_dataset=test_dataset,
            batch_size=batch_size,
        )
    logger.info("=== GENERATE AND EVALUATE COMPLETED ===")
    
    
if __name__ == "__main__":
    dataset = load_and_split_dataset(small=True)
    fine_tune(
        model_name="gpt2",
        dataset=dataset,
        max_length=128,
        epochs=1,
        batch_size=4,
        lr=5e-5,
        warmup_steps=50
    )
