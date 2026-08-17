import unittest
import torch
from src.tokenizer import CharTokenizer
from src.model import GPT, GPTConfig
from src.utils import generate_text

class TestLLM(unittest.TestCase):
    def setUp(self):
        self.text = "Hello world! This is a test dataset for building a basic LLM from scratch."
        self.tokenizer = CharTokenizer(self.text)
        self.config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0
        )
        self.model = GPT(self.config)

    def test_tokenizer(self):
        encoded = self.tokenizer.encode("Hello")
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual("Hello", decoded)

    def test_forward_pass_shape_and_loss(self):
        batch_size = 4
        seq_len = 8
        dummy_input = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        dummy_targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        logits, loss = self.model(dummy_input, dummy_targets)

        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0)

    def test_generation(self):
        prompt = "Hello"
        out = generate_text(self.model, self.tokenizer, prompt=prompt, max_new_tokens=10, temperature=0.7)
        self.assertTrue(out.startswith("Hello"))
        self.assertTrue(len(out) > len(prompt))

if __name__ == "__main__":
    unittest.main()
