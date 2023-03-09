from transformers import XLNetTokenizer, XLNetModel

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

tokens = tokenizer.tokenize("Hello, my dog is cute")

print(tokens)