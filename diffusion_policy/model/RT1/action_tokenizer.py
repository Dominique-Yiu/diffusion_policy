import torch

class RT1ActionTokenizer:
    def __init__(self, 
                 vocab_size: int,
                 low: float=-1.0,
                 high: float=1.0) -> None:
        self.vocab_size = vocab_size
        self.low = low
        self.high = high
        
    def tokenize(self, action):
        action = torch.clamp(action, self.low, self.high)
        token = (action - self.low) / (self.high - self.low)
        token = token * (self.vocab_size - 1)
        token = token.to(torch.int32)

        return token

    def detokenize(self, token):
        action = token.to(torch.float32)
        action = action / (self.vocab_size - 1)
        action = action * (self.high - self.low) + self.low

        return action
    
if __name__=='__main__':
    tokenizer = RT1ActionTokenizer(vocab_size=256)

    action_logits = torch.randn(2, 6, 7, 256)
    action_logits = action_logits[:, -1]
    action_logits = torch.clamp(action_logits, -1.0, 1.0)
    print(f'original action: {action_logits.shape}')

    token = tokenizer.tokenize(action_logits)
    print(f'token: {token.shape}')

    token = torch.argmax(token, dim=-1)
    restore_action = tokenizer.detokenize(token)
    print(f'restore_action: {restore_action}')