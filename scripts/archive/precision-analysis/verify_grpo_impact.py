"""
Simulate how logprobs error compounds in GRPO policy gradient.

In GRPO, the key quantity is:
  ratio = exp(sum(logp_new[t] - logp_old[t]) for t in response_tokens)

If each logp has error e_t, the ratio becomes:
  ratio_noisy = exp(sum((logp_new[t] + e_t) - logp_old[t]))
             = ratio_true * exp(sum(e_t))

For a response of N tokens, the cumulative error is sum(e_t).
If errors are iid with mean mu and std sigma, then:
  E[sum(e_t)] = N * mu
  Std[sum(e_t)] = sqrt(N) * sigma

This script quantifies this for realistic response lengths.
"""

import torch
import torch.nn.functional as F

torch.manual_seed(42)

VOCAB_SIZE = 151936
N_SAMPLES = 1000  # number of random logit vectors to sample

# Compute per-token logprob error distribution
errors = []
for _ in range(N_SAMPLES):
    logits_fp32 = torch.randn(1, VOCAB_SIZE, dtype=torch.float32)
    label = torch.randint(0, VOCAB_SIZE, (1,))

    # fp32 reference (H200-like)
    lp_fp32 = F.log_softmax(logits_fp32, dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1)

    # bf16 path (NPU-like)
    logits_bf16 = logits_fp32.to(torch.bfloat16)
    lp_bf16 = F.log_softmax(logits_bf16, dim=-1).gather(-1, label.unsqueeze(-1)).squeeze(-1).float()

    errors.append((lp_bf16 - lp_fp32).item())

errors = torch.tensor(errors)
mu = errors.mean().item()
sigma = errors.std().item()
print(f"Per-token logprob error (bf16 vs fp32):")
print(f"  mean: {mu:.6f}")
print(f"  std:  {sigma:.6f}")
print(f"  max:  {errors.abs().max().item():.6f}")
print()

# Simulate cumulative error for different response lengths
print(f"{'Response tokens':>16} | {'E[cum_error]':>12} | {'Std[cum_error]':>14} | {'Ratio perturbation (1σ)':>24} | {'Ratio perturbation (2σ)':>24}")
print("-" * 100)

for N in [100, 500, 1000, 2000, 4000, 8000]:
    cum_mean = N * mu
    cum_std = (N ** 0.5) * sigma
    # The policy ratio is perturbed by exp(cum_error)
    ratio_1sigma = torch.exp(torch.tensor(abs(cum_mean) + cum_std)).item()
    ratio_2sigma = torch.exp(torch.tensor(abs(cum_mean) + 2 * cum_std)).item()
    print(f"{N:>16} | {cum_mean:>12.4f} | {cum_std:>14.4f} | {ratio_1sigma:>24.4f} | {ratio_2sigma:>24.4f}")

print()
print("Interpretation:")
print("  - GRPO clips the ratio to [1-eps_low, 1+eps_high] = [0.9997, 1.0004]")
print("  - If the ratio perturbation exceeds this clip range, the gradient")
print("    direction can flip or be zeroed out incorrectly.")
print("  - For 4000+ token responses, the cumulative error is large enough")
print("    to cause significant gradient noise.")
print()

# Now simulate the actual GRPO loss impact
print("=" * 70)
print("GRPO loss simulation with logprob noise")
print("=" * 70)

# Simulate a mini-batch of 8 rollouts
N_ROLLOUTS = 8
RESPONSE_LEN = 2000  # typical response length

# Generate "true" logprobs and advantages
true_logp_new = torch.randn(N_ROLLOUTS, RESPONSE_LEN) * 0.1 - 5.0  # typical logprob range
true_logp_old = true_logp_new + torch.randn(N_ROLLOUTS, RESPONSE_LEN) * 0.01  # small diff
advantages = torch.randn(N_ROLLOUTS)  # GRPO advantages
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalized

# Compute true GRPO loss (simplified)
true_ratio = torch.exp((true_logp_new - true_logp_old).sum(dim=-1))
true_clipped = torch.clamp(true_ratio, 0.9997, 1.0004)
true_loss = -(torch.min(true_ratio * advantages, true_clipped * advantages)).mean()

# Add bf16 noise to logprobs
noise = torch.normal(mean=mu, std=sigma, size=(N_ROLLOUTS, RESPONSE_LEN))
noisy_logp_new = true_logp_new + noise

noisy_ratio = torch.exp((noisy_logp_new - true_logp_old).sum(dim=-1))
noisy_clipped = torch.clamp(noisy_ratio, 0.9997, 1.0004)
noisy_loss = -(torch.min(noisy_ratio * advantages, noisy_clipped * advantages)).mean()

print(f"\nTrue GRPO loss:  {true_loss.item():.6f}")
print(f"Noisy GRPO loss: {noisy_loss.item():.6f}")
print(f"Loss difference: {abs(noisy_loss.item() - true_loss.item()):.6f}")
print(f"\nTrue ratios:  {true_ratio.tolist()}")
print(f"Noisy ratios: {noisy_ratio.tolist()}")
print(f"\nNote: With eps_clip=[0.0003, 0.0004], the clip range is [0.9997, 1.0004].")
print(f"Any ratio outside this range gets clipped. The bf16 noise pushes")
print(f"ratios far outside this range, making almost all samples hit the clip.")
print(f"This effectively destroys the gradient signal for most tokens.")
