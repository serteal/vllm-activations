"""Quick smoke test for the activation collection engine."""

from vllm.activations import ActivationEngine

LAYERS = [0, 8, 15]

engine = ActivationEngine(
    model="Qwen/Qwen3-0.6B",
    layers=LAYERS,
)

# Test 1: collect from text prompts
print("=== Test 1: text prompts ===")
result = engine.collect(prompts=["Hello world", "The quick brown fox jumps"])

for layer in LAYERS:
    tensors = result.activations[layer]
    print(f"  Layer {layer}: {len(tensors)} sequences")
    for i, t in enumerate(tensors):
        print(f"    seq {i}: shape={tuple(t.shape)}, dtype={t.dtype}")

print(f"  num_tokens: {result.num_tokens}")

# Test 2: stack into padded batch
print("\n=== Test 2: stack + mask ===")
stacked = result.stack(LAYERS[1])
mask = result.mask()
print(f"  stacked shape: {tuple(stacked.shape)}")
print(f"  mask shape: {tuple(mask.shape)}")

# Test 3: collect from token IDs
print("\n=== Test 3: token_ids ===")
result2 = engine.collect(token_ids=[[1, 2, 3], [10, 20, 30, 40, 50]])
for layer in LAYERS:
    tensors = result2.activations[layer]
    print(f"  Layer {layer}: seq shapes = {[tuple(t.shape) for t in tensors]}")

print(f"  num_tokens: {result2.num_tokens}")

engine.remove_hooks()
print("\nAll tests passed!")
