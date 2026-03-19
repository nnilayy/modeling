Everything in this framework is driven by configs/ — configs flow out to each and every task across data, training, evaluation, and deployment.
YAML is the chosen config file format — industry standard across all major ML frameworks, handles deep nesting naturally, supports comments and anchors, and pairs with Pydantic v2 + tyro for type safety and CLI overrides.

LLM evaluation — 4 pillars only:
1. Tool Calling — BFCL V4, NexusBench, Live API-Bench
2. Humanness & Expressiveness — HumT, WritingBench, Chatbot Arena, MT-Bench, WildBench, HEART
3. Hallucination & Factuality — TruthfulQA, FActScore, HaluEval, HalluLens
4. Long Context — NoLiMa, RULER/RULERv2, LongBench v2

Target models (Qwen 3.5 + Gemma 3 only — multilingual requirement: 100+ languages):

< 1B:
- Gemma 3-270M (Dense, Google)
- Qwen 3.5-0.8B (Dense, Alibaba)

1B - 10B:
- Gemma 3-1B (Dense, Google)
- Qwen 3.5-2B (Dense, Alibaba)
- Gemma 3-4B (Dense, Google)
- Qwen 3.5-4B (Dense, Alibaba)
- Qwen 3.5-9B (Dense, Alibaba)

10B - 100B:
- Gemma 3-12B (Dense, Google)
- Gemma 3-27B (Dense, Google)
- Qwen 3.5-27B (Dense, Alibaba)
- Qwen 3.5-35B-A3B (MoE 35B total, 3B active, Alibaba)

> 100B:
- Qwen 3.5-122B-A10B (MoE 122B total, 10B active, Alibaba)
- Qwen 3.5-397B-A17B (MoE 397B total, 17B active, Alibaba)

Inference engines:
- HuggingFace Transformers — development server (loading, testing, forward checks)
- vLLM — production multi-user serving
- SGLang — multi-turn chat, agentic, tool calling
- TensorRT-LLM — max NVIDIA performance
- LMDeploy — quantized models, lowest latency
- TGI — simple Docker deploy, HF integration
