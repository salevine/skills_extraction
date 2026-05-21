from skills_extraction.config import PipelineConfig
from skills_extraction.llm_vllm import call_vllm

cfg = PipelineConfig(
    backend='vllm',
    vllm_base_port=8000,
    vllm_num_endpoints=1,
    vllm_max_retries=1,
)
try:
    result = call_vllm(cfg, 'mistralai/Mistral-Nemo-Instruct-2407', 'You are helpful.', 'Say hello', 0.1, 'verifier')
    print('SUCCESS:', result[:100])
except Exception as e:
    print('FAILED:', e)
