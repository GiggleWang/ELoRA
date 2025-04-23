"""
这个示例展示了如何使用vLLM进行离线推理，不使用LoRA功能。

需要HuggingFace凭证以访问Llama2。
"""

from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams


def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    """创建测试提示列表及其采样参数。
    
    包含6个请求，全部使用基础模型，没有LoRA。
    使用更简单的采样参数，避免批处理大小不匹配问题。
    """
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0, max_tokens=128)),
        ("To be or not to be,",
         SamplingParams(temperature=0.8, top_k=5, max_tokens=128)),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
            SamplingParams(temperature=0.0, max_tokens=128)),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
            SamplingParams(temperature=0.0, max_tokens=128)),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
            SamplingParams(temperature=0.0, max_tokens=128)),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
            SamplingParams(temperature=0.0, max_tokens=128)),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """持续处理提示列表并处理输出。"""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)


def initialize_engine() -> LLMEngine:
    """初始化LLMEngine。"""
    engine_args = EngineArgs(model="meta-llama/Llama-2-7b-hf",
                             enable_prefix_caching=True,
                             max_num_seqs=100)  # 减小最大序列数
    return LLMEngine.from_engine_args(engine_args)


def main():
    """设置并运行提示处理的主函数。"""
    engine = initialize_engine()
    test_prompts = create_test_prompts()
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    main()
