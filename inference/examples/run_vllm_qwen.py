#!/usr/bin/env python3

# import torch
# from qwen_asr import Qwen3ASRModel

# if __name__ == '__main__':
#     model = Qwen3ASRModel.LLM(
#         model="Qwen/Qwen3-ASR-1.7B",
#         gpu_memory_utilization=0.7,
#         max_inference_batch_size=128, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
#         max_new_tokens=4096, # Maximum number of tokens to generate. Set a larger value for long audio input.
#         forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
#         forced_aligner_kwargs=dict(
#             dtype=torch.bfloat16,
#             device_map="cuda:0",
#             # attn_implementation="flash_attention_2",
#         ),
#     )

#     results = model.transcribe(
#         audio=[
#         "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
#         "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
#         ],
#         language=["Chinese", "English"], # can also be set to None for automatic language detection
#         return_time_stamps=True,
#     )

#     for r in results:
#         print(r.language, r.text, r.time_stamps[0])



from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
import base64
import requests

# Initialize the LLM
llm = LLM(
    model="Qwen/Qwen3-ASR-1.7B"
)

# Load audio
audio_asset = AudioAsset("winning_call")

# Create conversation with audio content
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": audio_asset.url}
            }
        ]
    }
]

sampling_params = SamplingParams(temperature=0.01, max_tokens=256)

# Run inference using .chat()
outputs = llm.chat(conversation, sampling_params=sampling_params)
# print(outputs[0].outputs[0].text)
print(outputs[0])
