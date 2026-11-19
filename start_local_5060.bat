@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set ACESTEP_GENERATION_ONLY=true
set HF_HOME=%~dp0.hf-cache
set HF_MODULES_CACHE=%~dp0.hf-cache\modules
set ACESTEP_INIT_LLM=true
set ACESTEP_BATCH_SIZE=1
set ACESTEP_COMPILE_MODEL=false
"%~dp0.venv\Scripts\python.exe" -X utf8 -m acestep.acestep_v15_pipeline --server-name 127.0.0.1 --port 7860 --batch_size 1 --language zh-TW --config_path acestep-v15-turbo --init_service true --init_llm true --lm_model_path acestep-5Hz-lm-1.7B --backend pt --offload_to_cpu true --offload_dit_to_cpu true --quantization none --use_flash_attention false
pause

