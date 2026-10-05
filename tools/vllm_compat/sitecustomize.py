"""Import-time shim for the CGFM VLM server ONLY (added to PYTHONPATH by
launch_cgfm_vlm_server.sh; the Python env that provides vLLM is not modified).

vLLM 0.7.2 imports `Qwen2_5_VLImageProcessor` from transformers.models.qwen2_5_vl,
a name that existed only in transformers 4.49 dev builds; the 4.49.0 release in
the vLLM env uses Qwen2VLImageProcessor for Qwen2.5-VL instead (same processor).
Alias the old name so vLLM (and its model-inspection subprocesses, which inherit
PYTHONPATH) can load Qwen2.5-VL.
"""
try:
    import transformers.models.qwen2_5_vl as _q25
    if not hasattr(_q25, "Qwen2_5_VLImageProcessor"):
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        _q25.Qwen2_5_VLImageProcessor = Qwen2VLImageProcessor
except Exception:  # never break interpreter start-up
    pass
