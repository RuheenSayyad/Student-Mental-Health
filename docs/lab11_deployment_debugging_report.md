# Lab 11: Deployment Debugging & Production Readiness

## Troubleshooting & Mitigations
1. **Model & Preprocessor Synchronization**:
   - Issue: Training pipeline changes can cause feature mismatches in inference if preprocessors are mismatched.
   - Solution: Model bundle packaging bundles exact preprocessor artifact with model in `deployment/` with `model_metadata.json`.
2. **Path Resolution Across Containers**:
   - Issue: Absolute paths breaking inside container filesystems.
   - Solution: Relative path anchoring via `Path(__file__).resolve().parents[1]`.
3. **Graceful Error Responses**:
   - Issue: Pydantic 422 errors exposing server internals.
   - Solution: Custom validation exception handler formatting error fields cleanly into `ErrorResponse`.
