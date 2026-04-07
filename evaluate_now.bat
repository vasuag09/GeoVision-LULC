@echo off
call conda activate geovision
cd /d %~dp0

echo.
echo ============================================================
echo  Evaluation - SegFormer and DeepLabV3+
echo ============================================================

echo.
echo [1/2] Evaluating SegFormer...
python scripts/evaluate.py --configs_dir configs/rtx4070_segformer.yaml --output_dir results
if errorlevel 1 echo WARNING: SegFormer evaluation failed, continuing...

echo.
echo [2/2] Evaluating DeepLabV3+...
python scripts/evaluate.py --configs_dir configs/rtx4070_deeplabv3plus.yaml --output_dir results
if errorlevel 1 echo WARNING: DeepLabV3+ evaluation failed, continuing...

echo.
echo ============================================================
echo  GradCAM Explainability
echo ============================================================

echo.
echo [1/2] GradCAM - SegFormer...
python scripts/explainability.py --config configs/rtx4070_segformer.yaml --weights checkpoints_segformer/best_model_segformer.pt --output_dir results/explainability
if errorlevel 1 echo WARNING: SegFormer GradCAM failed, continuing...

echo.
echo [2/2] GradCAM - DeepLabV3+...
python scripts/explainability.py --config configs/rtx4070_deeplabv3plus.yaml --weights "checkpoints_deeplabv3plus/best_model_deeplabv3+.pt" --output_dir results/explainability
if errorlevel 1 echo WARNING: DeepLabV3+ GradCAM failed, continuing...

echo.
echo ============================================================
echo  Exporting Research Artifacts
echo ============================================================
python scripts/export_research_artifacts.py
if errorlevel 1 echo WARNING: Research artifact export failed, continuing...

echo.
echo ============================================================
echo  Pushing Results to GitHub
echo ============================================================
git add results/ checkpoints_segformer/ checkpoints_deeplabv3plus/
git commit -m "Add SegFormer and DeepLabV3+ results, evaluation metrics and GradCAM visualizations"
git push origin main

echo.
echo ============================================================
echo  All done! Results saved to: results/
echo ============================================================
pause
