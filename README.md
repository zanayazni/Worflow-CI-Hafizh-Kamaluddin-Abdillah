# Workflow CI - MLflow + RandomForest (Seattle Weather)

Repo ini berisi contoh workflow CI menggunakan GitHub Actions untuk:

- Melatih model machine learning (RandomForest) saat workflow ter-trigger
- Menyimpan artefak hasil training ke repo yang sama (commit balik dari CI)
- Membuat dan push Docker image ke Docker Hub menggunakan `mlflow models build-docker`

## Struktur proyek

- `MLProject/modelling.py`: script training + evaluasi + logging ke MLflow
- `MLProject/seattle-weather_preprocessing.csv`: dataset (hasil preprocessing)
- `MLProject/conda.yaml`: environment conda untuk menjalankan training
- `MLProject/MLProject`: file MLflow Project (entry point menjalankan `python modelling.py`)
- `.github/workflows/mlflow-ci.yml`: workflow CI

Artefak yang dihasilkan training:

- `MLProject/classification_report_rf_ci.txt`
- `MLProject/confusion_matrix.txt`
- `MLProject/last_run_id.txt`

