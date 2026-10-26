import kagglehub

# Download latest version
path = kagglehub.dataset_download("albatani/leukemiaattri-yolo-train-val-test")

print("Path to dataset files:", path)
