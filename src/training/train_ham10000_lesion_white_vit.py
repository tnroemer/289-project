from training.ham10000_training import train_ham10000_model


if __name__ == "__main__":
    train_ham10000_model(model_type="vit", image_source="lesion_white")
