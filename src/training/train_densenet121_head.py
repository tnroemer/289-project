from training.ham10000_training import train_ham10000_model


if __name__ == "__main__":
    for image_source in ("full_image", "lesion_white"):
        train_ham10000_model("densenet121_head", image_source)
