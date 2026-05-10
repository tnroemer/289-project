from data_setup.create_ham10000_lesion_white_data import main as create_ham10000_lesion_white_data
from data_setup.create_pad_ufes20_lesion_white_data import main as create_pad_ufes20_lesion_white_data


if __name__ == "__main__":
    print("Creating HAM10000 lesion-white data")
    create_ham10000_lesion_white_data()
    print("Creating PAD-UFES-20 lesion-white data")
    create_pad_ufes20_lesion_white_data()
