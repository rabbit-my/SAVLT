""" how to run : python main.py  --use_vision_peft"""

from utils.build_options import set_random_seed , build_default_options

from model import load_clip_model, run_lora


from datasets import build_dataset
from datasets.utils import build_data_loader
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

os.environ["TOKENIZERS_PARALLELISM"] = "false"  


def main():
    # load args
    args = build_default_options()
    set_random_seed(args.seed)
    
    print("===== Parsed Arguments =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================")
        
        
    clip_model, preprocess = load_clip_model(args.model_name)
    # print(clip_model)
    print("Preparing dataset.")

    # data argumentation （same with segmentation）
    def get_train_transforms():
        return A.Compose([
        A.RandomResizedCrop(
            size=(224, 224),  
            scale=(0.75, 1.0),
            interpolation=cv2.INTER_CUBIC
        ),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
        
    train_tranform = get_train_transforms()
    mixed_dataset = build_dataset(args.train_dataset, args.root_path, args.shots)
    
    train_loader = build_data_loader(data_source=mixed_dataset.train_x, batch_size=args.batch_size, tfm=train_tranform, is_train=True, shuffle=True, num_workers=8)
    print(train_loader)
    
    mixed_val_loader = build_data_loader(data_source=mixed_dataset.val, batch_size=256, is_train=False, tfm=train_tranform, shuffle=False,  num_workers=8)
    mixed_test_loader = build_data_loader(data_source=mixed_dataset.test, batch_size=256, is_train=False, tfm=train_tranform, shuffle=False,  num_workers=8)
    
    xiangya_dataset = build_dataset(args.test_dataset_a, args.root_path, args.shots)
    huaxi_dataset = build_dataset(args.test_dataset_b, args.root_path, args.shots)
    
    xiangya_test_loader = build_data_loader(data_source=xiangya_dataset.train_x, batch_size=256, is_train=False, tfm=train_tranform, shuffle=False, num_workers=8)
    huaxi_test_loader = build_data_loader(data_source=huaxi_dataset.train_x, batch_size=256, is_train=False, tfm=train_tranform, shuffle=False, num_workers=8)


    run_lora(args, clip_model, preprocess, mixed_dataset, train_loader, mixed_val_loader, mixed_test_loader,xiangya_test_loader,huaxi_test_loader)



if __name__ == '__main__':
    main()
