import random
import argparse  
import numpy as np 
import torch


def build_default_options():
    parser = argparse.ArgumentParser(description="Arguments for training model.")

    parser.add_argument('--seed', default=5, type=int, help='Random seed for reproducibility')

    # dataset
    parser.add_argument('--root_path', type=str, default='/home/codebase/Yinmi/lora-my/DATA/', help='Path to the dataset')
    parser.add_argument('--train_dataset', type=str, default='mixed_center_oct', help='Train Dataset name')
    parser.add_argument('--test_dataset_a', type=str, default='xiangya_oct', help='Test Dataset name')
    parser.add_argument('--test_dataset_b', type=str, default='huaxi_oct', help='Test Dataset name')

    parser.add_argument('--shots', default=32, type=int, help='Number of shots for few-shot learning')

    # model
    parser.add_argument('--model_name', default='openai/clip-vit-base-patch16', type=str, help='Backbone model for CLIP')
    parser.add_argument('--peft_type', default='lora', type=str, help="['LoRA','loha','LoKr','AdaLoRA'],type of adapter")
    
    parser.add_argument("--use_text_peft",action="store_true", help="Whether to apply PEFT to the text encoder.")
    parser.add_argument("--use_vision_peft",action="store_true", help="Whether to apply PEFT to the vision encoder.")
    

    # parser.add_argument('--vision_peft_type', type=str, default='lora')
    # parser.add_argument('--text_peft_type', type=str, default='none')
    # parser.add_argument('--vision_lora_config', type=dict, default={'r': 4, 'lora_alpha': 2 ,"gradual_step": 12})
    parser.add_argument('--vision_lora_config', type=str, default='{"r":8, "lora_alpha":4, "gradual_step":12}')
    parser.add_argument('--text_lora_config', type=dict, default={'r': 2, 'lora_alpha': 16})

    # train
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--n_iters', default=10, type=int, help='Number of training iterations')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')

    parser.add_argument('--seg_path', default="/home/codebase/Yinmi/vlm_oct/segmentation/Unet/unet_best_model_fold0.pth", type=str, help='segmentation checkpoint path')

    parser.add_argument('--proto_alpha', default=0.5, type=float, help='proto_weight')

    # parser.add_argument('--log_to_file', action='store_true', help='Save log to file')

    # parser.add_argument('--output_dir', default="/home/codebase/Yinmi/vlm_oct/output_info_dir", type=str, help='output info dir')
    
    
    return parser.parse_args()

    
def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True