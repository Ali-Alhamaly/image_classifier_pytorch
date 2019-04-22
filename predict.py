import argparse
import torch
import util_functions as fnc
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        help="/path/to/image. root directory is ImageClassifier")
    
    parser.add_argument("checkpoint",
                        help="pytorch checkpoint file. root directory is ImageClassifier")
    
    parser.add_argument("--top_k",
                        type = int,
                        help ="top k probs for image class")
    
    parser.add_argument("--category_names",
                        help ="cat_to_name.json file")
    
    parser.add_argument("--gpu",
                        help = "flag used to specify using GPU for training the network",
                        action="store_true")
    
    args = parser.parse_args()
    
    predict_img(args)

def predict_img(args):
    
    checkpoint_path = args.checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else: 
        checkpoint = torch.load(checkpoint_path,map_location='cpu')
    
    
    net_name=checkpoint['pretrained_net']

    if "dense" in net_name: 
        flag =1
    else:
        flag=2

    model = fnc.load_checkpoint(checkpoint_path,flag)

    if args.gpu:
        device = 'cuda'
    else: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    image_path = args.input

    a=fnc.process_image(image_path)
    a=a.to(device)
    model.eval()
    
    if args.top_k is None:
        topk = 1
    else: 
        topk = args.top_k
    
    with torch.no_grad():
        
        top_k_prob , idx=torch.exp(model(a)).topk(k=topk , dim=1)
        idx_to_class=dict(map(reversed, model.class_to_idx.items()))
        top_k_class = []
        for i in idx[0]:
            top_k_class.append(idx_to_class[i.item()])
    
    top_k_prob=top_k_prob.cpu().numpy().ravel()
    
    
    if args.category_names is None:
            print(top_k_class)        
            print(list(map('{:.3f}'.format,top_k_prob)))
    else:
        
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
            
        class_name = [cat_to_name[x] for x in top_k_class]
        print(class_name)        
        print(list(map('{:.3f}'.format,top_k_prob)))
        

    
if __name__ == '__main__':
    main()