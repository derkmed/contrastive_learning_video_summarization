import argparse
import glob
import torch
import clip
import tqdm
import os


def extract_id(path):
    return os.path.basename(path).split('_')[0]


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = clip.load(args.model, device=device)
    for p in tqdm.tqdm(
            glob.glob(f'{args.caption_dir}/*_captions.pth')):
        file_id = extract_id(p)
        captions = torch.load(p)
        captions = list(map(lambda x: x['sentence'], captions))
        text = clip.tokenize(captions).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            torch.save(text_features,
                       f'{args.output_dir}/{file_id}_clip_embedding.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Features')
    parser.add_argument('--caption_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model', required=True, choices=[
        'RN50',
        'RN101',
        'RN50x4',
        'RN50x16',
        'RN50x64',
        'ViT-B/32',
        'ViT-B/16',
        'ViT-L/14',
        'ViT-L/14@336px',
    ])
    args = parser.parse_args()
    main(args)

