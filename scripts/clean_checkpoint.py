import argparse
import torch

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, default='input')
    parser.add_argument('output', type=str, default='output')

    args = parser.parse_args()

    state = torch.load(args.input, map_location='cpu')

    clean_state = state['model']
    torch.save(clean_state, args.output)