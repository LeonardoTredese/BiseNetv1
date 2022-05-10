import argparse
from torch import randn
from thop import profile
from thop import clever_format
from model.build_BiSeNet import BiSeNet
from model.discriminator import FCDiscriminator

def complexity(model, input):
    macs, params = profile(model, inputs=(input, ), verbose=False)
    # FLOPs is nearly two times as MACs (multiply–accumulate operation = a + (b x c))
    flops = 2*macs
    return macs, flops, params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="discriminator", help='The model to which the complexity is calculated')
    parser.add_argument('--input_height', type=int, default=512, help='Height of input to network')
    parser.add_argument('--input_width', type=int, default=1024, help='Width of input to network')
    parser.add_argument('--input_channels', type=int, default=19, help='Number of channels of input to network')
    parser.add_argument('--num_classes', type=int, default=19, help='Number of object classes')
    parser.add_argument('--context_path', type=str, default="resnet101",help='The context path model for bisenet: either resnet18, resnet101.')

    args = parser.parse_args()

    if args.model == "bisenet":
        model_name = "BiSeNet"
        model = BiSeNet(args.num_classes, args.context_path)
    else:
        model_name = "Discriminator"
        model = FCDiscriminator(args.num_classes)
    
    input = randn(1, args.input_channels, args.input_width, args.input_height)
    
    macs, flops, params = complexity(model, input)
    macs, flops, params = clever_format([macs, flops, params], "%.3f")

    print(f"COMPLEXITY OF {model_name}")
    print(f"MACs: {macs}")
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")

if __name__ == '__main__':
    main()