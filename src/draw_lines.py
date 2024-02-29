from pathlib import Path
from hough import HoughTransform
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description='Script to find and draw lines on image.')
    parser.add_argument("--image", "-i", type=Path, required=True,
                        help="Path to image in PIL accessible format (PNG, JPEG, TIFF, etc.).")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Path to save given image with drawn lines on it. \
                            If None, it only finds lines.")
    parser.add_argument("--quantile", "-q", type=float, default=0.99,
                        help="Float number in [0, 1]: Quantile of the brightest pixels in Hough transform \
                            to consider them lines. It determines threshold to binarize Hough Transform. \
                            Default value is 0.99.")
    parser.add_argument("--min_cluster_size", "-mcs", type=float, default=0.0001,
                        help="If >= 1, it is min number of pixels in a cluster \
                            (of pixels in binarized Hough transform) to select line in it. \
                            If in [0, 1) it is considered as min percentage of pixels in the image. \
                            Default value is 30 (pixels).")
    parser.add_argument("--vertical", "-vx", default=True,
                        help="Flag to find lines connected to vertical borders. \
                            True by default. You can disable it to speed up computation.")
    parser.add_argument("--horizontal", "-hx", default=True,
                        help="Flag to find lines connected to horizontal borders. \
                            True by default. You can disable it to speed up computation.")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Max limit on resulted lines, so it shows <= 'limit' most obvious lines. \
                            If None, unlimited.")
    parser.add_argument("--color", "-c", default='red',
                        help="Name of color to draw lines. Default value is 'red'. \
                            See https://pillow.readthedocs.io/en/stable/reference/ImageColor.html#color-names")
    parser.add_argument("--thickness", "-th", type=int, default=5,
                        help="Thickness of lines in pixles. Default value is 5.")
    parser.add_argument("--gradient", "-g", action="store_true",
                        help="Flag to say that input image is already a gradient map. \
                            If True it provides the image in Hough Transform without preparations.")
    parser.add_argument("--edge_detection_method", "-m", 
                        choices=['laplacian', 'sobel'],
                        default='laplacian',
                        help="If gradient flag is False, it transforms image to edge map \
                            before Hough Transform using this method.")
    args = parser.parse_args()
    if not args.gradient:
        print("Computing edge map of the image...")
    # Find lines
    ht = HoughTransform(
        args.image, 
        compute_edges=None if args.gradient else args.edge_detection_method
    )
    
    print("Searching for lines...")
    
    lines = ht.find_lines(
        horizontal=args.horizontal,
        vertical=args.vertical,
        quantile=args.quantile,
        min_cluster_size=args.min_cluster_size,
        max_lines=args.limit
    )
    
    print(f"Found {len(lines)} lines: ")
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line
        print(f"{i}.\tp1=({x1:0.2f}, {y1:0.2f}),\tp2=({x2:0.2f}, {y2:0.2f})")
    
    print("Drawing lines...")
    output_img = ht.draw_lines(
        lines=lines, 
        color=args.color,
        thickness=args.thickness
    )
    output_img.save(args.output)
    print(f"Image with lines was saved to '{args.output}'.")

    
if __name__ == "__main__":
    main()