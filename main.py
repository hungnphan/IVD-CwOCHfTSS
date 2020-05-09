from camera.camera import Camera
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a dataset')
    parser.add_argument(
        '--data',
        metavar='data',
        default='PVD01',
        help='Input the dataset codename')
    args = parser.parse_args()
    
    # print(type(args.data))
    # print(args.data)

    process_camera = Camera(args.data)
    

