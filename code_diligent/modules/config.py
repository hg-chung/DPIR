
import argparse


def config_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--conf', type=str)
    parser.add_argument("--splatting_r", type=float, default=0.015, help='the initial radius for the splatting')
    parser.add_argument("--raster_n", type=int, default=15, help='the point number for soft raterization')
    parser.add_argument("--refine_n", type=int, default=5, help='the point number for soft raterization') # 2, 5
    parser.add_argument("--data_r", type=float, default=0.012, help='the point number for soft raterization') # epoch 13, 20

    parser.add_argument("--lr1", type=float, default=1e-4, help='the learning rate for the point position') # 1
    parser.add_argument("--lr2", type=float, default=5e-4, help='the learning rate for the gemoetry and brdf')# 5
    parser.add_argument("--lrexp", type=float, default=0.93, help='the coefficient for the exponential lr decay') # 0.85
    parser.add_argument("--img_s", type=int, default=400, help='the coefficient for the total variance loss')
    parser.add_argument("--memitem", type=object, default=None, help='the coefficient for the total variance loss')
    parser.add_argument("--testitem", type=object, default=None, help='the coefficient for the total variance loss')
    


    parser.add_argument("--expname", type=str, default='pcdata',
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='../logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='/tigress/qz9238/workspace/workspace/data/nerf/nerf_synthetic/',
                        help='input data directory')
    parser.add_argument("--dataname", type=str, default='hotdog',
                        help='dataset name')
    parser.add_argument("--half_res", default=True,
                        help='load blender synthetic data at 400x400 instead of 800x800')

    return parser



