from utils import get_config, check_folder, mkdir_output_test,get_file_list
from data_processing.data_processing import load_data_testing, save_images
import argparse, os
from inference import Inference
parses = argparse.ArgumentParser()
parses.add_argument('--config',type=str, default='configs/test.yaml', help='Path to the configs file.')
opts = parses.parse_args()

def predict_test(inference, result_dir, style_dir, content_dir):
    list_path_content = get_file_list(content_dir)
    list_path_style = get_file_list(style_dir)

    dir_out_img = os.path.join(result_dir, 'image')
    check_folder(dir_out_img)

    for style_file in list_path_style:
        style_prefix, _ = os.path.splitext(style_file)
        style_prefix = os.path.basename(style_prefix)
        style_img = load_data_testing(style_file)

        for content_file in list_path_content:
            content_prefix, _ = os.path.splitext(content_file)
            content_prefix = os.path.basename(content_prefix)
            content_img = load_data_testing(content_file)

            print("Processing: size_content: (%d,%d)   size_style: (%d,%d)" % (
                content_img.shape[1], content_img.shape[2], style_img.shape[1], style_img.shape[2]))
            print(style_file)
            print(content_file)

            results = inference.predict(content_img, style_img)

            img_fakes = results['img_fakes'][0]
            for i in range(len(img_fakes)):
                image_path = os.path.join(dir_out_img, '{}-{}-{}.jpg'.format(style_prefix, content_prefix,str(i)))
                save_images(img_fakes[i], [1, 1], image_path)


def main():
    args = get_config(opts.config)
    if args is None:
        exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['GPU_ID'])
    style_dir   = args['data']['dir_style']
    content_dir = args['data']['dir_content']
    result_dir = mkdir_output_test(args)

    inference = Inference(args)

    predict_test(inference, result_dir, style_dir, content_dir)

if __name__ == '__main__':
    main()